# bootstrap_llmops_lab.sh
#!/usr/bin/env bash
set -euo pipefail

# 0) Folders
mkdir -p app scripts rag data .docker .mlflow mlartifacts

# 1) .env
cat > .env << 'EOF'
# ---- Core endpoints ----
OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama                     # placeholder; Ollama ignores but OpenAI client expects it
OLLAMA_MODEL=llama3.1

QDRANT_URL=http://localhost:6333
MLFLOW_TRACKING_URI=http://localhost:5000

# ---- RAG settings ----
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
QDRANT_COLLECTION=docs
TOP_K=4
EOF

# 2) requirements.txt (host FastAPI + RAG)
cat > requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn==0.30.6
python-dotenv==1.0.1
qdrant-client==1.11.3
sentence-transformers==3.0.1
numpy==1.26.4
pydantic==2.8.2
httpx==0.27.2
openai==1.50.2
langchain==0.2.15
langgraph==0.2.39
tiktoken==0.7.0
EOF

# 3) docker-compose.yml (Qdrant + MLflow only)
cat > docker-compose.yml << 'EOF'
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:v1.11.0
    restart: unless-stopped
    ports:
      - "6333:6333"   # REST
      - "6334:6334"   # gRPC
    volumes:
      - ./.docker/qdrant_storage:/qdrant/storage
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:6333/healthz"]
      interval: 5s
      timeout: 2s
      retries: 30

  mlflow:
    image: bitnami/mlflow:latest
    restart: unless-stopped
    environment:
      - MLFLOW_HOST=0.0.0.0
      - MLFLOW_PORT_NUMBER=5000
      - MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    ports:
      - "5000:5000"
    volumes:
      - ./.mlflow:/mlflow
      - ./mlartifacts:/mlflow/artifacts
    healthcheck:
      test: ["CMD", "bash", "-lc", "exec 3<>/dev/tcp/127.0.0.1/5000 && echo -e 'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n' >&3 && grep -q '200 OK' <&3"]
      interval: 5s
      timeout: 3s
      retries: 30
EOF

# 4) Minimal FastAPI app
cat > app/main.py << 'EOF'
import os
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, Body
from pydantic import BaseModel
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchRequest, PointStruct, VectorParams, Distance, FieldCondition, MatchValue

from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

app = FastAPI(title="LLMOps Lab API", version="0.1.0")

# Env
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))

OLLAMA_BASE = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# Clients
qdrant = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL_NAME)
openai_client = OpenAI(base_url=OLLAMA_BASE, api_key=OLLAMA_KEY)

def embed(texts: List[str]):
    return embedder.encode(texts, normalize_embeddings=True).tolist()

@app.get("/health")
def health():
    return {"ok": True}

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    # 1) Embed query
    q_vec = embed([req.question])[0]

    # 2) Search Qdrant
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=q_vec,
        limit=TOP_K
    )

    # 3) Build context + citations
    contexts = []
    citations = []
    for r in results:
        payload = r.payload or {}
        text = payload.get("text", "")
        doc_id = payload.get("doc_id", "unknown")
        contexts.append(text)
        citations.append({"doc_id": doc_id, "score": r.score})

    context_block = "\n\n---\n".join(contexts) if contexts else "No context found."

    # 4) Call Ollama (OpenAI-compatible)
    prompt = f"You are a helpful RAG assistant. Use the context to answer.\n\nContext:\n{context_block}\n\nQuestion: {req.question}\nAnswer with citations as [doc_id]."
    resp = openai_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content
    return {"answer": answer, "citations": citations}
EOF

# 5) Ingestion script
cat > scripts/ingest.py << 'EOF'
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DATA_FILE = Path("data/readme.txt")

def chunks(text: str, max_chars: int = 800):
    buf = []
    cur = 0
    while cur < len(text):
        buf.append(text[cur:cur+max_chars])
        cur += max_chars
    return buf

def ensure_collection(client, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"[ingest] Created collection '{COLLECTION}' (dim={dim})")
    else:
        print(f"[ingest] Collection '{COLLECTION}' exists")

def main():
    assert DATA_FILE.exists(), f"Missing {DATA_FILE}"
    text = DATA_FILE.read_text(encoding="utf-8").strip()
    parts = [p for p in chunks(text) if p.strip()]

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    vecs = embedder.encode(parts, normalize_embeddings=True)

    dim = vecs.shape[1]
    client = QdrantClient(url=QDRANT_URL)

    ensure_collection(client, dim)

    points = []
    for i, (p, v) in enumerate(zip(parts, vecs)):
        pid = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=pid,
                vector=v.tolist(),
                payload={
                    "doc_id": f"readme.txt#{i+1}",
                    "text": p
                }
            )
        )

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"[ingest] Upserted {len(points)} chunks into '{COLLECTION}'")

if __name__ == "__main__":
    main()
EOF

# 6) Tiny LangGraph placeholder (optional; wired later)
cat > rag/graph.py << 'EOF'
# Minimal placeholder to expand later.
# We will wire a RetrieverNode -> LLMNode graph in the subsequent steps.
def build_graph():
    return {"status": "placeholder"}
EOF

# 7) Sample corpus
cat > data/readme.txt << 'EOF'
This is a tiny sample corpus for the LLMOps lab.
It demonstrates end-to-end RAG with Qdrant and an Ollama-served model.
Add more lines here to test retrieval quality and citations.
EOF

# 8) Helpful run hints
cat > RUN.md << 'EOF'
# Run guide (host)
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# Start FastAPI (host):
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Bring up Qdrant + MLflow (Docker):
docker compose up -d

# Verify Qdrant:
curl -s localhost:6333/healthz && echo

# Verify MLflow:
curl -s localhost:5000 | head

# Ingest sample corpus:
python scripts/ingest.py

# Test chat:
curl -s http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"question":"What does this project do?"}' | jq .
EOF

echo "✅ Bootstrap complete."
echo "Next:"
echo "  1) source .venv/bin/activate && pip install -r requirements.txt"
echo "  2) docker compose up -d"
echo "  3) Ensure Ollama has model:   ollama pull llama3.1"
echo "  4) Start API:                 uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo "  5) Ingest:                    python scripts/ingest.py"
echo "  6) Chat:                      curl -s http://localhost:8000/chat -H 'Content-Type: application/json' -d '{\"question\":\"What does this project do?\"}' | jq ."
