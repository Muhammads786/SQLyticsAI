import os
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, Body
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchRequest, PointStruct, VectorParams, Distance, FieldCondition, MatchValue

#from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

app = FastAPI(title="LLMOps Lab API", version="0.1.0")

# Env
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
#EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_MODEL_NAME = "nomic-embed-text"
TOP_K = int(os.getenv("TOP_K", "4"))

OLLAMA_BASE = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# Clients
qdrant = QdrantClient(url=QDRANT_URL)
#embedder = SentenceTransformer(EMBED_MODEL_NAME)


openai_client = OpenAI(base_url=OLLAMA_BASE, api_key=OLLAMA_KEY)

#def embed(texts: List[str]):
#    return embedder.encode(texts, normalize_embeddings=True).tolist()

def embed(texts: list[str]):
    resp = openai_client.embeddings.create(model=EMBED_MODEL_NAME, input=texts)
    return [d.embedding for d in resp.data]

@app.get("/",include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

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
        max_tokens=256,                     # limit generation length
        extra_body={
            "options": {
                "num_thread":  (os.cpu_count() or 4),   # or a bit less than cores to reduce contention
                "num_ctx": 1024,                        # smaller context = faster
                "num_batch": 64,                        # tune; smaller reduces peak RAM
                "keep_alive": "30m"
            }
        }
    )

    answer = resp.choices[0].message.content
    return {"answer": answer, "citations": citations}
