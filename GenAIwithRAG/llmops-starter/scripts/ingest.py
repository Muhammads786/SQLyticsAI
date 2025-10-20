import os
import uuid
from pathlib import Path
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
#from sentence_transformers import SentenceTransformer

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

    #embedder = SentenceTransformer(EMBED_MODEL_NAME)
    #vecs = embedder.encode(parts, normalize_embeddings=True)

    client_oa = OpenAI(base_url=os.getenv("OLLAMA_OPENAI_BASE_URL","http://localhost:11434/v1"),api_key=os.getenv("OLLAMA_API_KEY","ollama"))
    vecs = client_oa.embeddings.create(model=EMBED_MODEL_NAME, input=parts)    
    vecs = np.array([d.embedding for d in vecs.data], dtype="float32")

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
