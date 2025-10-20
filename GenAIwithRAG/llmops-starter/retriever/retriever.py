from typing import List, Dict
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchText, PointStruct
from sentence_transformers import SentenceTransformer
from api import settings

_embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

def embed(texts: List[str]):
    return _embedder.encode(texts, normalize_embeddings=True)

def retrieve_chunks(query: str, top_k: int = 4) -> List[Dict]:
    vec = embed([query])[0]
    res = _client.search(
        collection_name=settings.COLLECTION_NAME,
        query_vector=vec.tolist(),
        limit=top_k
    )
    chunks = []
    for p in res:
        payload = p.payload or {}
        chunks.append({
            "text": payload.get("text",""),
            "source": payload.get("source",""),
            "score": p.score
        })
    return chunks
