# One-time Qdrant collection setup can be extended here if needed.
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from api import settings

def ensure_collection():
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    if settings.COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    return client
