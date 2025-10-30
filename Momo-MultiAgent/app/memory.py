import chromadb, os
from chromadb.utils import embedding_functions

CHROMA_DIR = os.getenv("CHROMA_DIR","./.chroma")
EMB = embedding_functions.DefaultEmbeddingFunction()

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("agent_memory", embedding_function=EMB)

def upsert_docs(docs: list[dict]):
    # docs: [{"id","text","meta":{...}}]
    collection.upsert(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d.get("meta",{}) for d in docs],
    )

def search(query: str, k: int = 5):
    res = collection.query(query_texts=[query], n_results=k)
    hits = []
    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i] or {}
        hits.append({"text": doc, "meta": meta, "id": res["ids"][0][i]})
    return hits
