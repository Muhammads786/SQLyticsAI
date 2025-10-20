docker compose exec api python - <<'PY'
from retriever.qdrant_setup import ensure_collection
c = ensure_collection()
print([c.name for c in c.get_collections().collections])
PY
