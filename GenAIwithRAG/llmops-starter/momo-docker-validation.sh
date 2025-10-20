# from project root
docker compose ps
docker compose logs -n 200 api
docker compose exec api python - <<'PY'
import os, requests
print("OPENAI_BASE_URL =", os.getenv("OPENAI_BASE_URL"))
print("Trying /v1/models ...")
try:
    r = requests.get(os.getenv("OPENAI_BASE_URL") + "/models", timeout=5)
    print("Status:", r.status_code, "Body:", r.text[:400])
except Exception as e:
    print("Error:", e)
PY
