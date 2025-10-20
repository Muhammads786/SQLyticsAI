from fastapi import FastAPI
from api.routers import chat
from api import settings

app = FastAPI(title="LLMOps Starter API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(chat.router)
