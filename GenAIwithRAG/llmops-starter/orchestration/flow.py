"""
LangGraph flow: retrieve -> generate -> guard -> cite
"""
from typing import Dict, Any
from retriever.retriever import retrieve_chunks
from api import settings
import requests

SYS_PROMPT = "You are a helpful assistant. Use provided context to answer. Cite sources as [#]. If unsure, say you don't know."

def call_llm(prompt: str) -> str:
    url = f"{settings.OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
    payload = {
        "model": "llama3.1",  # vLLM serves the configured model id
        "messages": [{"role": "system", "content": SYS_PROMPT},
                     {"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 512,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def guard_output(text: str) -> str:
    # Minimal placeholder. Add content safety checks here.
    return text

def run_flow(user_msg: str) -> Dict[str, Any]:
    ctx = retrieve_chunks(user_msg, top_k=int(settings.RAG_TOP_K))
    ctx_text = "\n\n".join([f"[{i+1}] " + c["text"] for i, c in enumerate(ctx)])
    prompt = f"Context:\n{ctx_text}\n\nQuestion: {user_msg}\nAnswer with citations."
    raw = call_llm(prompt)
    safe = guard_output(raw)
    return {"answer": safe, "citations": [c.get("source","") for c in ctx]}
