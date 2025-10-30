from pydantic import ValidationError
from langchain_openai import ChatOpenAI
#from loguru import logger
from .schemas import AgentInput, AgentOutput, Finding
from .prompts import SYSTEM_PROMPT, HUMAN_TEMPLATE
from .tools import web_search, fetch_url_text, load_pdf, chunk
from .memory import upsert_docs, search as mem_search
import hashlib, pathlib

llm = ChatOpenAI(model="deepseek-chat", temperature=0)

def _context_to_text(ctx):
    blobs = []
    for item in ctx:
        if item.type == "url":
            text = fetch_url_text(item.uri)
        elif item.type == "pdf":
            text = load_pdf(item.uri)
        else:
            text = item.text or ""
        if text:
            # persist to vector memory
            doc_id = hashlib.md5((item.uri+text[:200]).encode()).hexdigest()
            upsert_docs([{"id":doc_id, "text":text, "meta":{"uri":item.uri}}])
            for c in chunk(text):
                cid = hashlib.md5((c[:50]+doc_id).encode()).hexdigest()
                upsert_docs([{"id":cid, "text":c, "meta":{"uri":item.uri}}])
            blobs.append(f"[{item.type}] {item.uri}\n{text[:4000]}")
    return "\n\n".join(blobs[:5])

def run_agent(payload: dict) -> AgentOutput:
    inp = AgentInput(**payload)
    mem_hits = mem_search(inp.question, k=5)
    mem_snips = "\n".join([h["text"][:500] for h in mem_hits])

    context_bullets = ""
    if inp.context:
        context_bullets = "\n".join([f"- ({c.type}) {c.uri}" for c in inp.context])

    human = HUMAN_TEMPLATE.format(
        question=inp.question,
        context_bullets=context_bullets + ("\n\nMemory hits:\n" + mem_snips if mem_snips else "")
    )

    msgs = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":human}
    ]
    resp = llm.invoke(msgs).content

    # Safe parse via JSON fence marker
    # Expect model to include a ```json block; fallback regex if needed.
    import re, json
    j = {}
    code_blocks = re.findall(r"```json\n(.*?)```", resp, flags=re.S)
    if code_blocks:
        j = json.loads(code_blocks[-1])
    else:
        # fallback: try to extract braces
        brace = re.search(r"\{.*\}\s*$", resp, flags=re.S)
        if brace:
            j = json.loads(brace.group(0))

    try:
        return AgentOutput(**j)
    except ValidationError as e:
        #logger.error(f"Schema mismatch: {e}")
        print(f"Schema mismatch: {e}")
        # minimal salvage
        return AgentOutput(summary_md=resp[:1000], findings=[], citations=[], followups=["Re-run with stricter output parser"])
