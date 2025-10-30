from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from app.agent import run_agent
from app.schemas import AgentInput, AgentOutput
from app.outputs import md_to_html, md_to_pdf
from app.multiagent import build_graph

from dotenv import load_dotenv, find_dotenv
# Search upwards from current working dir and load the first .env found
load_dotenv(find_dotenv(usecwd=True), override=False)


app = FastAPI(title="InsightBridge Agent API")
graph = build_graph()

@app.post("/agent", response_model=AgentOutput)
def single_agent(inp: AgentInput):
    try:
        return run_agent(inp.model_dump())
    except Exception as e:
        raise HTTPException(500, str(e))

class MultiInput(BaseModel):
    question: str

@app.post("/multi", response_model=dict)
def multi_agent(mi: MultiInput):
    state = {"question": mi.question}
    final = graph.invoke(state)
    return final["final"]

@app.post("/agent/pdf")
def agent_pdf(inp: AgentInput):
    out = run_agent(inp.model_dump())
    path = md_to_pdf(out.summary_md, "reports/executive_brief.pdf")
    return FileResponse(path, media_type="application/pdf", filename="executive_brief.pdf")

@app.get("/")
def home():
    return HTMLResponse("<h2>InsightBridge Agent is running</h2>")
