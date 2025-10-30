from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import List
from .agent import run_agent

class State(BaseModel):
    question: str
    plan: str = ""
    research_notes: List[str] = []
    final: dict | None = None

def planner(state: State) -> State:
    state.plan = f"Plan: search 3 sources; summarize; compile JSON."
    return state

def researcher(state: State) -> State:
    # You can call run_agent for sub-questions or custom tool chains
    notes = []
    for subq in [state.question, state.question + " key stats", state.question + " risks"]:
        out = run_agent({"question": subq, "context": []})
        notes.append(out.summary_md)
    state.research_notes = notes
    return state

def reporter(state: State) -> State:
    # Compile notes via main agent once more
    joined = "\n\n".join(state.research_notes)
    out = run_agent({"question": f"Summarize and structure:\n{joined}" , "context": []})
    state.final = out.model_dump()
    return state

def build_graph():
    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("reporter", reporter)
    g.set_entry_point("planner")
    g.add_edge("planner","researcher")
    g.add_edge("researcher","reporter")
    g.add_edge("reporter", END)
    return g.compile()
