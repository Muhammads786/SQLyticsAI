from fastapi import APIRouter
from pydantic import BaseModel
import requests, os
from api import settings
from orchestration.flow import run_flow

router = APIRouter(prefix="", tags=["chat"])

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
def chat(req: ChatRequest):
    result = run_flow(user_msg=req.message)
    return result
