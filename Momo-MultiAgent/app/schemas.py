from pydantic import BaseModel, Field
from typing import List, Optional

class RetrievalItem(BaseModel):
    uri: str
    type: str = Field(..., description="url|pdf|text")
    text: Optional[str] = None

class AgentInput(BaseModel):
    question: str
    context: List[RetrievalItem] = []
    user_id: Optional[str] = None
    require_citations: bool = True

class Finding(BaseModel):
    claim: str
    evidence: List[str]
    confidence: float = Field(ge=0, le=1)

class AgentOutput(BaseModel):
    summary_md: str
    findings: List[Finding]
    citations: List[str]
    followups: List[str]
