from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel

class MethodPlan(BaseModel):
    """Simple container for a method plan."""

    steps: List[str]

class ResearchTask(BaseModel):
    """Representation of a research question and its plan."""

    question: str
    method_plan: Optional[MethodPlan] = None
