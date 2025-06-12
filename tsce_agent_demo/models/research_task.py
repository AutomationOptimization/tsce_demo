from dataclasses import dataclass
from typing import Any, List

from pydantic import BaseModel, ConfigDict

@dataclass
class PaperMeta:
    """Basic information about a research paper."""

    title: str
    url: str
    authors: List[str]
    year: int
    abstract: str


class MethodPlan(BaseModel):
    """Structured method plan for an experiment."""

    steps: List[str] = []
    model_config = ConfigDict(extra="allow")


class ResearchTask(BaseModel):
    """Information about a research goal and related plan."""

    question: str
    id: str | None = None
    method_plan: Any | None = None
    model_config = ConfigDict(extra="allow")
