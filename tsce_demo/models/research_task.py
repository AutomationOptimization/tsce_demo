from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ResearchTask(BaseModel):
    """Pydantic model representing a single research task."""

    id: Optional[str] = None
    question: str
    literature: Optional[List[str]] = None
    method_plan: Optional[str] = None
    execution_artifacts: Optional[List[str]] = None
    summary: Optional[str] = None

