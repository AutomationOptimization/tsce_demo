from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Paper:
    """Lightweight representation of a publication."""

    title: str
    year: int


@dataclass
class ResearchTask:
    """Container for a research question and gathered literature."""

    question: str
    literature: List[Paper] = field(default_factory=list)

@dataclass
class PaperMeta:
    """Basic information about a research paper."""

    title: str
    url: str
    authors: List[str]
    year: int
    abstract: str
