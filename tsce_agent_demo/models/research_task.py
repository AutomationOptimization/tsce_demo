from dataclasses import dataclass
from typing import List

@dataclass
class PaperMeta:
    """Basic information about a research paper."""

    title: str
    url: str
    authors: List[str]
    year: int
    abstract: str
