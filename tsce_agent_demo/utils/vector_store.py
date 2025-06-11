from __future__ import annotations
from typing import Iterable, List

def query(text: str, k: int = 1) -> List[str]:
    """Return ``k`` dummy evidence snippets for ``text``."""
    return [f"evidence for {text}"] * k
