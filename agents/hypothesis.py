from __future__ import annotations

from .researcher import Researcher

TERMINATE_TOKEN = "TERMINATE"


def record_agreed_hypothesis(sci_view: str, res_view: str, *, researcher: Researcher | None = None) -> str | None:
    """Write the agreed hypothesis to ``leading_hypothesis.txt`` and return the terminate token.

    Parameters
    ----------
    sci_view : str
        Hypothesis proposed by the Scientist.
    res_view : str
        Hypothesis echoed or proposed by the Researcher.
    researcher : Researcher | None, optional
        Researcher instance used to write the file. A new one is created if omitted.

    Returns
    -------
    str | None
        ``TERMINATE_TOKEN`` when the hypotheses match (case-insensitive), ``None`` otherwise.
    """
    if sci_view.strip().lower() == res_view.strip().lower():
        agent = researcher or Researcher()
        agent.write_file("leading_hypothesis.txt", sci_view)
        return TERMINATE_TOKEN
    return None

__all__ = ["record_agreed_hypothesis", "TERMINATE_TOKEN"]
