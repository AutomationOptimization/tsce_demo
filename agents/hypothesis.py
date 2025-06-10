from __future__ import annotations

import os
from typing import Union

from tsce_agent_demo.tsce_chat import TSCEReply
from .researcher import Researcher

TERMINATE_TOKEN = "TERMINATE"


def record_agreed_hypothesis(
    sci_view: Union[str, TSCEReply],
    res_view: Union[str, TSCEReply],
    *,
    path: str = "leading_hypothesis.txt",
    researcher: Researcher | None = None,
) -> str | None:
    """Write the agreed hypothesis to ``path`` and return the terminate token.

    Parameters
    ----------
    sci_view : str | TSCEReply
        Hypothesis proposed by the Scientist. ``TSCEReply`` objects are
        automatically converted to their ``content`` strings.
    res_view : str | TSCEReply
        Hypothesis echoed or proposed by the Researcher. ``TSCEReply`` objects
        are also accepted.
    path : str, optional
        File path where the hypothesis should be recorded.
    researcher : Researcher | None, optional
        Researcher instance used to write the file. A new one is created if omitted.

    Returns
    -------
    str | None
        ``TERMINATE_TOKEN`` when the hypotheses match (case-insensitive), ``None`` otherwise.
    """
    # Accept either raw strings or TSCEReply objects for convenience
    sci_text = sci_view.content if isinstance(sci_view, TSCEReply) else sci_view
    res_text = res_view.content if isinstance(res_view, TSCEReply) else res_view

    if sci_text.strip().lower() == res_text.strip().lower():
        agent = researcher or Researcher()
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if os.path.exists(path):
            agent.write_file(path, sci_text)
        else:
            agent.create_file(path, sci_text)
        return TERMINATE_TOKEN
    return None

__all__ = ["record_agreed_hypothesis", "TERMINATE_TOKEN"]
