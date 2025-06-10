import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import agents.domain_planner as domain_planner_mod


def test_domain_planner_sections():
    planner = domain_planner_mod.DomainAwarePlanner()
    output = planner.act("How do plants grow?")
    for section in [
        "Literature Search",
        "Method Design",
        "Expected Results",
        "Analysis Plan",
    ]:
        assert section in output
