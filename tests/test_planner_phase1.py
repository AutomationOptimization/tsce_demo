from tsce_agent_demo.models.research_task import ResearchTask
from agents import planner


def test_bibliography_builds():
    task = ResearchTask(question="graph neural networks for materials science")
    out = planner.plan(task)
    assert len(out.literature) >= 5
