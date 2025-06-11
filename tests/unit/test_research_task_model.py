import json
import pytest
from pydantic import ValidationError

from tsce_agent_demo.models.research_task import ResearchTask


def test_schema_validation():
    with pytest.raises(ValidationError):
        ResearchTask()
    task = ResearchTask(question="What is AI?")
    assert task.question == "What is AI?"
    assert task.id is None


def test_json_round_trip():
    task = ResearchTask(question="Where is the moon?")
    data = task.json()
    # ensure valid JSON
    loaded = json.loads(data)
    assert loaded["question"] == "Where is the moon?"
    same = ResearchTask.parse_raw(data)
    assert same == task


