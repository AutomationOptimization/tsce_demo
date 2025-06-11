import json
import sys
import types
from pathlib import Path
from freezegun import freeze_time
import pytest

import tsce_agent_demo.__main__ as cli
from tsce_agent_demo.models.research_task import ResearchTask
import tsce_agent_demo.tsce_chat as chat_mod


class DummyChat:
    def __init__(self):
        self.total_tokens = 10
        self.total_cost_usd = 0.01

    def __call__(self, messages):
        return types.SimpleNamespace(content="ok")

    def totals(self):
        return self.total_tokens, self.total_cost_usd


def test_orchestrator_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_mod, "_make_client", lambda: ("dummy", object(), ""))
    monkeypatch.setattr(chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    out_json = tmp_path / "run.json"
    monkeypatch.setattr(sys, "argv", [
        "tsce_agent_demo",
        "--question",
        "Toy harmonic oscillator",
        "--max-cost",
        "0.05",
        "--json-out",
        str(out_json),
    ])
    task = ResearchTask(question="Toy harmonic oscillator")
    with freeze_time("2024-01-01"):
        cli.main()
    data = json.loads(out_json.read_text())
    assert data["status"] == "success"
    summary = Path(data["summary_file"])
    assert summary.exists()
    assert summary.stat().st_size > 300
    fixture = Path("tests/int/fixtures/latest_run.json")
    fixture.write_text(json.dumps(data, indent=2))
