import json
import sys
import types
from pathlib import Path
import pytest

import tsce_demo.__main__ as cli
from tsce_demo.utils import result_aggregator as agg


class DummyOrchestrator:
    def __init__(self, *a, **k):
        self.run_id = "id"
        self.output_dir = k.get("output_dir", "tmp")
        self.chat = types.SimpleNamespace(total_tokens=10, total_cost_usd=0.1, totals=lambda: (10, 0.1))

    def run(self):
        pass


def _patch(monkeypatch, tmp_path, cost):
    orch = DummyOrchestrator(output_dir=str(tmp_path))
    orch.chat.total_cost_usd = cost
    monkeypatch.setattr(cli, "Orchestrator", lambda *a, **k: orch)
    monkeypatch.setattr(agg, "create_summary", lambda *a, **k: Path(tmp_path/"summary.md").write_text("hi") or Path(tmp_path/"summary.md"))
    return orch


def test_aborts_on_overrun(monkeypatch, tmp_path, capsys):
    _patch(monkeypatch, tmp_path, 0.2)
    monkeypatch.setattr(sys, "argv", ["tsce_demo", "--question", "Q", "--max-cost", "0.05"])
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out.splitlines()[-1]
    data = json.loads(out)
    assert data["status"] == "failure"
    assert data["reason"] == "Budget exceeded"


def test_success_json(monkeypatch, tmp_path, capsys):
    orch = _patch(monkeypatch, tmp_path, 0.01)
    monkeypatch.setattr(sys, "argv", ["tsce_demo", "--question", "Q", "--max-cost", "0.05"])
    cli.main()
    out = capsys.readouterr().out.splitlines()[-1]
    data = json.loads(out)
    assert data == {"task_id": orch.run_id, "status": "success", "summary_file": str(Path(tmp_path/"summary.md"))}
