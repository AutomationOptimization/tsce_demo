import types
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import agents.orchestrator as orchestrator_mod
import agents.researcher as researcher_mod
import tsce_agent_demo.tsce_chat as tsce_chat_mod
import agents.base_agent as base_agent_mod


class DummyChat:
    def __init__(self):
        self.calls = []

    def __call__(self, messages):
        if isinstance(messages, list):
            content = messages[-1]["content"]
        else:
            content = messages
        self.calls.append(content)
        if "plan" in content:
            return types.SimpleNamespace(content="1. scrape http://example.com\n2. run tool.py")
        return types.SimpleNamespace(content=content)


class DummyResearcher:
    def __init__(self, *args, **kwargs):
        self.history = []

    def search(self, query):
        return "search:" + query

    def send_message(self, message):
        # echo back the scientist's hypothesis so the stage advances
        return message

    def write_file(self, path, content):
        Path(path).write_text(content)
        return "ok"

    def create_file(self, path, content=""):
        Path(path).write_text(content)
        return "ok"

    def read_file(self, path):
        return Path(path).read_text() if Path(path).exists() else ""

    def scrape(self, url):
        return "scraped:" + url

    def run_script(self, path):
        return "ran:" + path


class LoggingResearcher(DummyResearcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def scrape(self, url):
        self.calls.append(("scrape", url))
        return super().scrape(url)

    def run_script(self, path):
        self.calls.append(("run", path))
        return super().run_script(path)


class DummyResearcherList(DummyResearcher):
    def search(self, query):
        return ["result1", "result2"]


class DummyChatNoTools(DummyChat):
    pass


def test_scientist_instructs_researcher(tmp_path, monkeypatch):
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(researcher_mod, "TSCEChat", lambda model=None: DummyChat())
    monkeypatch.setattr(orchestrator_mod, "Researcher", LoggingResearcher)
    monkeypatch.setattr(researcher_mod, "Researcher", LoggingResearcher)

    orch = orchestrator_mod.Orchestrator(["goal", "terminate"], model="test", output_dir=str(tmp_path))
    orch.drop_stage("script")
    orch.drop_stage("qa")
    orch.drop_stage("simulate")
    orch.drop_stage("evaluate")
    orch.drop_stage("judge")

    orch.run()

    assert ("scrape", "http://example.com") in orch.researcher.calls
    assert ("run", "tool.py") in orch.researcher.calls

    run_path = Path(orch.output_dir)
    assert (run_path / "research.txt").read_text() == "search:goal"


def test_search_results_list_joined(tmp_path, monkeypatch):
    monkeypatch.setattr(tsce_chat_mod, "_make_client", lambda: ("dummy", object(), ""))
    monkeypatch.setattr(tsce_chat_mod, "TSCEChat", lambda model=None: DummyChatNoTools())
    monkeypatch.setattr(base_agent_mod, "TSCEChat", lambda model=None: DummyChatNoTools())
    monkeypatch.setattr(orchestrator_mod, "TSCEChat", lambda model=None: DummyChatNoTools())
    monkeypatch.setattr(researcher_mod, "TSCEChat", lambda model=None: DummyChatNoTools())
    monkeypatch.setattr(orchestrator_mod, "Researcher", DummyResearcherList)
    monkeypatch.setattr(researcher_mod, "Researcher", DummyResearcherList)

    orch = orchestrator_mod.Orchestrator(["goal", "terminate"], model="test", output_dir=str(tmp_path))
    orch.drop_stage("script")
    orch.drop_stage("qa")
    orch.drop_stage("simulate")
    orch.drop_stage("evaluate")
    orch.drop_stage("judge")

    orch.run()

    run_path = Path(orch.output_dir)
    lines = (run_path / "research.txt").read_text().splitlines()
    assert lines == ["result1", "result2"]

