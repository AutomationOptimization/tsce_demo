import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open
import requests
import subprocess
import os
import pytest
import tools

sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools import (
    GoogleSearch,
    WebScrape,
    CreateFileTool,
    ReadFileTool,
    EditFileTool,
    DeleteFileTool,
    RunScriptTool,
    use_tool,
)


def test_google_search_success(monkeypatch):
    html = "<h3>One</h3><h3>Two</h3><h3>Three</h3>"

    class FakeResp:
        text = html

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())
    res = GoogleSearch()("query", num_results=2)
    assert res == ["One", "Two"]


def test_google_search_error(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("bad")

    monkeypatch.setattr(requests, "get", boom)
    res = GoogleSearch()("query")
    assert res == ["Search error: bad"]


def test_web_scrape_success(monkeypatch):
    html = "<html><body><h1>Title</h1><script>x</script><p>hi</p></body></html>"

    class FakeResp:
        text = html

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())
    res = WebScrape()("http://x")
    assert res == "Title hi"


def test_web_scrape_error(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")))
    res = WebScrape()("http://x")
    assert res == "Scrape error: fail"


def test_create_file_success(tmp_path):
    path = tmp_path / "f.txt"
    tool = CreateFileTool()
    msg = tool(str(path), "data")
    assert msg == f"Created {path}"
    assert path.read_text() == "data"


def test_create_file_exists(monkeypatch):
    def raise_exists(*a, **kw):
        raise FileExistsError

    monkeypatch.setattr("builtins.open", raise_exists)
    tool = CreateFileTool()
    msg = tool("x.txt")
    assert msg == "Error: x.txt already exists"


def test_read_file_success(monkeypatch):
    m = mock_open(read_data="hello")
    monkeypatch.setattr("builtins.open", m)
    tool = ReadFileTool()
    res = tool("a.txt")
    m.assert_called_once_with("a.txt", "r", encoding="utf-8")
    assert res == "hello"


def test_read_file_missing(monkeypatch):
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError))
    tool = ReadFileTool()
    res = tool("a.txt")
    assert res == "Error: a.txt not found"


def test_edit_file_success(monkeypatch):
    m = mock_open()
    monkeypatch.setattr("builtins.open", m)
    tool = EditFileTool()
    msg = tool("b.txt", "new")
    handle = m()
    handle.seek.assert_called_once_with(0)
    handle.write.assert_called_once_with("new")
    handle.truncate.assert_called_once()
    assert msg == "Updated b.txt"


def test_edit_file_missing(monkeypatch):
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError))
    tool = EditFileTool()
    msg = tool("b.txt", "new")
    assert msg == "Error: b.txt not found"


def test_delete_file_success(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "remove", lambda p: calls.append(p))
    tool = DeleteFileTool()
    msg = tool("c.txt")
    assert calls == ["c.txt"]
    assert msg == "Deleted c.txt"


def test_delete_file_missing(monkeypatch):
    monkeypatch.setattr(os, "remove", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError))
    tool = DeleteFileTool()
    msg = tool("c.txt")
    assert msg == "Error: c.txt not found"


def test_run_script_success(monkeypatch):
    def fake_run(cmd, capture_output, text, check, timeout):
        assert cmd == ["python", "script.py"]
        return SimpleNamespace(stdout="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)
    tool = RunScriptTool()
    msg = tool("script.py")
    assert msg == "outerr"


def test_run_script_missing(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError))
    tool = RunScriptTool()
    msg = tool("script.py")
    assert msg == "Error: script.py not found"


def test_use_tool_dispatch(monkeypatch):
    class Dummy:
        def __call__(self, **kw):
            return "ok"

    monkeypatch.setitem(tools.__dict__["TOOL_CLASSES"], "dummy", Dummy)
    assert use_tool("dummy") == "ok"


def test_use_tool_unknown():
    with pytest.raises(ValueError):
        use_tool("nope")
