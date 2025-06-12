import pandas as pd
from pathlib import Path
import tools


def test_memory_search_top_result(tmp_path, monkeypatch):
    df = pd.DataFrame({"embedding": [[1.0, 0.0], [0.0, 1.0]], "text": ["a", "b"]})
    path = tmp_path / "mem.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setenv("MEMORY_PATH", str(path))
    monkeypatch.setattr(tools, "embed_text", lambda q: [1.0, 0.0])
    res = tools.memory_search("q", k=1)
    assert res == ["a"]


def test_memory_search_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "none.parquet"))
    res = tools.memory_search("q")
    assert res == []
