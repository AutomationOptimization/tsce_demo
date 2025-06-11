import json
from pathlib import Path
import pytest
import tsce_agent_demo.utils.result_aggregator as agg


def test_summary_contains_artifacts(tmp_path):
    art = tmp_path / agg.ART_DIR
    art.mkdir()
    (art / "file1.meta.json").write_text(json.dumps({}))
    (art / "file2.meta.json").write_text(json.dumps({}))
    (art / "file3.meta.json").write_text(json.dumps({}))

    biblio = "Paper A\nPaper B\nPaper C"
    summary = agg.create_summary("Q", tmp_path, bibliography=biblio)
    text = summary.read_text()
    assert "file1.meta.json" in text
    assert text.count("[") == 3


def test_empty_artifacts_raise(tmp_path):
    art = tmp_path / agg.ART_DIR
    art.mkdir()
    (art / "empty.json").write_text("")
    with pytest.raises(FileNotFoundError):
        agg.create_summary("Q", tmp_path, bibliography="")

