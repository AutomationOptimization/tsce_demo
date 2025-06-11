from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import agents.hypothesis as hyp

class FakeResearcher:
    def __init__(self):
        self.created = False
        self.written = False
    def create_file(self, path: str, content: str = ""):
        self.created = True
        Path(path).write_text(content)
        return "ok"
    def write_file(self, path: str, content: str):
        self.written = True
        Path(path).write_text(content)
        return "ok"


def test_record_agreed_hypothesis_creates_file(tmp_path):
    p = tmp_path / "dir" / "lead.txt"
    r = FakeResearcher()
    token = hyp.record_agreed_hypothesis("A", "a", path=str(p), researcher=r)
    assert token == hyp.TERMINATE_TOKEN
    assert p.exists()
    assert r.created
    assert not r.written


def test_record_agreed_hypothesis_overwrites(tmp_path):
    p = tmp_path / "lead.txt"
    p.write_text("old")
    r = FakeResearcher()
    token = hyp.record_agreed_hypothesis("B", "B", path=str(p), researcher=r)
    assert token == hyp.TERMINATE_TOKEN
    assert p.read_text() == "B"
    assert r.written
