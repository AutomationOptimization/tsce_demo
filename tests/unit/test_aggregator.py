import json
from freezegun import freeze_time
from tsce_demo.utils import result_aggregator as agg


def test_details_and_citations(tmp_path):
    art = tmp_path / agg.ART_DIR
    art.mkdir()
    for i in range(3):
        (art / f"file{i}.meta.json").write_text(json.dumps({}))
    biblio = "Ref1\nRef2\nRef3"
    with freeze_time("2024-01-01"):
        summary = agg.create_summary("Q", tmp_path, bibliography=biblio)
    text = summary.read_text()
    assert "<details>" in text
    assert all(f"[{i+1}]" in text for i in range(3))
