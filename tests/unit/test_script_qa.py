import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import agents.script_qa as script_qa_mod


def test_run_tests_requires_golden_thread(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")
    success, output = script_qa_mod.run_tests(script)
    assert not success
    assert "GOLDEN_THREAD" in output


def test_run_tests_executes_when_marker_present(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("# GOLDEN_THREAD:1\nprint('hi')")
    success, output = script_qa_mod.run_tests(script)
    assert success
    assert "hi" in output
