from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.orchestrator import Orchestrator


def test_sanitize_script_uses_raw_prefix():
    orch = object.__new__(Orchestrator)
    bad_script = 'print("hi)'
    sanitized = orch._sanitize_script(bad_script)
    assert sanitized.startswith('r"""')
    compile(sanitized, '<string>', 'exec')
