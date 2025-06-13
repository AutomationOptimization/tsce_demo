from __future__ import annotations
import subprocess
from pathlib import Path
import sys
import time
import shutil
import os

from .base_agent import BaseAgent, compose_sections


def run_simulation(path: str) -> str:
    """Execute a Python file and store the result in ``results/``.

    Parameters
    ----------
    path: str
        Path to the Python file to execute.

    Returns
    -------
    str
        Path to the log file containing stdout and stderr.
    """
    script = Path(path)
    if not script.is_file():
        raise FileNotFoundError(f"No such file: {script}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = out_dir / f"{script.stem}_{timestamp}.log"

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- stderr ---\n")
            f.write(proc.stderr)
        f.write(f"\n--- return code: {proc.returncode} ---\n")

    return str(log_file)


class Simulator(BaseAgent):
    """Execute Python scripts and record the output."""

    def __init__(self, *, log_dir: str | None = None, output_dir: str = "results") -> None:
        super().__init__(name="Simulator", log_dir=log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def send_message(self, message: str) -> str:  # pragma: no cover
        output = self.act(message)
        return compose_sections("", "", output)

    # ------------------------------------------------------------------
    def act(self, job: str | dict) -> str | dict:
        """Execute a simulation ``job`` and return the result.

        Parameters
        ----------
        job : str | dict
            Either a path to a Python script or a job dictionary with a
            ``"type"`` key describing the action to perform.
        """

        if isinstance(job, dict):
            job_type = job.get("type")
            if job_type == "script":
                return self.run_simulation(job["path"])
            elif job_type == "molecule_eval":
                from rdkit.Chem import Descriptors, MolFromSmiles

                smi = job["smiles"]
                mol = MolFromSmiles(smi)
                return {
                    "MW": Descriptors.MolWt(mol),
                    "logP": Descriptors.MolLogP(mol),
                    "TPSA": Descriptors.TPSA(mol),
                }
            else:
                raise ValueError(f"Unknown job type: {job_type}")

        # backward compatibility with ``path`` strings
        return self.run_simulation(job)

    # ------------------------------------------------------------------
    def run_simulation(self, path: str) -> str:
        """Run ``path`` and store the log in ``self.output_dir``.

        The log file path is appended to ``self.history`` and returned.
        """
        original_log = run_simulation(path)
        dest = self.output_dir / Path(original_log).name
        try:
            shutil.move(original_log, dest)
        except Exception:
            shutil.copy(original_log, dest)
            try:
                Path(original_log).unlink()
            except FileNotFoundError:  # pragma: no cover - cleanup
                pass
        log_path = str(dest)
        self.history.append(log_path)

        try:
            from tools.scoring import score_batch
            receptor = os.getenv("RECEPTOR_PDBQT", "")
            if receptor and Path(receptor).exists():
                for smi_file in self.output_dir.glob("*.smi"):
                    smiles = [l.strip() for l in smi_file.read_text().splitlines() if l.strip()]
                    if smiles:
                        df = score_batch(smiles, receptor)
                        df.to_csv(smi_file.with_name("scored.csv"), index=False)
        except Exception:
            pass
        return log_path


__all__ = ["Simulator", "run_simulation"]
