from __future__ import annotations
import random
from typing import List
from pathlib import Path

from tools import ChemVAE, score_batch
from .base_agent import BaseAgent
from selfies import encoder, decoder
import pandas as pd


class Evolver(BaseAgent):
    """Simple genetic algorithm for molecule optimisation."""

    def __init__(self, receptor: str, *, log_dir: str | None = None) -> None:
        super().__init__("Evolver", log_dir=log_dir)
        self.receptor = receptor
        self.vae = ChemVAE()
        self.vocab = self.vae.vocab

    def _mutate(self, smi: str) -> str:
        sf = encoder(smi)
        tokens = sf.split()
        if not tokens:
            return smi
        idx = random.randrange(len(tokens))
        tokens[idx] = random.choice(self.vocab)
        return decoder(" ".join(tokens))

    def _pareto(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for i, row in df.iterrows():
            dominated = False
            for j, other in df.iterrows():
                if j == i:
                    continue
                cond = (
                    other["dg"] <= row["dg"]
                    and other["tox21"] <= row["tox21"]
                    and other["sa"] <= row["sa"]
                )
                better = (
                    other["dg"] < row["dg"]
                    or other["tox21"] < row["tox21"]
                    or other["sa"] < row["sa"]
                )
                if cond and better:
                    dominated = True
                    break
            if not dominated:
                keep.append(i)
        return df.loc[keep]

    def run(self, seeds: List[str], generations: int = 3) -> List[pd.Series]:
        pop = seeds
        scores = score_batch(pop, self.receptor)
        best_gen = [scores.sort_values("dg").iloc[0]]
        for _ in range(generations):
            mutants = [self._mutate(random.choice(pop)) for _ in pop]
            scores = score_batch(mutants, self.receptor)
            front = self._pareto(scores)
            pop = front["smiles"].tolist()
            best_gen.append(front.sort_values("dg").iloc[0])
        return best_gen


__all__ = ["Evolver"]
