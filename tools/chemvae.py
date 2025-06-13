from __future__ import annotations
import random
from pathlib import Path
from typing import List, Optional

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
from selfies import encoder as sf_encode, decoder as sf_decode
from rdkit import Chem


class _ToyVAE(nn.Module if nn else object):
    def __init__(self, vocab_size: int, max_len: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, latent_dim)
        self.fc_mu = nn.Linear(max_len * latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(max_len * latent_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, max_len * vocab_size)

    def forward(self, x):
        batch = x.size(0)
        h = self.embed(x).view(batch, -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = mu + (torch.randn_like(mu) if torch else 0) * (torch.exp(0.5 * logvar) if torch else 1)
        logits = self.decoder(z).view(batch, self.max_len, -1)
        return logits, mu, logvar

    def decode(self, z):
        logits = self.decoder(z).view(-1, self.max_len, self.embed.num_embeddings)
        return logits


def _safer_smiles(smiles: List[str]) -> List[str]:
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            out.append(Chem.MolToSmiles(mol))
    return out


class ChemVAE:
    """Tiny SELFIES-based VAE wrapper."""

    def __init__(self, data_path: str = "data/chembl_smiles.smi") -> None:
        path = Path(data_path)
        if not path.exists():
            from scripts.fetch_chembl import fetch_chembl_smiles

            smiles = fetch_chembl_smiles(100)
            path.parent.mkdir(exist_ok=True)
            path.write_text("\n".join(smiles))
        else:
            smiles = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        self.smiles = _safer_smiles(smiles)
        self.selfies = [sf_encode(s) for s in self.smiles]
        tokens = set(t for sf in self.selfies for t in sf.split())
        self.vocab = sorted(tokens)
        self.tok_to_idx = {t: i + 1 for i, t in enumerate(self.vocab)}
        self.idx_to_tok = {i: t for t, i in self.tok_to_idx.items()}
        self.max_len = max(len(sf.split()) for sf in self.selfies)
        self.model = _ToyVAE(len(self.vocab) + 1, self.max_len) if nn else None
        if self.model:
            self._train(1)

    def _encode(self, sf: str) -> List[int]:
        idxs = [self.tok_to_idx[t] for t in sf.split()]
        pad = [0] * (self.max_len - len(idxs))
        return idxs + pad

    def _decode(self, idxs: List[int]) -> str:
        toks = [self.idx_to_tok.get(i, "") for i in idxs if i]
        sf = " ".join(toks)
        return sf_decode(sf)

    def _train(self, epochs: int = 1) -> None:
        if not torch or not self.model:
            return
        data = torch.tensor([self._encode(sf) for sf in self.selfies], dtype=torch.long)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        for _ in range(epochs):
            opt.zero_grad()
            logits, mu, logvar = self.model(data)
            loss_recon = nn.functional.cross_entropy(logits.transpose(1, 2), data)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss_recon + kl
            loss.backward()
            opt.step()

    # ------------------------------------------------------------------
    def generate_smiles(self, n: int = 1, cond: Optional[dict] = None) -> List[str]:
        if not self.model:
            results = []
            for _ in range(n):
                base = random.choice(self.smiles)
                mutated = "F" + base
                if Chem.MolFromSmiles(mutated):
                    results.append(mutated)
                else:
                    results.append(base)
            return results
        self.model.eval()
        results = []
        for _ in range(n):
            z = torch.randn(1, self.model.latent_dim) if torch else 0
            logits = self.model.decode(z)[0]
            idxs = logits.argmax(dim=1).tolist()
            smi = self._decode(idxs)
            if Chem.MolFromSmiles(smi):
                results.append(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
            else:
                results.append(random.choice(self.smiles))
        return results


__all__ = ["ChemVAE"]
