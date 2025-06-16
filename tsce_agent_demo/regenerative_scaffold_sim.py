#!/usr/bin/env python
"""
regenerative_scaffold_sim_v2.py
──────────────────────────────────────────────────────────────────────────────
A *sandbox* for jellyfish-inspired liver-scaffold regeneration hypotheses.

New vs. v1
──────────
✓ Burst-mode “cancer-like” proliferation (configurable window & rate)  
✓ Nutrient / hypoxia cap via local-density threshold  
✓ Localised ‘scar zone’ with higher proliferation inside the zone  
✓ ECM-remodelling: cells can digest neighbouring solid voxels (MMP-like)  
✓ Quick porosity sweep option (loop over values)  
✓ More flexible protocol-selection heuristic & extra plots
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Iterable

# ─────────────────────────────────────────────────────────────────────────────
# 0 ▸ PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScaffoldParams:
    size: tuple[int, int, int] = (50, 50, 10)
    porosity: float = 0.85
    stiffness_kpa: float = 2.0

@dataclass
class CellParams:
    # Baseline behaviour
    proliferation_rate: float = 0.01
    death_rate: float = 0.0001
    migration_prob: float = 0.2
    # Burst-mode “cancer-like” phase
    burst_until: int = 30          # timesteps with elevated rate
    burst_rate: float = 0.05       # supersedes baseline during burst

@dataclass
class NutrientParams:
    # Simple crowding / hypoxia rule
    local_density_cap: int = 6     # cells in 3×3×3 cube before starvation
    starvation_death_boost: float = 0.002  # extra death prob if starving

@dataclass
class ZoneParams:
    # Scar zone where we “switch on” a reprogramming-like boost
    center: tuple[int, int, int] = (25, 25, 5)
    radius: int = 12
    zone_proliferation_multiplier: float = 3.0

@dataclass
class ECMParams:
    # Cells digest scaffold walls (MMP-9 etc.)
    remodel_prob: float = 0.02     # per neighbouring solid voxel per step

@dataclass
class SimParams:
    steps: int = 200
    random_seed: int | None = 42

@dataclass
class SimulationResults:
    cells_over_time: list[int] = field(default_factory=list)
    fibrotic_index: list[float] = field(default_factory=list)

# ─────────────────────────────────────────────────────────────────────────────
# 1 ▸ UTILS
# ─────────────────────────────────────────────────────────────────────────────

def design_scaffold(p: ScaffoldParams, *, rng=np.random) -> np.ndarray:
    """Boolean 3-D array: True = pore voxel."""
    return rng.random(p.size) < p.porosity

def make_scar_mask(shape: tuple[int, int, int], zone: ZoneParams) -> np.ndarray:
    """Boolean mask: True inside scar zone radius."""
    grid = np.indices(shape).transpose(1, 2, 3, 0)
    dist = np.linalg.norm(grid - np.array(zone.center), axis=-1)
    return dist <= zone.radius

NEIGHBOURS = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

# ─────────────────────────────────────────────────────────────────────────────
# 2 ▸ SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate(
    scaffold: np.ndarray,
    c: CellParams,
    n: NutrientParams,
    z: ZoneParams,
    ecm: ECMParams,
    sim: SimParams,
) -> SimulationResults:
    rng = np.random.default_rng(sim.random_seed)
    shape = scaffold.shape
    cells = np.zeros(shape, dtype=bool)

    # ── Seed the very first hepatocyte ─────────────────────────
    # Guarantee it starts on pore space; if the centre voxel is solid
    # (which happens ~15 % of the time when porosity = 0.85) pick a
    # random pore instead.
    center = tuple(s // 2 for s in shape)
    if not scaffold[center]:
        pore_coords = list(zip(*np.nonzero(scaffold)))     # all pore voxels
        center = pore_coords[rng.integers(len(pore_coords))]
    cells[center] = True

    scar_mask = make_scar_mask(shape, z)
    results = SimulationResults()

    # Precompute local-density kernel offsets for 3×3×3 cube
    kernel = [(dx,dy,dz)
              for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
              if not (dx==dy==dz==0)]

    for t in range(sim.steps):
        prolif_rate_now = (
            c.burst_rate if t < c.burst_until else c.proliferation_rate
        )
        # iterate over *copy* coords
        for idx in list(zip(*np.nonzero(cells))):
            # starve rule
            neighbourhood = [
                tuple(np.array(idx)+off) for off in kernel
                if all(0<=idx[d]+off[d]<shape[d] for d in range(3))
            ]
            local_density = sum(cells[nb] for nb in neighbourhood)
            starving = local_density > n.local_density_cap

            # choose rates this cell experiences
            death_prob = c.death_rate + (n.starvation_death_boost if starving else 0)

            # proliferation (boost if in scar zone)
            eff_prolif = prolif_rate_now * (
                z.zone_proliferation_multiplier if scar_mask[idx] else 1.0
            )
            if rng.random() < eff_prolif:
                rng.shuffle(NEIGHBOURS)
                for off in NEIGHBOURS:
                    npos = tuple(np.array(idx)+off)
                    if all(0<=npos[d]<shape[d] for d in range(3)) and scaffold[npos] and not cells[npos]:
                        cells[npos] = True
                        break

            # migration
            if rng.random() < c.migration_prob:
                rng.shuffle(NEIGHBOURS)
                for off in NEIGHBOURS:
                    npos = tuple(np.array(idx)+off)
                    if all(0<=npos[d]<shape[d] for d in range(3)) and scaffold[npos] and not cells[npos]:
                        cells[npos] = True
                        cells[idx] = False
                        break

            # death
            if rng.random() < death_prob:
                cells[idx] = False

            # ECM remodelling: convert solid neighbour → pore
            for off in NEIGHBOURS:
                npos = tuple(np.array(idx)+off)
                if all(0<=npos[d]<shape[d] for d in range(3)) and not scaffold[npos]:
                    if rng.random() < ecm.remodel_prob:
                        scaffold[npos] = True  # digest wall

        # metrics
        results.cells_over_time.append(int(cells.sum()))
        results.fibrotic_index.append(1.0 - cells.sum()/scaffold.sum())

    return results

# ─────────────────────────────────────────────────────────────────────────────
# 3 ▸ PROTOCOL SUGGESTION (TOY HEURISTIC)
# ─────────────────────────────────────────────────────────────────────────────

def propose_protocol(res: SimulationResults) -> dict[str,str]:
    early_fib = min(res.fibrotic_index[:20])
    final_fib = res.fibrotic_index[-1]
    growth_peak = max(res.cells_over_time)
    if final_fib < 0.1 and growth_peak > 0.8 * max(res.cells_over_time):
        return {
            "strategy": "partial_reprogramming",
            "factors": "Oct4, Sox2, Klf4, Myc",
            "delivery": "mRNA pulses (48 h × 4)",
        }
    if final_fib < 0.3:
        return {
            "strategy": "targeted ECM + partial repro",
            "factors": "MMP-9, VEGF + OSKM",
            "delivery": "AAV + peptide hydrogel",
        }
    return {
        "strategy": "full reprogramming & scaffold redesign",
        "factors": "OSKM, Lin28, Nanog",
        "delivery": "non-integrating lentivirus",
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4 ▸ EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_once(porosity: float):
    s_params = ScaffoldParams(porosity=porosity)
    c_params = CellParams()
    n_params = NutrientParams()
    z_params = ZoneParams()
    ecm_params = ECMParams()
    sim_params = SimParams()

    scaffold = design_scaffold(s_params)
    res = simulate(scaffold, c_params, n_params, z_params, ecm_params, sim_params)
    proto = propose_protocol(res)
    return res, proto

def porosity_sweep(values: Iterable[float]):
    fig, ax = plt.subplots()
    for p in values:
        res, _ = run_once(p)
        ax.plot(res.cells_over_time, label=f"porosity={p}")
    ax.set_xlabel("timestep")
    ax.set_ylabel("total cells")
    ax.set_title("Porosity sweep")
    ax.legend()
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 5 ▸ MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Single run with default porosity
    res, proto = run_once(0.85)
    print("\nSuggested protocol:\n", proto)

    plt.figure(figsize=(6,4))
    plt.plot(res.cells_over_time, label="cells")
    plt.plot(res.fibrotic_index, label="fibrotic index")
    plt.xlabel("timestep")
    plt.legend(); plt.tight_layout(); plt.show()

    # Optional: uncomment to perform a quick porosity sweep
    # porosity_sweep([0.6, 0.7, 0.8, 0.9])
