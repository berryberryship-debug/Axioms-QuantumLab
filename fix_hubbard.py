import itertools
import os
import time
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Config:
    lx: int = 2
    ly: int = 2
    lz: int = 2
    n_up: int = 4
    n_down: int = 4
    u: float = 4.0

def gen_masks(sites, particles):
    masks = []
    for combination in itertools.combinations(range(sites), particles):
        mask = 0
        for site in combination:
            mask |= 1 << site
        masks.append(mask)
    return np.asarray(masks, dtype=np.uint64)

def popcount_array(values):
    values = np.asarray(values, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    return np.fromiter((int(v).bit_count() for v in values.reshape(-1)), dtype=np.uint8, count=values.size).reshape(values.shape)

def interaction_diagonal_blocked(up_masks, down_masks, u, block_size):
    dim_up = up_masks.size
    dim_down = down_masks.size
    diagonal = np.empty(dim_up * dim_down, dtype=np.float64)
    for start in range(0, dim_up, block_size):
        stop = min(start + block_size, dim_up)
        overlaps = up_masks[start:stop, None] & down_masks[None, :]
        counts = popcount_array(overlaps)
        diagonal[start * dim_down : stop * dim_down] = (float(u) * counts.astype(np.float64, copy=False)).reshape(-1)
    return diagonal

if __name__ == "__main__":
    config = Config()
    sites = config.lx * config.ly * config.lz
    up_masks = gen_masks(sites, config.n_up)
    down_masks = gen_masks(sites, config.n_down)
    print("=== Configuration Hubbard 3D ===")
    diagonal = interaction_diagonal_blocked(up_masks, down_masks, config.u, 16)
    print("Norme L2 de la diagonale :", np.linalg.norm(diagonal))
    print("Exécution OK")
