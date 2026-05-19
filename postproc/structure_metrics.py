from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_positions_types_from_data(data_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load particle positions, radii, and types from a LAMMPS data file (atom_style sphere).

    Expected per-atom line format in the Atoms section:
      id type diameter density x y z
    """
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    rs: List[float] = []
    types: List[int] = []

    atoms_section = False
    with data_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip().lower().startswith("atoms # sphere"):
                atoms_section = True
                continue
            if atoms_section:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 7:
                    break
                types.append(int(parts[1]))
                diameter = float(parts[2])
                x = float(parts[4])
                y = float(parts[5])
                z = float(parts[6])
                xs.append(x)
                ys.append(y)
                zs.append(z)
                rs.append(0.5 * diameter)

    pos = np.column_stack([xs, ys, zs])
    radii = np.array(rs)
    types_arr = np.array(types, dtype=int)
    return pos, radii, types_arr


def radial_distribution(
    pos: np.ndarray,
    types: np.ndarray,
    type_i: int,
    type_j: int,
    r_max: float,
    n_bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a simple radial distribution function g(r) between particles
    of type_i (centers) and type_j (neighbors) without periodic boundaries.
    """
    centers = pos[types == type_i]
    neighbors = pos[types == type_j]
    if centers.size == 0 or neighbors.size == 0:
        r = np.linspace(0.0, r_max, n_bins)
        return r, np.zeros_like(r)

    dr = r_max / n_bins
    hist = np.zeros(n_bins, dtype=float)

    for c in centers:
        d = np.linalg.norm(neighbors - c, axis=1)
        d = d[(d > 0.0) & (d < r_max)]
        idx = (d / dr).astype(int)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        for k in idx:
            hist[k] += 1.0

    # Normalize to get something comparable across cases (not a strict g(r))
    hist /= len(centers)
    r_vals = (np.arange(n_bins) + 0.5) * dr
    return r_vals, hist


if __name__ == "__main__":
    data = Path(__file__).resolve().parents[1] / "simulations" / "case_example" / "data.electrode"
    if not data.exists():
        print("No example data file found:", data)
    else:
        pos, radii, types = load_positions_types_from_data(data)
        r, g_am_am = radial_distribution(pos, types, type_i=1, type_j=1, r_max=10.0)
        print("Computed g(r) for AM-AM with", len(r), "bins")

