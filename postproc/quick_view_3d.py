from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_first_frame_lammpstrj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal reader for LAMMPS dump 'custom' with columns:
      id type x y z radius
    Returns (pos[N,3], types[N]).
    """
    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
        if not line.startswith("ITEM: TIMESTEP"):
            raise ValueError("Not a LAMMPS dump file (missing ITEM: TIMESTEP).")
        _timestep = int(f.readline().strip())

        if not f.readline().startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError("Not a LAMMPS dump file (missing ITEM: NUMBER OF ATOMS).")
        n = int(f.readline().strip())

        box_hdr = f.readline()
        if not box_hdr.startswith("ITEM: BOX BOUNDS"):
            raise ValueError("Not a LAMMPS dump file (missing ITEM: BOX BOUNDS).")
        # skip 3 lines of bounds
        _ = f.readline()
        _ = f.readline()
        _ = f.readline()

        atoms_hdr = f.readline().strip()
        if not atoms_hdr.startswith("ITEM: ATOMS"):
            raise ValueError("Not a LAMMPS dump file (missing ITEM: ATOMS).")

        cols = atoms_hdr.split()[2:]
        needed = ["type", "x", "y", "z"]
        for k in needed:
            if k not in cols:
                raise ValueError(f"Dump is missing required column '{k}'. Found: {cols}")

        idx_type = cols.index("type")
        idx_x = cols.index("x")
        idx_y = cols.index("y")
        idx_z = cols.index("z")

        pos = np.zeros((n, 3), dtype=float)
        types = np.zeros((n,), dtype=int)

        for i in range(n):
            parts = f.readline().split()
            types[i] = int(parts[idx_type])
            pos[i, 0] = float(parts[idx_x])
            pos[i, 1] = float(parts[idx_y])
            pos[i, 2] = float(parts[idx_z])

    return pos, types


def main():
    # Expect to be run from project root:
    #   python -m postproc.quick_view_3d
    dump_path = Path(__file__).resolve().parents[1] / "simulations" / "case_example" / "dump.calendering.lammpstrj"
    if not dump_path.exists():
        print("Dump file not found:", dump_path)
        print("Run: python run_calendering.py  (then try again)")
        return

    pos, types = read_first_frame_lammpstrj(dump_path)
    am = types == 1
    cbd = types == 2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pos[am, 0], pos[am, 1], pos[am, 2], s=6, alpha=0.8, label="AM")
    ax.scatter(pos[cbd, 0], pos[cbd, 1], pos[cbd, 2], s=2, alpha=0.5, label="CBD")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_zlabel("z (um)")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

