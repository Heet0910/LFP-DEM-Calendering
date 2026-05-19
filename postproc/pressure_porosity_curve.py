from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from input.geometry_materials_config import DomainConfig

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


@dataclass(frozen=True)
class FrameMetrics:
    step: int
    bed_height: float
    porosity_proxy: float


def sphere_volume(r: float) -> float:
    return (4.0 / 3.0) * np.pi * r**3


def iter_lammpstrj_frames(path: Path):
    """
    Stream parser for LAMMPS dump custom with columns:
      id type x y z radius
    Yields: (step:int, x:np.ndarray, y:np.ndarray, z:np.ndarray, r:np.ndarray)
    """
    with path.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                raise ValueError("Unexpected dump format: missing ITEM: TIMESTEP")
            step = int(f.readline().strip())

            if not f.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Unexpected dump format: missing ITEM: NUMBER OF ATOMS")
            n = int(f.readline().strip())

            box = f.readline().strip()
            if not box.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Unexpected dump format: missing ITEM: BOX BOUNDS")
            # skip bounds
            _ = f.readline()
            _ = f.readline()
            _ = f.readline()

            atoms_hdr = f.readline().strip()
            if not atoms_hdr.startswith("ITEM: ATOMS"):
                raise ValueError("Unexpected dump format: missing ITEM: ATOMS")

            cols = atoms_hdr.split()[2:]
            required = ["x", "y", "z", "radius"]
            for k in required:
                if k not in cols:
                    raise ValueError(f"Dump missing required column '{k}'. Found: {cols}")

            ix = cols.index("x")
            iy = cols.index("y")
            iz = cols.index("z")
            ir = cols.index("radius")

            x = np.empty(n, dtype=float)
            y = np.empty(n, dtype=float)
            z = np.empty(n, dtype=float)
            r = np.empty(n, dtype=float)

            for i in range(n):
                parts = f.readline().split()
                x[i] = float(parts[ix])
                y[i] = float(parts[iy])
                z[i] = float(parts[iz])
                r[i] = float(parts[ir])

            yield step, x, y, z, r


def load_pressure_log(path: Path) -> dict[int, float]:
    """
    pressure_log.txt columns:
      step disp_top top_force pressure
    Returns {step: P}
    """
    if not path.exists():
        return {}
    raw = np.loadtxt(path, skiprows=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 4:
        raise ValueError(f"pressure_log has unexpected columns: {raw.shape[1]}")
    steps = raw[:, 0].astype(int)
    P = raw[:, 3].astype(float)
    return dict(zip(steps, P, strict=False))


def compute_metrics(dump_path: Path) -> tuple[list[FrameMetrics], float]:
    """
    Porosity proxy definition:
      epsilon(t) = 1 - V_solid / (Area * bed_height(t))
    where bed_height(t) is estimated from particle extents:
      z_max = max(z + r), z_min = min(z - r)
      bed_height = max(z_max - z_min, small)
    """
    area = DomainConfig.lx * DomainConfig.ly

    metrics: list[FrameMetrics] = []
    Vsolid: float | None = None

    for step, _x, _y, z, r in iter_lammpstrj_frames(dump_path):
        if Vsolid is None:
            Vsolid = float(np.sum(sphere_volume(r)))
        zmax = float(np.max(z + r))
        zmin = float(np.min(z - r))
        bed_h = max(zmax - zmin, 1e-9)
        eps = 1.0 - (Vsolid / (area * bed_h))
        metrics.append(FrameMetrics(step=step, bed_height=bed_h, porosity_proxy=eps))

    if Vsolid is None:
        raise ValueError("No frames found in dump file.")

    return metrics, Vsolid


def main():
    root = Path(__file__).resolve().parents[1] / "simulations" / "case_example"
    dump_path = root / "dump.calendering.lammpstrj"
    pressure_path = root / "pressure_log.txt"

    if not dump_path.exists():
        print("Missing dump file:", dump_path)
        print("Run: python run_calendering.py")
        return
    if plt is None:
        print("matplotlib is not installed, so plots cannot be displayed.")
        return

    P_by_step = load_pressure_log(pressure_path)
    metrics, _Vsolid = compute_metrics(dump_path)

    steps = np.array([m.step for m in metrics], dtype=int)
    por = np.array([m.porosity_proxy for m in metrics], dtype=float)
    bed_h = np.array([m.bed_height for m in metrics], dtype=float)
    P = np.array([P_by_step.get(int(s), np.nan) for s in steps], dtype=float)

    # Plots
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ax[0].plot(steps, bed_h, lw=1.5)
    ax[0].set_ylabel("Bed height (um)")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(steps, por, lw=1.5)
    ax[1].set_ylabel("Porosity proxy (-)")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(steps, P, lw=1.5)
    ax[2].set_ylabel("Pressure (simulation units)")
    ax[2].set_xlabel("Timestep")
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Pressure vs porosity curve (drop NaNs)
    mask = np.isfinite(P)
    if np.any(mask):
        plt.figure(figsize=(6, 5))
        plt.plot(por[mask], P[mask], "o-", ms=3)
        plt.xlabel("Porosity proxy (-)")
        plt.ylabel("Pressure (simulation units)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No pressure values found (pressure_log.txt missing or empty).")


if __name__ == "__main__":
    main()

