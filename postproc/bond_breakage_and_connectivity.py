"""
Bond breakage + connectivity metrics (A+B).
==========================================

Implements two complementary analyses for bonded DEM calendering runs:

(A) Broken-bond event counting from LAMMPS `dump local` output produced by
    BPM bond styles using `store/local ... time id1 id2 x y z`.

(B) Bond-network connectivity metrics computed by starting from the initial
    bond list (from `data.bonded_relaxed` / `data.bonded_pre_damage`) and
    progressively removing bonds as they break.

This file is intentionally dependency-light (no networkx/pandas required).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class BrokenBondEvent:
    timestep: int
    id1: int
    id2: int
    x: float | None = None
    y: float | None = None
    z: float | None = None


def _safe_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_broken_bond_events(dump_local_path: Path) -> Iterator[BrokenBondEvent]:
    """
    Parse LAMMPS `dump local` file that looks like:

      ITEM: TIMESTEP
      1000
      ITEM: NUMBER OF ENTRIES
      1225
      ...
      ITEM: ENTRIES f_brkbond[1] f_brkbond[2] ...
      5 1052 1189 10.6075 2.07098 0.628876

    where columns are: time id1 id2 x y z (as configured in run_calendering.py).
    """
    if not dump_local_path.exists():
        return

    timestep: int | None = None
    in_entries = False

    with dump_local_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("ITEM:"):
                in_entries = line.startswith("ITEM: ENTRIES")
                continue

            if not in_entries:
                # TIMESTEP value appears right after ITEM: TIMESTEP
                if timestep is None:
                    # not yet set, keep scanning
                    pass
                continue

            # Inside entries section: one broken bond per line
            parts = line.split()
            if len(parts) < 3:
                continue

            t_local = _safe_int(parts[0])
            id1 = _safe_int(parts[1])
            id2 = _safe_int(parts[2])
            if t_local is None or id1 is None or id2 is None:
                continue

            x = _safe_float(parts[3]) if len(parts) >= 4 else None
            y = _safe_float(parts[4]) if len(parts) >= 5 else None
            z = _safe_float(parts[5]) if len(parts) >= 6 else None

            # The "time" recorded by BPM is the timestep bond broke at, which is
            # what we want to group by. We store it as `timestep`.
            yield BrokenBondEvent(timestep=t_local, id1=id1, id2=id2, x=x, y=y, z=z)


def read_bonds_from_lammps_data(data_path: Path) -> set[tuple[int, int]]:
    """
    Read the Bonds section from a LAMMPS data file and return undirected edges (min,max).
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Missing bonded data file: {data_path}")

    in_bonds = False
    edges: set[tuple[int, int]] = set()
    with data_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            if not in_bonds:
                if line.lower() == "bonds":
                    in_bonds = True
                continue

            # Stop at next section header (e.g., Angles) if present
            if line[0].isalpha():
                break

            parts = line.split()
            if len(parts) < 4:
                continue
            # bond_id bond_type atom_i atom_j
            ai = _safe_int(parts[2])
            aj = _safe_int(parts[3])
            if ai is None or aj is None or ai == aj:
                continue
            edges.add((min(ai, aj), max(ai, aj)))

    return edges


def _build_adjacency(edges: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _largest_component_size(nodes: Iterable[int], adj: dict[int, set[int]]) -> int:
    seen: set[int] = set()
    best = 0
    for start in nodes:
        if start in seen:
            continue
        q: deque[int] = deque([start])
        seen.add(start)
        size = 0
        while q:
            u = q.popleft()
            size += 1
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        best = max(best, size)
    return best


def compute_bond_network_timeseries(
    initial_bonds: set[tuple[int, int]],
    broken_events: list[BrokenBondEvent],
    *,
    timesteps: list[int] | None = None,
) -> dict:
    """
    Compute connectivity metrics over time by removing broken bonds.
    """
    # Determine evaluation timesteps
    if timesteps is None:
        unique_ts = sorted({evt.timestep for evt in broken_events})
        timesteps = unique_ts

    # Group broken edges by timestep
    breaks_by_ts: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for evt in broken_events:
        edge = (min(evt.id1, evt.id2), max(evt.id1, evt.id2))
        breaks_by_ts[evt.timestep].append(edge)

    edges = set(initial_bonds)
    # Nodes are those that appear in the initial network
    nodes = sorted({n for e in edges for n in e})

    results = {
        "timesteps": [],
        "active_bonds": [],
        "broken_bonds_cumulative": [],
        "largest_component_size": [],
        "largest_component_fraction": [],
    }

    broken_cum = 0
    for ts in sorted(timesteps):
        for edge in breaks_by_ts.get(ts, []):
            if edge in edges:
                edges.remove(edge)
        broken_cum += len(breaks_by_ts.get(ts, []))

        adj = _build_adjacency(edges)
        lcc = _largest_component_size(nodes, adj) if nodes else 0
        results["timesteps"].append(ts)
        results["active_bonds"].append(len(edges))
        results["broken_bonds_cumulative"].append(broken_cum)
        results["largest_component_size"].append(lcc)
        results["largest_component_fraction"].append(
            (lcc / len(nodes)) if nodes else 0.0
        )

    return results


def analyze_case(case_dir: Path) -> dict:
    """
    Analyze one simulation case folder.
    """
    # Prefer the bonded restart that is closest to the damage stage.
    candidate_data = [
        case_dir / "data.bonded_pre_damage",
        case_dir / "data.bonded_relaxed",
        case_dir / "data.bonded_initial",
    ]
    data_path = next((p for p in candidate_data if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            f"No bonded data file found in {case_dir}. Expected one of: "
            + ", ".join(p.name for p in candidate_data)
        )

    broken_dump = case_dir / "dump.broken_bonds.local"
    broken_events = list(iter_broken_bond_events(broken_dump))
    initial_bonds = read_bonds_from_lammps_data(data_path)

    timeseries = compute_bond_network_timeseries(initial_bonds, broken_events)
    summary = {
        "bond_data_file": data_path.name,
        "initial_bonds": len(initial_bonds),
        "broken_events": len(broken_events),
        "final_active_bonds_reconstructed": timeseries["active_bonds"][-1]
        if timeseries["active_bonds"]
        else len(initial_bonds),
        "final_largest_component_fraction": timeseries["largest_component_fraction"][-1]
        if timeseries["largest_component_fraction"]
        else 1.0,
    }

    return {"summary": summary, "timeseries": timeseries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--out", default="bond_breakage_connectivity.json")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    payload = analyze_case(case_dir)
    out_path = case_dir / args.out
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

