"""
Connectivity Loss Damage Analysis
==================================

Post-processes LAMMPS dump files to compute particle damage metrics based on
loss of contacts during compression. This method is based on:

- Sangrós Giménez et al. (2020): "Mechanical, electrical, and ionic behavior
  of lithium‐ion battery electrodes via discrete element method simulations"
- Zhang et al. (2022): "Investigation on mechanical and microstructural evolution
  of lithium-ion battery electrode during the calendering process"

Damage Indicator:
  For each particle i at time t:
    Z_i(t) = number of contacts at time t
    Z_i(0) = number of contacts at initial state
    Damage_i(t) = max(0, 1 - Z_i(t) / Z_i(0))

  Global damage metric:
    D_global(t) = average(Damage_i(t)) across all particles

Usage:
    python connectivity_loss_damage.py --case-dir simulations/0910
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Try to import pandas for CSV export (optional)
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def parse_lammpstrj_header(lines: List[str], start_idx: int) -> Tuple[int, Dict]:
    """
    Parse LAMMPS dump file header starting at line start_idx.
    Returns (next_line_idx, header_dict)
    """
    header = {}
    idx = start_idx

    # ITEM: TIMESTEP
    if not lines[idx].startswith("ITEM: TIMESTEP"):
        raise ValueError("Expected 'ITEM: TIMESTEP'")
    idx += 1
    header["timestep"] = int(lines[idx].strip())
    idx += 1

    # ITEM: NUMBER OF ATOMS
    if not lines[idx].startswith("ITEM: NUMBER OF ATOMS"):
        raise ValueError("Expected 'ITEM: NUMBER OF ATOMS'")
    idx += 1
    header["natoms"] = int(lines[idx].strip())
    idx += 1

    # ITEM: BOX BOUNDS
    if not lines[idx].startswith("ITEM: BOX BOUNDS"):
        raise ValueError("Expected 'ITEM: BOX BOUNDS'")
    bounds_line = lines[idx].strip()
    idx += 1

    box_data = [list(map(float, lines[idx].split())) for _ in range(3)]
    header["box"] = box_data
    idx += 3

    # ITEM: ATOMS
    if not lines[idx].startswith("ITEM: ATOMS"):
        raise ValueError("Expected 'ITEM: ATOMS'")
    atom_header = lines[idx].strip()
    header["columns"] = atom_header.split()[2:]  # Skip "ITEM: ATOMS"
    idx += 1

    return idx, header


def parse_lammpstrj_frame(lines: List[str], start_idx: int) -> Tuple[int, Dict]:
    """
    Parse one complete frame from LAMMPS dump file.
    Returns (next_frame_start_idx, frame_dict)
    """
    idx, header = parse_lammpstrj_header(lines, start_idx)

    particles = {}
    for atom_idx in range(header["natoms"]):
        parts = lines[idx].split()
        pid = int(parts[0])

        # Map columns to values
        data = {}
        for col_idx, col_name in enumerate(header["columns"]):
            try:
                data[col_name] = float(parts[col_idx + 1])
            except (ValueError, IndexError):
                data[col_name] = (
                    parts[col_idx + 1] if col_idx + 1 < len(parts) else None
                )

        particles[pid] = data
        idx += 1

    return idx, {
        "timestep": header["timestep"],
        "particles": particles,
        "box": header["box"],
    }


def read_lammpstrj_file(filepath: Path) -> List[Dict]:
    """
    Read all frames from a LAMMPS trajectory file.
    Returns list of frame dictionaries.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    frames = []
    idx = 0
    while idx < len(lines):
        try:
            idx, frame = parse_lammpstrj_frame(lines, idx)
            frames.append(frame)
        except (ValueError, IndexError) as e:
            break

    return frames


def calculate_neighbors(
    positions: Dict[int, Tuple[float, float, float]],
    cutoff: float = 3.6,
    box: List[List[float]] = None,
) -> Dict[int, int]:
    """
    Calculate number of neighbors for each particle within cutoff distance.
    Uses periodic boundary conditions if box is provided.

    Args:
        positions: Dict of {particle_id: (x, y, z)}
        cutoff: Neighbor distance cutoff (micrometers)
        box: [[xlo, xhi], [ylo, yhi], [zlo, zhi]] for PBC

    Returns:
        Dict of {particle_id: neighbor_count}
    """
    neighbors = {pid: 0 for pid in positions}
    pids = sorted(positions.keys())

    for i, pid_i in enumerate(pids):
        for pid_j in pids[i + 1 :]:
            pos_i = np.array(positions[pid_i])
            pos_j = np.array(positions[pid_j])

            # Calculate distance with PBC if applicable
            delta = pos_j - pos_i

            if box is not None:
                for dim in range(3):
                    box_len = box[dim][1] - box[dim][0]
                    if delta[dim] > box_len / 2:
                        delta[dim] -= box_len
                    elif delta[dim] < -box_len / 2:
                        delta[dim] += box_len

            distance = np.linalg.norm(delta)

            if distance < cutoff:
                neighbors[pid_i] += 1
                neighbors[pid_j] += 1

    return neighbors


def analyze_connectivity_loss(case_dir: Path, cutoff: float = 3.6) -> Dict:
    """
    Main analysis function.
    Computes connectivity loss (damage) for all frames in dump file.
    """
    dump_file = case_dir / "dump.calendering.lammpstrj"

    if not dump_file.exists():
        raise FileNotFoundError(f"Dump file not found: {dump_file}")

    print(f"Reading dump file: {dump_file}")
    frames = read_lammpstrj_file(dump_file)
    print(f"Loaded {len(frames)} frames")

    if len(frames) == 0:
        raise ValueError("No frames found in dump file")

    # Extract positions from first frame (reference/initial state)
    initial_frame = frames[0]
    initial_positions = {
        pid: (data["x"], data["y"], data["z"])
        for pid, data in initial_frame["particles"].items()
    }
    initial_box = initial_frame["box"]

    # Calculate initial neighbor count for each particle
    print("Computing initial neighbor counts...")
    initial_neighbors = calculate_neighbors(initial_positions, cutoff, initial_box)

    mean_initial = np.mean(list(initial_neighbors.values()))
    print(f"Mean initial neighbor count: {mean_initial:.2f}")

    # Analyze all frames
    results = {
        "frames": [],
        "timesteps": [],
        "avg_damage": [],
        "max_damage": [],
        "min_damage": [],
        "particles_with_loss": [],
        "particles_severely_damaged": [],
    }

    print("\nAnalyzing frames...")
    for frame_idx, frame in enumerate(frames):
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{len(frames)}")

        timestep = frame["timestep"]
        current_positions = {
            pid: (data["x"], data["y"], data["z"])
            for pid, data in frame["particles"].items()
        }
        current_box = frame["box"]

        # Calculate current neighbor count
        current_neighbors = calculate_neighbors(current_positions, cutoff, current_box)

        # Compute damage for each particle
        damages = {}
        for pid in initial_neighbors:
            if initial_neighbors[pid] > 0:
                damage = max(
                    0.0, 1.0 - current_neighbors.get(pid, 0) / initial_neighbors[pid]
                )
                damages[pid] = damage
            else:
                damages[pid] = 0.0

        # Compute statistics
        damage_values = list(damages.values())
        avg_damage = np.mean(damage_values)
        max_damage = np.max(damage_values)
        min_damage = np.min(damage_values)

        particles_with_loss = sum(1 for d in damage_values if d > 0.1)
        particles_severely_damaged = sum(1 for d in damage_values if d > 0.5)

        results["frames"].append(frame_idx)
        results["timesteps"].append(timestep)
        results["avg_damage"].append(avg_damage)
        results["max_damage"].append(max_damage)
        results["min_damage"].append(min_damage)
        results["particles_with_loss"].append(particles_with_loss)
        results["particles_severely_damaged"].append(particles_severely_damaged)

    print("\nAnalysis complete!")
    print(f"Final average damage: {results['avg_damage'][-1]:.3f}")
    print(
        f"Final severely damaged particles: {results['particles_severely_damaged'][-1]}"
    )

    return results, initial_neighbors


def save_results(case_dir: Path, results: Dict, initial_neighbors: Dict) -> None:
    """Save analysis results to CSV and JSON files."""
    output_dir = case_dir / "damage_analysis"
    output_dir.mkdir(exist_ok=True)

    # Save as CSV if pandas available
    if HAS_PANDAS:
        df = pd.DataFrame(results)
        csv_file = output_dir / "connectivity_loss_damage.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved CSV: {csv_file}")

    # Save as JSON
    json_file = output_dir / "connectivity_loss_damage.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: {json_file}")

    # Save initial neighbors
    initial_file = output_dir / "initial_neighbors.json"
    with open(initial_file, "w", encoding="utf-8") as f:
        json.dump(initial_neighbors, f, indent=2)
    print(f"Saved initial neighbors: {initial_file}")


def plot_results(case_dir: Path, results: Dict) -> None:
    """Generate plots of damage evolution."""
    output_dir = case_dir / "damage_analysis"
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Connectivity Loss Damage Analysis", fontsize=14, fontweight="bold")

    # Plot 1: Average damage evolution
    ax = axes[0, 0]
    ax.plot(results["timesteps"], results["avg_damage"], "b-", linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average Damage")
    ax.set_title("Global Average Damage")
    ax.grid(True, alpha=0.3)

    # Plot 2: Max/min damage envelope
    ax = axes[0, 1]
    ax.fill_between(
        results["timesteps"], results["min_damage"], results["max_damage"], alpha=0.3
    )
    ax.plot(results["timesteps"], results["max_damage"], "r-", label="Max", linewidth=2)
    ax.plot(results["timesteps"], results["min_damage"], "g-", label="Min", linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Damage")
    ax.set_title("Damage Range (Min/Max)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Particle count with damage
    ax = axes[1, 0]
    ax.plot(
        results["timesteps"],
        results["particles_with_loss"],
        "orange",
        label="Damage > 10%",
        linewidth=2,
    )
    ax.plot(
        results["timesteps"],
        results["particles_severely_damaged"],
        "red",
        label="Damage > 50%",
        linewidth=2,
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of Particles")
    ax.set_title("Damaged Particle Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Damage rate (derivative)
    ax = axes[1, 1]
    avg_damage_array = np.array(results["avg_damage"])
    damage_rate = np.gradient(avg_damage_array, results["timesteps"])
    ax.plot(results["timesteps"], damage_rate, "purple", linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Damage Rate (d/dt)")
    ax.set_title("Rate of Damage Accumulation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / "connectivity_loss_damage.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze connectivity loss damage from LAMMPS dump files"
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Path to simulation case directory (e.g., simulations/0910)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=3.6,
        help="Neighbor distance cutoff in micrometers (default: 3.6)",
    )

    args = parser.parse_args()
    case_dir = args.case_dir

    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    print(f"\n{'=' * 70}")
    print(f"Connectivity Loss Damage Analysis")
    print(f"Case: {case_dir}")
    print(f"Neighbor cutoff: {args.cutoff} µm")
    print(f"{'=' * 70}\n")

    # Run analysis
    results, initial_neighbors = analyze_connectivity_loss(case_dir, args.cutoff)

    # Save results
    save_results(case_dir, results, initial_neighbors)

    # Generate plots
    plot_results(case_dir, results)

    print(f"\nResults saved to: {case_dir / 'damage_analysis'}")
    print("\nSummary:")
    print(f"  Initial mean neighbors: {np.mean(list(initial_neighbors.values())):.2f}")
    print(
        f"  Final average damage: {results['avg_damage'][-1]:.3f} ({results['avg_damage'][-1] * 100:.1f}%)"
    )
    print(f"  Max damage observed: {max(results['max_damage']):.3f}")
    print(
        f"  Particles severely damaged: {results['particles_severely_damaged'][-1]}/{len(initial_neighbors)}"
    )


if __name__ == "__main__":
    main()
