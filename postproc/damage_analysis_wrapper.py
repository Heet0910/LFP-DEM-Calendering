"""
Damage Analysis Wrapper
======================

Provides integration layer for connectivity loss damage analysis.
Can be called from run_calendering.py or ui_calendering.py after simulations complete.

Usage:
    from postproc.damage_analysis_wrapper import run_damage_analysis

    case_dir = Path("simulations/0910")
    results, summary = run_damage_analysis(case_dir)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# Setup logging
logger = logging.getLogger(__name__)


def calculate_neighbors(
    positions: Dict[int, Tuple[float, float, float]],
    cutoff: float = 3.6,
    box: Optional[List[List[float]]] = None,
) -> Dict[int, int]:
    """
    Calculate number of neighbors for each particle within cutoff distance.
    Uses periodic boundary conditions if box is provided.
    """
    neighbors = {pid: 0 for pid in positions}
    pids = sorted(positions.keys())

    for i, pid_i in enumerate(pids):
        for pid_j in pids[i + 1 :]:
            pos_i = np.array(positions[pid_i])
            pos_j = np.array(positions[pid_j])

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


def parse_lammpstrj_header(lines: List[str], start_idx: int) -> Tuple[int, Dict]:
    """Parse LAMMPS dump file header."""
    header = {}
    idx = start_idx

    if not lines[idx].startswith("ITEM: TIMESTEP"):
        raise ValueError("Expected 'ITEM: TIMESTEP'")
    idx += 1
    header["timestep"] = int(lines[idx].strip())
    idx += 1

    if not lines[idx].startswith("ITEM: NUMBER OF ATOMS"):
        raise ValueError("Expected 'ITEM: NUMBER OF ATOMS'")
    idx += 1
    header["natoms"] = int(lines[idx].strip())
    idx += 1

    if not lines[idx].startswith("ITEM: BOX BOUNDS"):
        raise ValueError("Expected 'ITEM: BOX BOUNDS'")
    idx += 1

    box_data = [list(map(float, lines[idx].split())) for _ in range(3)]
    header["box"] = box_data
    idx += 3

    if not lines[idx].startswith("ITEM: ATOMS"):
        raise ValueError("Expected 'ITEM: ATOMS'")
    atom_header = lines[idx].strip()
    header["columns"] = atom_header.split()[2:]
    idx += 1

    return idx, header


def parse_lammpstrj_frame(lines: List[str], start_idx: int) -> Tuple[int, Dict]:
    """Parse one complete frame from LAMMPS dump file."""
    idx, header = parse_lammpstrj_header(lines, start_idx)

    particles = {}
    for atom_idx in range(header["natoms"]):
        parts = lines[idx].split()
        pid = int(parts[0])

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


def read_lammpstrj_file(filepath: Path, max_frames: Optional[int] = None) -> List[Dict]:
    """
    Read frames from LAMMPS trajectory file.

    Args:
        filepath: Path to dump file
        max_frames: Maximum frames to read (None = all)

    Returns:
        List of frame dictionaries
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read dump file {filepath}: {e}")
        raise

    frames = []
    idx = 0
    while idx < len(lines):
        try:
            idx, frame = parse_lammpstrj_frame(lines, idx)
            frames.append(frame)

            if max_frames and len(frames) >= max_frames:
                break
        except (ValueError, IndexError):
            break

    return frames


def analyze_connectivity_loss(
    case_dir: Path,
    cutoff: float = 3.6,
    max_frames: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """
    Main analysis function.

    Args:
        case_dir: Path to simulation case directory
        cutoff: Neighbor distance cutoff in micrometers
        max_frames: Maximum frames to analyze (None = all)

    Returns:
        (results_dict, initial_neighbors_dict)
    """
    dump_file = case_dir / "dump.calendering.lammpstrj"

    if not dump_file.exists():
        logger.warning(f"Dump file not found: {dump_file}")
        return {}, {}

    logger.info(f"Reading dump file: {dump_file.name}")
    frames = read_lammpstrj_file(dump_file, max_frames)
    logger.info(f"Loaded {len(frames)} frames")

    if len(frames) == 0:
        logger.warning("No frames found in dump file")
        return {}, {}

    # Extract initial state
    initial_frame = frames[0]
    initial_positions = {
        pid: (data["x"], data["y"], data["z"])
        for pid, data in initial_frame["particles"].items()
    }
    initial_box = initial_frame["box"]

    # Calculate initial neighbors
    logger.info("Computing initial neighbor counts...")
    initial_neighbors = calculate_neighbors(initial_positions, cutoff, initial_box)
    mean_initial = np.mean(list(initial_neighbors.values()))
    logger.info(f"Mean initial neighbor count: {mean_initial:.2f}")

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

    logger.info(f"Analyzing {len(frames)} frames...")
    for frame_idx, frame in enumerate(frames):
        if frame_idx % max(1, len(frames) // 10) == 0:
            logger.info(f"  Frame {frame_idx}/{len(frames)}")

        timestep = frame["timestep"]
        current_positions = {
            pid: (data["x"], data["y"], data["z"])
            for pid, data in frame["particles"].items()
        }
        current_box = frame["box"]

        current_neighbors = calculate_neighbors(current_positions, cutoff, current_box)

        damages = {}
        for pid in initial_neighbors:
            if initial_neighbors[pid] > 0:
                damage = max(
                    0.0, 1.0 - current_neighbors.get(pid, 0) / initial_neighbors[pid]
                )
                damages[pid] = damage
            else:
                damages[pid] = 0.0

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

    logger.info("Analysis complete!")
    logger.info(f"Final average damage: {results['avg_damage'][-1]:.3f}")
    logger.info(
        f"Final severely damaged particles: {results['particles_severely_damaged'][-1]}"
    )

    return results, initial_neighbors


def save_results(case_dir: Path, results: Dict, initial_neighbors: Dict) -> Path:
    """
    Save analysis results to files.

    Returns:
        Path to output directory
    """
    output_dir = case_dir / "damage_analysis"
    output_dir.mkdir(exist_ok=True)

    try:
        # Save as CSV if pandas available
        if HAS_PANDAS and results:
            df = pd.DataFrame(results)
            csv_file = output_dir / "connectivity_loss_damage.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved CSV: {csv_file}")
    except Exception as e:
        logger.warning(f"Could not save CSV: {e}")

    try:
        # Save as JSON
        json_file = output_dir / "connectivity_loss_damage.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved JSON: {json_file}")
    except Exception as e:
        logger.warning(f"Could not save JSON: {e}")

    try:
        # Save initial neighbors
        initial_file = output_dir / "initial_neighbors.json"
        with open(initial_file, "w", encoding="utf-8") as f:
            json.dump(initial_neighbors, f, indent=2)
        logger.info(f"Saved initial neighbors: {initial_file}")
    except Exception as e:
        logger.warning(f"Could not save initial neighbors: {e}")

    return output_dir


def plot_results(case_dir: Path, results: Dict, output_dir: Path) -> None:
    """Generate diagnostic plots."""
    if not HAS_MATPLOTLIB or not results:
        logger.warning("Matplotlib not available or no results to plot")
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            "Connectivity Loss Damage Analysis", fontsize=14, fontweight="bold"
        )

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
            results["timesteps"],
            results["min_damage"],
            results["max_damage"],
            alpha=0.3,
        )
        ax.plot(
            results["timesteps"], results["max_damage"], "r-", label="Max", linewidth=2
        )
        ax.plot(
            results["timesteps"], results["min_damage"], "g-", label="Min", linewidth=2
        )
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

        # Plot 4: Damage rate
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
        logger.info(f"Saved plot: {plot_file}")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")


def create_summary(results: Dict, initial_neighbors: Dict) -> Dict:
    """
    Create a summary of damage metrics for display.

    Returns:
        Dictionary with key damage metrics
    """
    if not results or not initial_neighbors:
        return {}

    return {
        "initial_mean_neighbors": float(np.mean(list(initial_neighbors.values()))),
        "final_average_damage": float(results["avg_damage"][-1]),
        "final_average_damage_percent": float(results["avg_damage"][-1] * 100),
        "max_damage_observed": float(max(results["max_damage"])),
        "particles_severely_damaged_final": int(
            results["particles_severely_damaged"][-1]
        ),
        "total_particles": len(initial_neighbors),
        "particles_with_any_loss_final": int(results["particles_with_loss"][-1]),
    }


def run_damage_analysis(
    case_dir: Path,
    cutoff: float = 3.6,
    generate_plots: bool = True,
    max_frames: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """
    Run complete damage analysis workflow on a completed simulation case.

    This is the main entry point for integration with run_calendering.py and ui_calendering.py.

    Args:
        case_dir: Path to simulation case directory
        cutoff: Neighbor distance cutoff in micrometers (default: 3.6)
        generate_plots: Whether to generate diagnostic plots (default: True)
        max_frames: Maximum frames to analyze (None = all)

    Returns:
        (results_dict, summary_dict)

    Example:
        >>> from postproc.damage_analysis_wrapper import run_damage_analysis
        >>> results, summary = run_damage_analysis(Path("simulations/0910"))
        >>> print(f"Final damage: {summary['final_average_damage_percent']:.1f}%")
    """
    logger.info(f"Starting damage analysis for {case_dir.name}")

    try:
        # Run analysis
        results, initial_neighbors = analyze_connectivity_loss(
            case_dir, cutoff, max_frames
        )

        if not results:
            logger.warning("No analysis results generated")
            return {}, {}

        # Save results
        output_dir = save_results(case_dir, results, initial_neighbors)

        # Generate plots if requested
        if generate_plots:
            plot_results(case_dir, results, output_dir)

        # Create summary
        summary = create_summary(results, initial_neighbors)

        logger.info(f"Damage analysis complete for {case_dir.name}")

        return results, summary

    except Exception as e:
        logger.error(f"Error during damage analysis: {e}")
        return {}, {}
