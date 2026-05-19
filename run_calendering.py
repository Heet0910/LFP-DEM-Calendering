from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from input.geometry_materials_config import (
    BondParams,
    ContactParams,
    DomainConfig,
    LiteratureAnchors,
    ParticleTypes,
    ResearchBaselineProfile,
)

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"
SIM_DIR = PROJECT_ROOT / "simulations"


# ---------------------------------------------------------------------------
# Case configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseConfig:
    label: str = "lfp_r25"
    top_start: float | None = None
    top_clearance: float = 2.0
    settling_steps: int = 20_000
    compression_ratio: float | None = 0.25
    compression_velocity: float = -LiteratureAnchors.plane_speed_um_per_us
    compression_steps: int | None = None
    compression_displacement: float | None = None
    hold_steps: int = 5_000
    decompression_velocity: float = LiteratureAnchors.plane_speed_um_per_us
    decompression_steps: int | None = None
    thermo_every: int = 500
    dump_every: int = 2_000
    timestep_us: float = LiteratureAnchors.timestep_us
    gravity_um_per_us2: float = ContactParams.gravity_um_per_us2
    areal_loading_mg_cm2: float = ParticleTypes.target_areal_loading_mg_cm2
    bonded_fracture: bool = False
    regenerate_structure: bool = False
    prepare_only: bool = False
    allow_high_risk: bool = False
    # Fraction of compression distance to decompress (1.0 = full elastic springback)
    # Models a gap-set calender: wall stops at target calendered gap.
    # Literature values (Schreiner 2020, Ngandjong 2021):
    #   NMC cathode: 4-37% (typ. 1-3% hot-calendered, 10-20% cold)
    #   LFP cathode: ~10-20% (hard sub-micron particles, no plastic slip)
    #   Graphite anode: ~5%
    # Default 0.15 = 15% springback → literature midpoint for cold-rolled LFP
    partial_decompression_fraction: float = 0.15

    @classmethod
    def from_dict(cls, data: dict) -> "CaseConfig":
        """Construct from a JSON-loaded dict, ignoring unknown keys.

        Previously run_lammps() manually listed every field, making it
        fragile whenever a new field was added.  This classmethod reads
        only the field names that exist on the dataclass so the JSON
        may contain extra provenance keys without causing a TypeError.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Research references (written into case_config.json for provenance)
# ---------------------------------------------------------------------------

RESEARCH_REFERENCES: dict[str, str] = {
    "dem_foundation": "Cundall & Strack 1979, Geotechnique, doi:10.1680/geot.1979.29.1.47",
    "bonded_particle_model": "Potyondy & Cundall 2004, IJRMMS, doi:10.1016/j.ijrmms.2004.09.011",
    "battery_calendering_dem": "Ngandjong et al. 2021, J Power Sources 485, 229320, doi:10.1016/j.jpowsour.2020.229320",
    "dem_calendering_parameterization": "Schreiner et al. 2021, Procedia CIRP 104, 91-97, doi:10.1016/j.procir.2021.11.016",
    "dem_tomography": "Ge et al. 2022, Powder Technology 403, 117366, doi:10.1016/j.powtec.2022.117366",
    "bpm_lammps_style": "Clemmer et al. 2024, Soft Matter 20, 1702-1718, doi:10.1039/D3SM01373A",
    "sjkr_cohesion": "Johnson, Kendall & Roberts 1971, Proc. R. Soc. London A 324, 301-313",
}


# ---------------------------------------------------------------------------
# LAMMPS executable discovery
# ---------------------------------------------------------------------------

def resolve_lammps_exe() -> str:
    exe = shutil.which("lmp")
    if exe:
        return exe
    candidates = [
        PROJECT_ROOT / "LAMMPS 64-bit 22Jul2025" / "bin" / "lmp.exe",
        PROJECT_ROOT / "LAMMPS" / "bin" / "lmp.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "lmp"


def lammps_cmd(script_name: str, n_threads: int | None = None) -> list[str]:
    """Build a LAMMPS command with OpenMP threading enabled.

    Uses all logical threads by default (os.cpu_count()).
    Flags: -sf omp -pk omp N  → runs OpenMP-accelerated pair/bond styles.

    Note: this build uses MPI STUBS (single-process), so parallelism is
    via OpenMP only.  OMP_NUM_THREADS must also be set in the environment
    — done by run_lammps() when calling subprocess.run.
    """
    import os
    exe = resolve_lammps_exe()
    n = n_threads or os.cpu_count() or 1
    return [exe, "-sf", "omp", "-pk", "omp", str(n), "-in", script_name]


def _lammps_env(n_threads: int | None = None) -> dict:
    """Return environment dict with OMP_NUM_THREADS set.

    If OMP_NUM_THREADS is already in the environment (e.g. set by the
    calling shell), that value wins — allows multiple parallel sweeps,
    each with a reduced thread count (e.g. set OMP_NUM_THREADS=4 in
    each terminal when running two sweeps simultaneously on 8 threads).
    """
    import os
    n = n_threads or os.cpu_count() or 1
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(n))  # shell value takes priority
    return env


# ---------------------------------------------------------------------------
# Template rendering (main settling + calendering script)
# ---------------------------------------------------------------------------

def _build_bond_blocks(config: CaseConfig) -> tuple[str, str, str, str, str, str]:
    """Return (bond_setup_block, bond_create_block, dump_cols,
               thermo_extra, pressurelog_extra, cleanup_extra).

    3-phase PTFE dry electrode bond types:
      Bond 1: AM–CB   (LFP ↔ Carbon Black aggregate)
      Bond 2: AM–PTFE (LFP ↔ PTFE fibril)
      Bond 3: CB–PTFE (Carbon Black ↔ PTFE fibril junction)
      Bond 4: CB–CB   (direct Carbon Black aggregate contact)
    """
    create_commands: list[str] = []
    if config.bonded_fracture:
        if BondParams.enable_am_am_bonds:
            create_commands.append(
                "create_bonds    many AM AM 1 0.0 __BOND_AM_AM_CUTOFF__"
            )
        if BondParams.enable_am_cb_bonds:
            create_commands.append(
                "create_bonds    many AM CB 1 0.0 __BOND_AM_CB_CUTOFF__"
            )
        if BondParams.enable_am_ptfe_bonds:
            create_commands.append(
                "create_bonds    many AM PTFE 2 0.0 __BOND_AM_PTFE_CUTOFF__"
            )
        if BondParams.enable_cb_ptfe_bonds:
            create_commands.append(
                "create_bonds    many CB PTFE 3 0.0 __BOND_CB_PTFE_CUTOFF__"
            )
        if BondParams.enable_cb_cb_bonds:
            create_commands.append(
                "create_bonds    many CB CB 4 0.0 __BOND_CB_CB_CUTOFF__"
            )

    # Shared preamble: always declare the nbond compute + placeholder variable
    _shared_preamble = [
        "compute         nbond all nbond/atom",
        "variable        active_bonds equal 0.0",
    ]

    # Full bond-style block (4 bond types for 3-phase system)
    _bond_style_block = [
        "special_bonds   lj 0.0 1.0 1.0 coul 0.0 1.0 1.0",
        "bond_style      bpm/spring/plastic break yes store/local brkbond __THERMO_EVERY__ time id1 id2 x y z",
        "bond_coeff      1 __BOND_AM_CB_COEFFS__",   # AM-CB: moderate
        "bond_coeff      2 __BOND_AM_PTFE_COEFFS__", # AM-PTFE: ductile PTFE fibril
        "bond_coeff      3 __BOND_CB_PTFE_COEFFS__", # CB-PTFE: most ductile (fibril network)
        "bond_coeff      4 __BOND_CB_CB_COEFFS__",   # CB-CB: brittle
        *create_commands,
        "reset_atoms     image all",
        "special_bonds   lj 0.0 1.0 1.0 coul 1.0 1.0 1.0",
        "compute         total_nbond all reduce sum c_nbond",
        "variable        active_bonds delete",
        "variable        active_bonds equal c_total_nbond/2.0",
        "dump            broken all local __THERMO_EVERY__ dump.broken_bonds.local "
        "f_brkbond[1] f_brkbond[2] f_brkbond[3] f_brkbond[4] f_brkbond[5] f_brkbond[6]",
    ]

    if not config.bonded_fracture:
        bond_setup_block = "\n".join(_shared_preamble)
        bond_create_block = ""
        cleanup_extra = ""
    elif BondParams.create_bonds_after_settling:
        # Bond style defined up-front (pre-settling) but bonds created after settling
        bond_setup_block = "\n".join(_shared_preamble)
        bond_create_block = "\n".join(_bond_style_block)
        cleanup_extra = "\nundump broken"
    else:
        # Bonds created before settling: merge preamble + style block into setup
        bond_setup_block = "\n".join(_shared_preamble + _bond_style_block)
        bond_create_block = ""
        cleanup_extra = "\nundump broken"

    dump_columns = "id type x y z radius c_nbond"
    thermo_extra = " v_active_bonds"
    pressurelog_extra = " ${active_bonds}"
    return (
        bond_setup_block,
        bond_create_block,
        dump_columns,
        thermo_extra,
        pressurelog_extra,
        cleanup_extra,
    )


def _validate_no_unreplaced_tokens(text: str, context: str) -> None:
    """Raise if any __TOKEN__ placeholders survive after substitution."""
    leftovers = re.findall(r"__[A-Z_0-9]+__", text)
    if leftovers:
        raise RuntimeError(
            f"Unreplaced template tokens in {context}: {leftovers}\n"
            "Check that all __TOKEN__ keys are present in the replacements dict."
        )


def render_template(config: CaseConfig) -> str:
    template_path = INPUT_DIR / "in.calendering_template.liggghts"
    text = template_path.read_text(encoding="utf-8")

    (
        bond_setup_block,
        bond_create_block,
        dump_columns,
        thermo_extra,
        pressurelog_extra,
        cleanup_extra,
    ) = _build_bond_blocks(config)

    replacements = {
        "__TOP_START__": f"{config.top_start}",
        "__LX__": f"{DomainConfig.lx_um}",
        "__LY__": f"{DomainConfig.ly_um}",
        "__BOUNDARY__": (
            "p p f"
            if DomainConfig.periodic_x and DomainConfig.periodic_y and not DomainConfig.periodic_z
            else "f f f"
        ),
        "__SETTLING_STEPS__": str(config.settling_steps),
        "__COMPRESSION_VELOCITY__": f"{config.compression_velocity}",
        "__COMPRESSION_STEPS__": str(config.compression_steps),
        "__HOLD_STEPS__": str(config.hold_steps),
        "__DECOMPRESSION_VELOCITY__": f"{config.decompression_velocity}",
        "__DECOMPRESSION_STEPS__": str(config.decompression_steps),
        "__THERMO_EVERY__": str(config.thermo_every),
        "__DUMP_EVERY__": str(config.dump_every),
        "__TIMESTEP_US__": f"{config.timestep_us}",
        "__GRAVITY__": f"{config.gravity_um_per_us2}",
        "__SETTLING_DAMP__": f"{ContactParams.settling_viscous_damping}",
        "__PROCESS_DAMP__": f"{ContactParams.process_viscous_damping}",
        "__DECOMP_DAMP__": f"{ContactParams.decompression_viscous_damping}",
        "__HERTZ_KN__": f"{ContactParams.hertz_kn}",
        "__HERTZ_GAMMA_N__": f"{ContactParams.hertz_gamma_n}",
        "__HERTZ_FRICTION__": f"{ContactParams.hertz_friction}",
        "__JKR_GAMMA__": f"{ContactParams.jkr_gamma}",
        "__SJKR_CED__": f"{ContactParams.jkr_gamma}",  # legacy token alias
        "__READ_DATA_ARGS__": (
            f"extra/bond/per/atom {BondParams.extra_bonds_per_atom} "
            f"extra/special/per/atom {BondParams.extra_special_per_atom}"
        ),
        # ── 3-phase bond tokens ────────────────────────────────────────────────
        "__BOND_AM_AM_CUTOFF__":    f"{BondParams.am_am_cutoff_um}",
        "__BOND_AM_CB_CUTOFF__":    f"{BondParams.am_cb_cutoff_um}",
        "__BOND_AM_PTFE_CUTOFF__":  f"{BondParams.am_ptfe_cutoff_um}",
        "__BOND_CB_PTFE_CUTOFF__":  f"{BondParams.cb_ptfe_cutoff_um}",
        "__BOND_CB_CB_CUTOFF__":    f"{BondParams.cb_cb_cutoff_um}",
        "__BOND_AM_CB_COEFFS__":    " ".join(str(v) for v in BondParams.am_cb_coeffs),
        "__BOND_AM_PTFE_COEFFS__":  " ".join(str(v) for v in BondParams.am_ptfe_coeffs),
        "__BOND_CB_PTFE_COEFFS__":  " ".join(str(v) for v in BondParams.cb_ptfe_coeffs),
        "__BOND_CB_CB_COEFFS__":    " ".join(str(v) for v in BondParams.cb_cb_coeffs),
        # ── Legacy aliases (2-phase backward compat) ───────────────────────────
        "__BOND_AM_CBD_CUTOFF__":   f"{BondParams.am_cb_cutoff_um}",
        "__BOND_CBD_CBD_CUTOFF__":  f"{BondParams.cb_cb_cutoff_um}",
        "__BOND_AM_AM_COEFFS__":    " ".join(str(v) for v in BondParams.am_am_coeffs),
        "__BOND_AM_CBD_COEFFS__":   " ".join(str(v) for v in BondParams.am_cb_coeffs),
        "__BOND_CBD_CBD_COEFFS__":  " ".join(str(v) for v in BondParams.cb_cb_coeffs),
        "__BOND_SETUP_BLOCK__": bond_setup_block,
        "__BOND_CREATE_BLOCK__": bond_create_block,
        "__BOND_RELAX_STEPS__": str(BondParams.bond_relax_steps),
        "__DUMP_COLUMNS__": dump_columns,
        "__THERMO_EXTRA__": thermo_extra,
        "__PRESSURELOG_EXTRA__": pressurelog_extra,
        "__CLEANUP_EXTRA__": cleanup_extra,
        "__SPRINGBACK_FRACTION__": f"{config.partial_decompression_fraction}",
        "__SPRINGBACK_FRACTION_PCT__": f"{config.partial_decompression_fraction * 100.0:.0f}",
    }

    # First pass: replace all direct tokens
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Second pass: replace tokens introduced inside block substitutions
    # (e.g. __BOND_SETUP_BLOCK__ itself contains __THERMO_EVERY__ etc.)
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    _validate_no_unreplaced_tokens(text, "in.calendering_template.liggghts")
    return text
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    _validate_no_unreplaced_tokens(text, "in.calendering_template.liggghts")
    return text


# ---------------------------------------------------------------------------
# Structure generation
# ---------------------------------------------------------------------------

def maybe_regenerate_structure(config: CaseConfig) -> None:
    generator_path = INPUT_DIR / "generate_initial_electrode.py"
    if not generator_path.exists():
        raise FileNotFoundError("Missing input/generate_initial_electrode.py")
    subprocess.run(
        [
            sys.executable,            # portable interpreter
            str(generator_path),
            "--areal-loading-mg-cm2", str(config.areal_loading_mg_cm2),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )


# ---------------------------------------------------------------------------
# Structure statistics
# ---------------------------------------------------------------------------

def read_structure_stats(data_path: Path) -> dict[str, float]:
    atoms_section = False
    z_top = float("-inf")
    z_bottom = float("inf")
    total_mass = 0.0

    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip().lower()
            if stripped.startswith("atoms #"):
                atoms_section = True
                continue
            if not atoms_section or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 7:
                break

            if len(parts) >= 8:
                diameter = float(parts[3])
                density = float(parts[4])
                z = float(parts[7])
            else:
                diameter = float(parts[2])
                density = float(parts[3])
                z = float(parts[6])

            radius = 0.5 * diameter
            z_top = max(z_top, z + radius)
            z_bottom = min(z_bottom, z - radius)
            # Use math.pi (not the hardcoded literal that was here before)
            total_mass += density * (4.0 / 3.0) * math.pi * radius**3

    if z_top == float("-inf") or z_bottom == float("inf"):
        raise ValueError(f"Could not read particle bed from {data_path}")

    bed_height = z_top - z_bottom
    return {
        "z_top": z_top,
        "z_bottom": z_bottom,
        "bed_height": bed_height,
        "bulk_density": total_mass / (DomainConfig.area_um2() * bed_height),
        "total_mass": total_mass,
    }


# ---------------------------------------------------------------------------
# Dump-file parsing
# ---------------------------------------------------------------------------

def parse_single_frame_dump(
    path: Path,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        if not handle.readline().startswith("ITEM: TIMESTEP"):
            raise ValueError(f"Unexpected dump format in {path}")
        handle.readline()
        if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Unexpected dump format in {path}")
        n_atoms = int(handle.readline().strip())

        if not handle.readline().startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"Unexpected dump format in {path}")
        x_bounds = handle.readline().split()
        y_bounds = handle.readline().split()
        z_bounds = handle.readline().split()

        atoms_header = handle.readline().strip()
        if not atoms_header.startswith("ITEM: ATOMS"):
            raise ValueError(f"Unexpected dump format in {path}")

        columns = atoms_header.split()[2:]
        indices = {name: idx for idx, name in enumerate(columns)}
        required = ("id", "type", "x", "y", "z", "radius")
        missing = [name for name in required if name not in indices]
        if missing:
            raise ValueError(f"Missing dump columns {missing} in {path}")

        particles: list[dict[str, float]] = []
        for _ in range(n_atoms):
            parts = handle.readline().split()
            particle: dict[str, float] = {
                "pid": int(parts[indices["id"]]),
                "type_id": int(parts[indices["type"]]),
                "x": float(parts[indices["x"]]),
                "y": float(parts[indices["y"]]),
                "z": float(parts[indices["z"]]),
                "radius": float(parts[indices["radius"]]),
            }
            if "c_nbond" in indices:
                particle["nbond"] = float(parts[indices["c_nbond"]])
            particles.append(particle)

    box = {
        "xlo": float(x_bounds[0]), "xhi": float(x_bounds[1]),
        "ylo": float(y_bounds[0]), "yhi": float(y_bounds[1]),
        "zlo": float(z_bounds[0]), "zhi": float(z_bounds[1]),
    }
    return particles, box


def compute_snapshot_height(particles: list[dict[str, float]]) -> dict[str, float]:
    z_bottom = min(p["z"] - p["radius"] for p in particles)
    z_top = max(p["z"] + p["radius"] for p in particles)
    return {"z_bottom": z_bottom, "z_top": z_top, "bed_height": z_top - z_bottom}


# ---------------------------------------------------------------------------
# Periodic geometry helpers
# ---------------------------------------------------------------------------

def minimum_image(delta: float, box_length: float) -> float:
    if box_length <= 0.0:
        return delta
    return delta - box_length * round(delta / box_length)


# ---------------------------------------------------------------------------
# Bond network construction from snapshots
# ---------------------------------------------------------------------------

def build_bonds_from_snapshot(
    particles: list[dict[str, float]],
    box: dict[str, float],
) -> list[tuple[int, int, int, int]]:
    """O(N²) bond search for 3-phase system (AM=1, CB=2, PTFE=3).

    Bond types:
      1 = AM–CB   (type pair {1,2})
      2 = AM–PTFE (type pair {1,3})
      3 = CB–PTFE (type pair {2,3})
      4 = CB–CB   (type pair {2,2})
    AM–AM bonds (type pair {1,1}) are only created if enable_am_am_bonds=True.

    For large systems (>5000 particles) replace with a cell-list to get O(N).
    """
    bonds: list[tuple[int, int, int, int]] = []
    bond_id = 1
    x_length = box["xhi"] - box["xlo"]
    y_length = box["yhi"] - box["ylo"]

    for i, particle_i in enumerate(particles):
        type_i = int(particle_i["type_id"])
        for particle_j in particles[i + 1:]:
            type_j = int(particle_j["type_id"])
            bond_type: int | None = None
            tolerance: float = 0.0
            pair = frozenset([type_i, type_j])

            if type_i == 1 and type_j == 1:
                # AM–AM (disabled by default)
                if not BondParams.enable_am_am_bonds:
                    continue
                bond_type, tolerance = 0, BondParams.am_am_gap_tolerance_um  # no LAMMPS bond type for AM-AM in 3-phase
            elif pair == frozenset([1, 2]):
                # AM–CB  → bond type 1
                if not BondParams.enable_am_cb_bonds:
                    continue
                bond_type, tolerance = 1, BondParams.am_cb_gap_tolerance_um
            elif pair == frozenset([1, 3]):
                # AM–PTFE → bond type 2
                if not BondParams.enable_am_ptfe_bonds:
                    continue
                bond_type, tolerance = 2, BondParams.am_ptfe_gap_tolerance_um
            elif pair == frozenset([2, 3]):
                # CB–PTFE → bond type 3
                if not BondParams.enable_cb_ptfe_bonds:
                    continue
                bond_type, tolerance = 3, BondParams.cb_ptfe_gap_tolerance_um
            elif type_i == 2 and type_j == 2:
                # CB–CB → bond type 4
                if not BondParams.enable_cb_cb_bonds:
                    continue
                bond_type, tolerance = 4, BondParams.cb_cb_gap_tolerance_um
            else:
                continue

            if bond_type is None or bond_type == 0:
                continue

            dx = minimum_image(particle_i["x"] - particle_j["x"], x_length)
            dy = minimum_image(particle_i["y"] - particle_j["y"], y_length)
            dz = particle_i["z"] - particle_j["z"]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= particle_i["radius"] + particle_j["radius"] + tolerance:
                bonds.append((bond_id, bond_type, int(particle_i["pid"]), int(particle_j["pid"])))
                bond_id += 1

    return bonds


def write_bonded_restart_data(
    source_dump: Path,
    target_data: Path,
) -> dict[str, float]:
    particles, box = parse_single_frame_dump(source_dump)
    bonds = build_bonds_from_snapshot(particles, box)

    def wrap_periodic(value: float, lo: float, hi: float) -> float:
        length = hi - lo
        if length <= 0.0:
            return value
        shift = value - lo
        n = math.floor(shift / length)
        wrapped = value - n * length
        if wrapped < lo:
            wrapped += length
        elif wrapped >= hi:
            wrapped -= length
        return wrapped

    xlo, xhi = float(box["xlo"]), float(box["xhi"])
    ylo, yhi = float(box["ylo"]), float(box["yhi"])
    x_length = xhi - xlo
    y_length = yhi - ylo

    for particle in particles:
        particle["_x0"] = float(particle["x"])
        particle["_y0"] = float(particle["y"])

    pid_to_idx = {int(p["pid"]): i for i, p in enumerate(particles)}
    image: dict[int, tuple[int, int, int]] = {int(p["pid"]): (0, 0, 0) for p in particles}

    shifts: dict[tuple[int, int], tuple[int, int]] = {}
    for _, _, atom_i, atom_j in bonds:
        pi = particles[pid_to_idx[atom_i]]
        pj = particles[pid_to_idx[atom_j]]
        raw_dx = float(pi["_x0"]) - float(pj["_x0"])
        raw_dy = float(pi["_y0"]) - float(pj["_y0"])
        min_dx = minimum_image(raw_dx, x_length)
        min_dy = minimum_image(raw_dy, y_length)
        sx = int(round((raw_dx - min_dx) / x_length)) if x_length > 0.0 else 0
        sy = int(round((raw_dy - min_dy) / y_length)) if y_length > 0.0 else 0
        shifts[(atom_i, atom_j)] = (sx, sy)
        shifts[(atom_j, atom_i)] = (-sx, -sy)

    # Union-find with potentials for consistent image flags
    class _PotentialDSU:
        def __init__(self, nodes: list[int]) -> None:
            self.parent: dict[int, int] = {n: n for n in nodes}
            self.rank: dict[int, int] = {n: 0 for n in nodes}
            self.potential: dict[int, int] = {n: 0 for n in nodes}

        def find(self, x: int) -> tuple[int, int]:
            if self.parent[x] == x:
                return x, 0
            root, pot_to_root = self.find(self.parent[x])
            pot = self.potential[x] + pot_to_root
            self.parent[x] = root
            self.potential[x] = pot
            return root, pot

        def union(self, a: int, b: int, b_minus_a: int) -> bool:
            ra, pa = self.find(a)
            rb, pb = self.find(b)
            if ra == rb:
                return (pb - pa) == b_minus_a
            if self.rank[ra] < self.rank[rb]:
                self.parent[ra] = rb
                self.potential[ra] = pb - pa - b_minus_a
            else:
                self.parent[rb] = ra
                self.potential[rb] = b_minus_a + pa - pb
                if self.rank[ra] == self.rank[rb]:
                    self.rank[ra] += 1
            return True

        def value(self, x: int) -> int:
            _, pot = self.find(x)
            return pot

    nodes = [int(p["pid"]) for p in particles]
    dsu_x = _PotentialDSU(nodes)
    dsu_y = _PotentialDSU(nodes)
    for _, _, atom_i, atom_j in bonds:
        sx, sy = shifts.get((atom_i, atom_j), (0, 0))
        dsu_x.union(atom_i, atom_j, sx)
        dsu_y.union(atom_i, atom_j, sy)
    for pid in nodes:
        image[pid] = (dsu_x.value(pid), dsu_y.value(pid), 0)

    for particle in particles:
        particle["x"] = wrap_periodic(float(particle["_x0"]), xlo, xhi)
        particle["y"] = wrap_periodic(float(particle["_y0"]), ylo, yhi)
        ix, iy, iz = image[int(particle["pid"])]
        particle["_ix"] = ix
        particle["_iy"] = iy
        particle["_iz"] = iz

    lines = [
        "LAMMPS data file via run_calendering.py bonded restart (3-phase PTFE dry electrode)", "",
        f"{len(particles)} atoms",
        f"{len(bonds)} bonds",
        "3 atom types",   # 1=AM, 2=CB, 3=PTFE
        "4 bond types", "",  # 1=AM-CB, 2=AM-PTFE, 3=CB-PTFE, 4=CB-CB
        f"{xlo} {xhi} xlo xhi",
        f"{ylo} {yhi} ylo yhi",
        f"{box['zlo']} {box['zhi']} zlo zhi", "",
        "Atoms # bpm/sphere", "",
    ]
    _density_by_type = {
        1: ParticleTypes.am_density_g_cm3,
        2: ParticleTypes.cb_density_g_cm3,
        3: ParticleTypes.ptfe_density_g_cm3,
    }
    for particle in sorted(particles, key=lambda item: int(item["pid"])):
        type_id = int(particle["type_id"])
        density = _density_by_type.get(type_id, ParticleTypes.am_density_g_cm3)
        diameter = 2.0 * particle["radius"]
        ix = int(particle.get("_ix", 0))
        iy = int(particle.get("_iy", 0))
        iz = int(particle.get("_iz", 0))
        lines.append(
            f"{int(particle['pid'])} 1 {type_id} {diameter:.6f} {density:.6f} "
            f"{particle['x']:.6f} {particle['y']:.6f} {particle['z']:.6f} {ix} {iy} {iz}"
        )

    lines.extend(["", "Bonds", ""])
    for bond_id, bond_type, atom_i, atom_j in bonds:
        lines.append(f"{bond_id} {bond_type} {atom_i} {atom_j}")

    target_data.write_text("\n".join(lines) + "\n", encoding="utf-8")
    snapshot_stats = compute_snapshot_height(particles)
    snapshot_stats["bond_count"] = float(len(bonds))
    return snapshot_stats


# ---------------------------------------------------------------------------
# Bonded restart template (inline f-string LAMMPS script)
# ---------------------------------------------------------------------------

def render_bonded_restart_template(
    config: CaseConfig,
    top_start: float,
    *,
    data_filename: str,
    breakable: bool,
    relax_only: bool,
    pressure_log_name: str = "pressure_log.txt",
    broken_dump_name: str = "dump.broken_bonds.local",
    traj_dump_name: str = "dump.calendering.lammpstrj",
    initial_snapshot_name: str = "snapshot.bonded_initial.lammpstrj",
    compressed_snapshot_name: str = "snapshot.compressed.lammpstrj",
    final_snapshot_name: str = "final.calendered.lammpstrj",
    write_initial_snapshot: bool = True,
    write_compressed_snapshot: bool = True,
    write_final_snapshot: bool = True,
) -> str:
    dump_columns = "id type x y z radius c_nbond"
    bond_style_line = (
        f"bond_style      bpm/spring/plastic break yes store/local brkbond "
        f"{config.thermo_every} time id1 id2 x y z"
        if breakable
        else "bond_style      bpm/spring/plastic break no smooth no"
    )
    broken_dump_block = (
        f"dump broken all local {config.thermo_every} {broken_dump_name} "
        f"f_brkbond[1] f_brkbond[2] f_brkbond[3] f_brkbond[4] f_brkbond[5] f_brkbond[6]"
        if breakable
        else ""
    )
    pressurelog_block = (
        f'fix pressurelog all print {config.thermo_every} '
        f'"${{simstep}} ${{disp_top}} ${{top_force}} ${{pressure}} ${{active_bonds}}" '
        f'file {pressure_log_name} screen no title "step disp_top top_force pressure active_bonds"'
    )
    traj_dump_block = (
        f"dump traj all custom {config.dump_every} {traj_dump_name} {dump_columns}\n"
        "dump_modify traj sort id"
        if breakable
        else ""
    )
    initial_snapshot_block = (
        f"write_dump all custom {initial_snapshot_name} {dump_columns}"
        if write_initial_snapshot else ""
    )
    compressed_snapshot_block = (
        f"write_dump all custom {compressed_snapshot_name} {dump_columns}"
        if write_compressed_snapshot else ""
    )
    final_snapshot_block = (
        f"write_dump all custom {final_snapshot_name} {dump_columns}"
        if write_final_snapshot else ""
    )

    if relax_only:
        run_block = f"""run 0
run {BondParams.bond_relax_steps}
run 0
{initial_snapshot_block}

  unfix pressure_guard
  unfix pressurelog
  unfix damp
  unfix integrator
  """
    else:
        pct = config.partial_decompression_fraction * 100.0   # springback % for comments

        # Stage 5 block: only included when there is an actual decompression phase
        if config.decompression_steps and config.decompression_steps > 0:
            stage5_block = f"""
# Stage 5: Free equilibration — remove top wall, let material settle naturally
# Must remove the compute referencing f_top before unfixing top wall
unfix pressure_guard
unfix pressurelog
uncompute top_force_sum
unfix top
thermo_style custom step atoms
run {config.hold_steps}

# Final state: material freely settled after {pct:.0f}% natural springback (wall removed)
{final_snapshot_block}

"""
            stage5_cleanup = f"""    {"undump traj" if traj_dump_block else ""}
    {"undump broken" if breakable else ""}
    unfix integrator
    unfix damp
    """
        else:
            # Settle-only: no decompression, no Stage 5
            stage5_block = f"""
{final_snapshot_block}
"""
            stage5_cleanup = f"""    {"undump traj" if traj_dump_block else ""}
    {"undump broken" if breakable else ""}
    unfix pressure_guard
    unfix pressurelog
    unfix integrator
    unfix damp
    """

        run_block = f"""
run 0
{initial_snapshot_block}

variable v equal {config.compression_velocity}
variable disp_top equal v_v*elapsed*dt
run {config.compression_steps}
{compressed_snapshot_block}

variable disp_top equal ${{v}}*{config.compression_steps}*dt
run {config.hold_steps}

# Snapshot at max compression (reference state — full calendering load)
write_dump all custom snapshot.at_max_compression.lammpstrj {dump_columns}
dump_modify traj sort id

# Stage 4: Partial decompression — wall lifts {pct:.0f}% of compression distance
# Literature LFP cold-rolled springback: 10-20% (Schreiner 2020, Ngandjong 2021)
unfix damp
fix damp all viscous {ContactParams.decompression_viscous_damping}

variable v2 equal {config.decompression_velocity}
variable disp_top equal ${{v}}*{config.compression_steps}*dt + v_v2*elapsed*dt
run {config.decompression_steps}
{stage5_block}{stage5_cleanup}"""

    # Wall fixes use Hertz only (no SJKR on walls — binder is inter-particle only)
    wall_fix = (
        f"hertz/history {ContactParams.hertz_kn} NULL "
        f"{ContactParams.hertz_gamma_n} NULL {ContactParams.hertz_friction} 1"
    )
    boundary = (
        "p p f"
        if DomainConfig.periodic_x and DomainConfig.periodic_y and not DomainConfig.periodic_z
        else "f f f"
    )

    return f"""units           micro
dimension       3
boundary        {boundary}

atom_style      bpm/sphere
special_bonds   lj 0.0 1.0 1.0 coul 1.0 1.0 1.0
newton          on off

read_data       {data_filename} extra/bond/per/atom {BondParams.extra_bonds_per_atom} extra/special/per/atom {BondParams.extra_special_per_atom}

reset_timestep  0
timestep        {config.timestep_us}

group           AM   type 1
group           CB   type 2
group           PTFE type 3

pair_style      gran/hertz/history {ContactParams.hertz_kn} NULL {ContactParams.hertz_gamma_n} NULL {ContactParams.hertz_friction} 1
pair_coeff      * *

neighbor        0.001 bin
neigh_modify    delay 0
comm_modify     cutoff 7.0 vel yes

{bond_style_line}
bond_coeff      1 {" ".join(str(v) for v in BondParams.am_cb_coeffs)}
bond_coeff      2 {" ".join(str(v) for v in BondParams.am_ptfe_coeffs)}
bond_coeff      3 {" ".join(str(v) for v in BondParams.cb_ptfe_coeffs)}
bond_coeff      4 {" ".join(str(v) for v in BondParams.cb_cb_coeffs)}
reset_atoms     image all
velocity all set 0 0 0
compute         nbond all nbond/atom
compute         total_nbond all reduce sum c_nbond
variable        active_bonds equal c_total_nbond/2.0

    fix             integrator all nve/bpm/sphere
    fix             damp all viscous {BondParams.bond_relax_viscous_damping}

variable ztop0 equal {top_start}
variable lx equal {DomainConfig.lx_um}
variable ly equal {DomainConfig.ly_um}
variable disp_top equal 0.0

region bottom_wall plane 0 0 0 0 0 1 side in units box
region top_wall plane 0 0 ${{ztop0}} 0 0 -1 side in move NULL NULL v_disp_top units box
region left_wall plane 0 0 0 1 0 0 side in units box
region right_wall plane {DomainConfig.lx_um} 0 0 -1 0 0 side in units box
region front_wall plane 0 0 0 0 1 0 side in units box
region back_wall plane 0 {DomainConfig.ly_um} 0 0 -1 0 side in units box

fix bottom all wall/gran/region {wall_fix} region bottom_wall contacts
fix top    all wall/gran/region {wall_fix} region top_wall contacts
fix left   all wall/gran/region {wall_fix} region left_wall contacts
fix right  all wall/gran/region {wall_fix} region right_wall contacts
fix front  all wall/gran/region {wall_fix} region front_wall contacts
fix back   all wall/gran/region {wall_fix} region back_wall contacts

compute top_force_sum all reduce sum f_top[4]
variable top_force equal abs(c_top_force_sum)
variable pressure equal v_top_force/(lx*ly)
variable simstep equal step
variable pressure_abort equal 10000.0

thermo {config.thermo_every}
thermo_style custom step atoms v_disp_top v_top_force v_pressure v_active_bonds
thermo_modify    lost error lost/bond warn
fix pressure_guard all halt {config.thermo_every} v_pressure > ${{pressure_abort}} error hard

{pressurelog_block}
{traj_dump_block}
{broken_dump_block}
{run_block}
"""


# ---------------------------------------------------------------------------
# Pressure-log helpers
# ---------------------------------------------------------------------------

def parse_pressure_log(
    path: Path,
) -> tuple[str, list[tuple[int, float, float, float, float]]]:
    header = "step disp_top top_force pressure active_bonds"
    rows: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return header, rows
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip()
        if first:
            header = first
        for line in handle:
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append((
                int(float(parts[0])),
                float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]),
            ))
    return header, rows


def write_pressure_log(
    path: Path,
    header: str,
    rows: list[tuple[int, float, float, float, float]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{header}\n")
        for step, disp_top, top_force, pressure, active_bonds in rows:
            handle.write(
                f"{step} {disp_top:.12g} {top_force:.12g} "
                f"{pressure:.12g} {active_bonds:.12g}\n"
            )


def merge_pressure_logs(
    preload_log: Path,
    fracture_log: Path,
    output_log: Path,
    *,
    preload_step_offset: int,
    preload_disp_offset: float,
) -> None:
    header_pre, preload_rows = parse_pressure_log(preload_log)
    header_frac, fracture_rows = parse_pressure_log(fracture_log)
    header = header_pre or header_frac
    merged = list(preload_rows)
    for step, disp_top, top_force, pressure, active_bonds in fracture_rows:
        merged.append((
            step + preload_step_offset,
            disp_top + preload_disp_offset,
            top_force, pressure, active_bonds,
        ))
    write_pressure_log(output_log, header, merged)


# ---------------------------------------------------------------------------
# Configuration finalisation
# ---------------------------------------------------------------------------

def finalize_config(config: CaseConfig, data_path: Path) -> CaseConfig:
    stats = read_structure_stats(data_path)
    top_start = (
        config.top_start
        if config.top_start is not None
        else stats["z_top"] + config.top_clearance
    )

    compression_displacement = config.compression_displacement
    if compression_displacement is None:
        if config.compression_ratio is None:
            raise ValueError("Specify either compression_ratio or compression_displacement.")
        target_min_height = max(stats["bed_height"] * (1.0 - config.compression_ratio), 1.0e-6)
        target_top_position = stats["z_bottom"] + target_min_height
        # top_start includes top_clearance (air gap above bed).
        # compression_displacement must cover clearance + actual bed reduction.
        # This is naturally correct: top_start = z_top + clearance,
        # target_top_position = z_bottom + target_height (no clearance).
        # So compression_displacement = (z_top + clearance) - (z_bottom + target_height)
        #                             = bed_height - target_height + clearance
        # We only want bed_height - target_height (pure bed compression),
        # so subtract the clearance to avoid over-compressing.
        clearance = top_start - stats["z_top"]   # actual clearance used
        compression_displacement = top_start - target_top_position - clearance

    if compression_displacement <= 0.0:
        raise ValueError("compression_displacement must be positive after finalisation.")

    compression_steps = config.compression_steps
    if compression_steps is None:
        speed = abs(config.compression_velocity)
        if speed <= 0.0:
            raise ValueError("compression_velocity must be non-zero when compression_steps is omitted.")
        compression_steps = max(
            int(round(compression_displacement / (speed * config.timestep_us))), 1
        )

    decompression_steps = config.decompression_steps
    if decompression_steps is None:
        speed = abs(config.decompression_velocity)
        if speed <= 0.0:
            raise ValueError("decompression_velocity must be non-zero when decompression_steps is omitted.")
        # Partial decompression: only lift by fraction of compression distance.
        # Models a gap-set calender — wall stops at the target calendered thickness.
        # Default fraction=0.05 → lift 5% of compression → 95% height retention.
        decomp_distance = compression_displacement * config.partial_decompression_fraction
        decompression_steps = max(
            int(round(decomp_distance / (speed * config.timestep_us))), 1
        )

    # Safety checks for bonded-fracture runs — single guard block, single return
    if config.bonded_fracture and not config.allow_high_risk:
        dt = config.timestep_us
        if dt > 1.0e-4:
            raise ValueError("For bonded fracture runs, use timestep_us <= 1e-4.")
        max_step_disp = 5.0e-7
        if abs(config.compression_velocity) * dt > max_step_disp:
            raise ValueError(
                f"Reduce compression_velocity or timestep_us: "
                f"|v|*dt = {abs(config.compression_velocity) * dt:.3g} > {max_step_disp:.3g} um/step."
            )
        if abs(config.decompression_velocity) * dt > max_step_disp:
            raise ValueError(
                f"Reduce decompression_velocity or timestep_us: "
                f"|v2|*dt = {abs(config.decompression_velocity) * dt:.3g} > {max_step_disp:.3g} um/step."
            )
        if (
            config.compression_ratio is not None
            and config.compression_ratio > ResearchBaselineProfile.compression_ratio
        ):
            raise ValueError(
                f"Compression ratio {config.compression_ratio:.2f} exceeds research baseline "
                f"({ResearchBaselineProfile.compression_ratio:.2f}). Use --allow-high-risk to override."
            )

    # Single CaseConfig construction — eliminates the earlier duplicate
    return CaseConfig(
        label=config.label,
        top_start=top_start,
        top_clearance=config.top_clearance,
        settling_steps=config.settling_steps,
        compression_ratio=config.compression_ratio,
        compression_velocity=config.compression_velocity,
        compression_steps=compression_steps,
        compression_displacement=compression_displacement,
        hold_steps=config.hold_steps,
        decompression_velocity=config.decompression_velocity,
        decompression_steps=decompression_steps,
        thermo_every=config.thermo_every,
        dump_every=config.dump_every,
        timestep_us=config.timestep_us,
        gravity_um_per_us2=config.gravity_um_per_us2,
        areal_loading_mg_cm2=config.areal_loading_mg_cm2,
        bonded_fracture=config.bonded_fracture,
        regenerate_structure=config.regenerate_structure,
        prepare_only=config.prepare_only,
        allow_high_risk=config.allow_high_risk,
    )


# ---------------------------------------------------------------------------
# Simulation folder preparation
# ---------------------------------------------------------------------------

def prepare_sim_folder(config: CaseConfig) -> Path:
    case_dir = SIM_DIR / config.label
    case_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-loading structure cache ────────────────────────────────────────────
    # Each unique loading level gets its own electrode file so that:
    #   1. Different loadings produce physically distinct structures.
    #   2. Parallel sweeps don't overwrite each other's structures.
    # Cache key: loading rounded to 2 dp, e.g. "8.00", "11.36", "16.00"
    loading_key = f"{config.areal_loading_mg_cm2:.2f}".replace(".", "p")
    data_src      = INPUT_DIR / f"data.electrode_{loading_key}"
    metadata_src  = INPUT_DIR / f"structure_metadata_{loading_key}.json"

    # Canonical (non-keyed) paths expected by maybe_regenerate_structure
    data_canonical     = INPUT_DIR / "data.electrode"
    metadata_canonical = INPUT_DIR / "structure_metadata.json"

    if config.regenerate_structure or not data_src.exists():
        # Generate into canonical paths, then save to the keyed cache
        maybe_regenerate_structure(config)
        import shutil as _shutil
        _shutil.copy2(data_canonical,     data_src)
        if metadata_canonical.exists():
            _shutil.copy2(metadata_canonical, metadata_src)

    data_dst    = case_dir / "data.electrode"
    metadata_dst = case_dir / "structure_metadata.json"
    in_dst      = case_dir / "in.calendering.liggghts"

    final_config = finalize_config(config, data_src)

    data_dst.write_bytes(data_src.read_bytes())
    if metadata_src.exists():
        metadata_dst.write_bytes(metadata_src.read_bytes())
    in_dst.write_text(render_template(final_config), encoding="utf-8")

    structure_stats = read_structure_stats(data_src)
    case_metadata = asdict(final_config)
    case_metadata["initial_bed_height_um"] = structure_stats["bed_height"]
    case_metadata["initial_bulk_density_g_cm3"] = structure_stats["bulk_density"]
    case_metadata["initial_mass_pg"] = structure_stats["total_mass"]
    case_metadata["areal_loading_mg_cm2"] = (
        structure_stats["total_mass"] / DomainConfig.area_um2() / 10.0
    )
    case_metadata["contact_model"] = {
        "hertz_kn": ContactParams.hertz_kn,
        "hertz_gamma_n": ContactParams.hertz_gamma_n,
        "hertz_friction": ContactParams.hertz_friction,
        "jkr_gamma": ContactParams.jkr_gamma,
        "note": "Hertz-Mindlin + SJKR cohesion (Schreiner et al. 2021 Table 2)",
    }
    case_metadata["bonded_fracture"] = final_config.bonded_fracture
    case_metadata["bond_creation_cutoffs_um"] = {
        "am_am":   BondParams.am_am_cutoff_um,
        "am_cb":   BondParams.am_cb_cutoff_um,
        "am_ptfe": BondParams.am_ptfe_cutoff_um,
        "cb_ptfe": BondParams.cb_ptfe_cutoff_um,
        "cb_cb":   BondParams.cb_cb_cutoff_um,
    }
    case_metadata["bond_creation_stage"] = (
        "after_settling" if BondParams.create_bonds_after_settling else "before_settling"
    )
    case_metadata["bond_topology"] = {
        "am_am":   BondParams.enable_am_am_bonds,
        "am_cb":   BondParams.enable_am_cb_bonds,
        "am_ptfe": BondParams.enable_am_ptfe_bonds,
        "cb_ptfe": BondParams.enable_cb_ptfe_bonds,
        "cb_cb":   BondParams.enable_cb_cb_bonds,
    }
    case_metadata["bond_coefficients"] = {
        "am_am":   BondParams.am_am_coeffs,
        "am_cb":   BondParams.am_cb_coeffs,
        "am_ptfe": BondParams.am_ptfe_coeffs,
        "cb_ptfe": BondParams.cb_ptfe_coeffs,
        "cb_cb":   BondParams.cb_cb_coeffs,
    }
    case_metadata["research_references"] = RESEARCH_REFERENCES
    case_metadata["research_baseline_profile"] = {
        "compression_ratio": ResearchBaselineProfile.compression_ratio,
        "compression_velocity_um_per_us": ResearchBaselineProfile.compression_velocity_um_per_us,
        "decompression_velocity_um_per_us": ResearchBaselineProfile.decompression_velocity_um_per_us,
        "timestep_us": ResearchBaselineProfile.timestep_us,
    }
    (case_dir / "case_config.json").write_text(
        json.dumps(case_metadata, indent=2), encoding="utf-8"
    )
    return case_dir


# ---------------------------------------------------------------------------
# LAMMPS execution (bonded + unbonded workflows)
# ---------------------------------------------------------------------------

def run_lammps(case_dir: Path) -> None:
    """Run LAMMPS for a prepared case directory.

    Previously this called os.chdir(case_dir) which permanently mutated the
    process working directory.  Now every subprocess.run call passes
    cwd=case_dir instead, leaving the parent process's cwd unchanged.
    """
    case_config = json.loads(
        (case_dir / "case_config.json").read_text(encoding="utf-8")
    )
    # Use CaseConfig.from_dict so that adding new fields to CaseConfig
    # does not require updating this call site manually.
    config = CaseConfig.from_dict(case_config)
    lammps_exe = resolve_lammps_exe()

    if not config.bonded_fracture:
        cmd = [lammps_exe, "-in", "in.calendering.liggghts"]
        print("Running:", " ".join(cmd), "in", case_dir)
        subprocess.run(cmd, check=True, cwd=case_dir)
        return

    bonds_enabled = (
        BondParams.enable_am_am_bonds
        or BondParams.enable_am_cb_bonds
        or BondParams.enable_am_ptfe_bonds
        or BondParams.enable_cb_ptfe_bonds
        or BondParams.enable_cb_cb_bonds
    )
    if not bonds_enabled:
        print(
            "Bonded fracture requested, but all bond types are disabled; "
            "running unbonded workflow instead."
        )
        cmd = lammps_cmd("in.calendering.liggghts")
        print("Running:", " ".join(cmd), "in", case_dir)
        subprocess.run(cmd, check=True, cwd=case_dir)
        return

    # --- Stage 1: settling only (no compression) ---
    settle_config = CaseConfig(
        **{**asdict(config), "bonded_fracture": False,
           "compression_steps": 0, "hold_steps": 0, "decompression_steps": 0}
    )
    settle_input = case_dir / "in.settle_only.liggghts"
    settle_input.write_text(render_template(settle_config), encoding="utf-8")
    settle_cmd = lammps_cmd(settle_input.name)
    print("Running:", " ".join(settle_cmd), "in", case_dir)
    subprocess.run(settle_cmd, check=True, cwd=case_dir, env=_lammps_env())

    settled_snapshot = case_dir / "snapshot.compressed.lammpstrj"
    bonded_data = case_dir / "data.bonded_initial"
    settled_stats = write_bonded_restart_data(settled_snapshot, bonded_data)

    top_start = settled_stats["z_top"] + config.top_clearance
    if config.compression_displacement is not None:
        compression_displacement = config.compression_displacement
    else:
        if config.compression_ratio is None:
            raise ValueError("Specify either compression_ratio or compression_displacement.")
        target_min_height = max(
            settled_stats["bed_height"] * (1.0 - config.compression_ratio), 1.0e-6
        )
        compression_displacement = top_start - (settled_stats["z_bottom"] + target_min_height)

    speed = abs(config.compression_velocity)
    compression_steps = (
        config.compression_steps
        if config.compression_steps is not None
        else max(int(round(compression_displacement / (speed * config.timestep_us))), 1)
    )
    decompression_steps = (
        config.decompression_steps
        if config.decompression_steps is not None
        else max(int(round(compression_displacement / (speed * config.timestep_us))), 1)
    )

    bonded_restart_config = CaseConfig(
        **{**asdict(config), "top_start": top_start,
           "compression_displacement": compression_displacement,
           "compression_steps": compression_steps,
           "decompression_steps": decompression_steps,
           "settling_steps": 0}
    )

    # --- Stage 2: bond relaxation ---
    bond_relax_input = case_dir / "in.bond_relax.liggghts"
    bond_relax_input.write_text(
        render_bonded_restart_template(
            bonded_restart_config, top_start,
            data_filename="data.bonded_initial",
            breakable=False, relax_only=True,
        ),
        encoding="utf-8",
    )
    bond_relax_cmd = lammps_cmd(bond_relax_input.name)
    print("Running:", " ".join(bond_relax_cmd), "in", case_dir)
    subprocess.run(bond_relax_cmd, check=True, cwd=case_dir, env=_lammps_env())

    relaxed_snapshot = case_dir / "snapshot.bonded_initial.lammpstrj"
    relaxed_bonded_data = case_dir / "data.bonded_relaxed"
    relaxed_stats = write_bonded_restart_data(relaxed_snapshot, relaxed_bonded_data)

    process_top_start = relaxed_stats["z_top"] + min(
        config.top_clearance, BondParams.process_contact_clearance_um
    )
    if config.compression_displacement is not None:
        process_compression_displacement = config.compression_displacement
    else:
        process_target_min_height = max(
            relaxed_stats["bed_height"] * (1.0 - config.compression_ratio), 1.0e-6
        )
        process_compression_displacement = (
            process_top_start - (relaxed_stats["z_bottom"] + process_target_min_height)
        )

    process_compression_steps = (
        config.compression_steps
        if config.compression_steps is not None
        else max(int(round(process_compression_displacement / (speed * config.timestep_us))), 1)
    )
    process_decompression_steps = (
        config.decompression_steps
        if config.decompression_steps is not None
        else max(int(round(process_compression_displacement / (speed * config.timestep_us))), 1)
    )

    process_config = CaseConfig(
        **{**asdict(bonded_restart_config),
           "top_start": process_top_start,
           "compression_displacement": process_compression_displacement,
           "compression_steps": process_compression_steps,
           "decompression_steps": process_decompression_steps}
    )

    damage_activation_fraction = min(
        max(BondParams.damage_activation_fraction_of_compression, 0.0), 1.0
    )
    preload_steps = min(
        int(round(process_compression_steps * damage_activation_fraction)),
        process_compression_steps,
    )
    damage_steps = process_compression_steps - preload_steps

    # --- Stage 3: calendering with fracture ---
    if preload_steps > 0 and damage_steps > 0:
        non_calendered_snapshot = case_dir / "snapshot.bonded_initial.reference.lammpstrj"
        shutil.copyfile(relaxed_snapshot, non_calendered_snapshot)

        preload_config = CaseConfig(
            **{**asdict(process_config),
               "compression_steps": preload_steps, "hold_steps": 0, "decompression_steps": 0}
        )
        preload_input = case_dir / "in.pre_damage_compaction.liggghts"
        preload_input.write_text(
            render_bonded_restart_template(
                preload_config, top_start=process_top_start,
                data_filename="data.bonded_relaxed", breakable=False, relax_only=False,
                pressure_log_name="pressure_log_preload.txt",
                traj_dump_name="dump.pre_damage_compaction.lammpstrj",
                initial_snapshot_name="snapshot.pre_damage_start.lammpstrj",
                compressed_snapshot_name="snapshot.pre_damage_end.lammpstrj",
                final_snapshot_name="snapshot.pre_damage_end.lammpstrj",
            ),
            encoding="utf-8",
        )
        preload_cmd = lammps_cmd(preload_input.name)
        print("Running:", " ".join(preload_cmd), "in", case_dir)
        subprocess.run(preload_cmd, check=True, cwd=case_dir, env=_lammps_env())

        pre_damage_snapshot = case_dir / "snapshot.pre_damage_end.lammpstrj"
        damage_bonded_data = case_dir / "data.bonded_pre_damage"
        pre_damage_stats = write_bonded_restart_data(pre_damage_snapshot, damage_bonded_data)
        damage_top_start = pre_damage_stats["z_top"] + min(
            config.top_clearance, BondParams.process_contact_clearance_um
        )
        preload_disp = abs(config.compression_velocity) * config.timestep_us * preload_steps
        damage_config = CaseConfig(
            **{**asdict(process_config),
               "top_start": damage_top_start,
               "compression_displacement": max(process_compression_displacement - preload_disp, 0.0),
               "compression_steps": damage_steps}
        )
        bonded_input = case_dir / "in.calendering.liggghts"
        bonded_input.write_text(
            render_bonded_restart_template(
                damage_config, top_start=damage_top_start,
                data_filename="data.bonded_pre_damage", breakable=True, relax_only=False,
                initial_snapshot_name="snapshot.damage_activation_start.lammpstrj",
            ),
            encoding="utf-8",
        )
        bonded_cmd = lammps_cmd(bonded_input.name)
        print("Running:", " ".join(bonded_cmd), "in", case_dir)
        subprocess.run(bonded_cmd, check=True, cwd=case_dir, env=_lammps_env())

        shutil.copyfile(
            non_calendered_snapshot, case_dir / "snapshot.bonded_initial.lammpstrj"
        )
        merge_pressure_logs(
            case_dir / "pressure_log_preload.txt",
            case_dir / "pressure_log.txt",
            case_dir / "pressure_log_merged.txt",
            preload_step_offset=preload_steps,
            preload_disp_offset=preload_disp,
        )
        shutil.move(
            case_dir / "pressure_log_merged.txt",
            case_dir / "pressure_log.txt",
        )
    else:
        bonded_input = case_dir / "in.calendering.liggghts"
        bonded_input.write_text(
            render_bonded_restart_template(
                process_config, top_start=process_top_start,
                data_filename="data.bonded_relaxed", breakable=True, relax_only=False,
            ),
            encoding="utf-8",
        )
        bonded_cmd = [lammps_exe, "-in", bonded_input.name]
        print("Running:", " ".join(bonded_cmd), "in", case_dir)
        subprocess.run(bonded_cmd, check=True, cwd=case_dir, env=_lammps_env())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> CaseConfig:
    parser = argparse.ArgumentParser(
        description="Prepare or run one LFP calendering case in simulations/<label>."
    )
    parser.add_argument("--label", default="lfp_r25")
    parser.add_argument("--top-start", type=float, default=None)
    parser.add_argument("--top-clearance", type=float, default=2.0)
    parser.add_argument("--settling-steps", type=int, default=20_000)
    parser.add_argument("--compression-ratio", type=float, default=0.25)
    parser.add_argument(
        "--compression-velocity", type=float,
        default=-LiteratureAnchors.plane_speed_um_per_us,
    )
    parser.add_argument("--compression-steps", type=int, default=None)
    parser.add_argument("--compression-displacement", type=float, default=None)
    parser.add_argument("--hold-steps", type=int, default=5_000)
    parser.add_argument(
        "--decompression-velocity", type=float,
        default=LiteratureAnchors.plane_speed_um_per_us,
    )
    parser.add_argument("--decompression-steps", type=int, default=None)
    parser.add_argument(
        "--partial-decompression-fraction", type=float, default=0.15,
        help="Fraction of compression distance to decompress (default 0.15 = 15%%, literature LFP cold-rolling springback)"
    )
    parser.add_argument("--thermo-every", type=int, default=1_000)
    parser.add_argument("--dump-every", type=int, default=2_000)
    parser.add_argument(
        "--timestep-us", type=float, default=LiteratureAnchors.timestep_us
    )
    parser.add_argument(
        "--gravity-um-per-us2", type=float, default=ContactParams.gravity_um_per_us2
    )
    parser.add_argument(
        "--areal-loading-mg-cm2", type=float,
        default=ParticleTypes.target_areal_loading_mg_cm2,
    )
    parser.add_argument("--no-bonded-fracture", action="store_true")
    parser.add_argument("--regenerate-structure", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-high-risk", action="store_true")
    args = parser.parse_args()
    return CaseConfig(
        label=args.label,
        top_start=args.top_start,
        top_clearance=args.top_clearance,
        settling_steps=args.settling_steps,
        compression_ratio=args.compression_ratio,
        compression_velocity=args.compression_velocity,
        compression_steps=args.compression_steps,
        compression_displacement=args.compression_displacement,
        hold_steps=args.hold_steps,
        decompression_velocity=args.decompression_velocity,
        decompression_steps=args.decompression_steps,
        thermo_every=args.thermo_every,
        dump_every=args.dump_every,
        timestep_us=args.timestep_us,
        gravity_um_per_us2=args.gravity_um_per_us2,
        areal_loading_mg_cm2=args.areal_loading_mg_cm2,
        bonded_fracture=not args.no_bonded_fracture,
        regenerate_structure=args.regenerate_structure,
        prepare_only=args.prepare_only,
        allow_high_risk=args.allow_high_risk,
        partial_decompression_fraction=args.partial_decompression_fraction,
    )


def main() -> None:
    config = parse_args()
    case_dir = prepare_sim_folder(config)
    print("Prepared simulation in", case_dir)
    print("Saved case configuration to", case_dir / "case_config.json")
    if config.prepare_only:
        print("Skipping LAMMPS execution because --prepare-only was requested.")
        return
    run_lammps(case_dir)


if __name__ == "__main__":
    main()