from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from postproc.bond_breakage_and_connectivity import analyze_case as analyze_bonds_ab

    HAS_BOND_CONNECTIVITY = True
except Exception:
    HAS_BOND_CONNECTIVITY = False

@dataclass(frozen=True)
class Particle:
    pid: int
    type_id: int
    radius_um: float
    x_um: float
    y_um: float
    z_um: float
    nbond: int | None = None


@dataclass(frozen=True)
class Snapshot:
    particles: list[Particle]
    x_length_um: float
    y_length_um: float
    z_length_um: float
    total_mass_pg: float


@dataclass(frozen=True)
class BondTrackingMetrics:
    bonded_particles: int
    bonded_particle_fraction: float
    mean_nbond: float
    mean_am_nbond: float
    am_zero_bond_particles: int
    am_zero_bond_fraction: float


@dataclass(frozen=True)
class AmNetworkDamageMetrics:
    am_particles_with_initial_links: int
    damaged_am_particles: int
    damaged_am_particle_fraction: float
    mean_am_bond_loss_fraction: float
    max_am_bond_loss_fraction: float
    top_damaged_am_fraction: float
    middle_damaged_am_fraction: float
    bottom_damaged_am_fraction: float


@dataclass(frozen=True)
class FractureMetrics:
    total_contacts: int
    am_involving_contacts: int
    am_am_contacts: int
    am_cbd_contacts: int
    max_overlap_um: float
    max_hertz_pressure_gpa: float
    am_contacts_over_threshold: int
    am_contact_over_threshold_fraction: float
    am_particles_at_risk: int
    am_particle_risk_fraction: float


@dataclass(frozen=True)
class StateMetrics:
    state: str
    source_file: str
    particle_count: int
    z_min_um: float
    z_max_um: float
    bed_height_um: float
    coating_thickness_um: float
    bulk_density_g_cm3: float
    coating_density_from_loading_thickness_g_cm3: float
    porosity: float
    reduction_vs_initial_pct: float
    gravimetric_energy_Wh_kg: float
    volumetric_energy_Wh_L: float
    volumetric_capacity_loading_based_mAh_cm3: float
    volumetric_energy_loading_based_Wh_L: float
    total_contacts: int
    am_involving_contacts: int
    am_am_contacts: int
    am_cbd_contacts: int
    max_overlap_um: float
    max_hertz_pressure_gpa: float
    am_contacts_over_threshold: int
    am_contact_over_threshold_fraction: float
    am_particles_at_risk: int
    am_particle_risk_fraction: float
    bonded_particles: int
    bonded_particle_fraction: float
    mean_nbond: float
    mean_am_nbond: float
    am_zero_bond_particles: int
    am_zero_bond_fraction: float
    am_particles_with_initial_links: int
    damaged_am_particles: int
    damaged_am_particle_fraction: float
    mean_am_bond_loss_fraction: float
    max_am_bond_loss_fraction: float
    top_damaged_am_fraction: float
    middle_damaged_am_fraction: float
    bottom_damaged_am_fraction: float


@dataclass(frozen=True)
class CaseSummary:
    case_label: str
    q_usable_mAh_g: float
    average_voltage_V: float
    fracture_threshold_gpa: float
    initial_coating_thickness_um: float
    min_coating_thickness_um: float
    final_coating_thickness_um: float
    peak_pressure: float
    final_pressure: float
    springback_um: float
    springback_relative_to_min: float
    springback_recovery_fraction: float
    max_active_bonds: float
    final_active_bonds: float
    broken_bond_events: int
    broken_bond_fraction_vs_max_active: float
    bond_loss_damage_threshold_fraction: float
    max_damaged_am_particle_fraction: float
    final_damaged_am_particle_fraction: float
    damage_interpretation: str


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_pressure_log(path: Path) -> list[tuple[float, float, float, float, float]]:
    rows: list[tuple[float, float, float, float, float]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("step"):
                continue
            parts = stripped.split()
            if len(parts) >= 4:
                active_bonds = float(parts[4]) if len(parts) >= 5 else 0.0
                rows.append(
                    (
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        active_bonds,
                    )
                )
    return rows


def parse_broken_bond_dump(path: Path) -> int:
    if not path.exists():
        return 0

    # `dump local` format has repeated ITEM blocks; count only entry rows
    # (numerical data lines) after "ITEM: ENTRIES".
    count = 0
    in_entries = False
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("ITEM:"):
                in_entries = line.startswith("ITEM: ENTRIES")
                continue
            if not in_entries:
                continue
            # Entries are numeric columns (time id1 id2 x y z).
            parts = line.split()
            if len(parts) >= 3:
                count += 1
    return count


def sphere_volume(radius_um: float) -> float:
    return (4.0 / 3.0) * math.pi * radius_um**3


def parse_lammps_data_sphere(path: Path) -> Snapshot:
    atoms_section = False
    particles: list[Particle] = []
    total_mass = 0.0
    x_length = 0.0
    y_length = 0.0
    z_length = 0.0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip().lower()
            if stripped.endswith("xlo xhi"):
                parts = line.split()
                x_length = float(parts[1]) - float(parts[0])
                continue
            if stripped.endswith("ylo yhi"):
                parts = line.split()
                y_length = float(parts[1]) - float(parts[0])
                continue
            if stripped.endswith("zlo zhi"):
                parts = line.split()
                z_length = float(parts[1]) - float(parts[0])
                continue
            if stripped.startswith("atoms #"):
                atoms_section = True
                continue
            if not atoms_section or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 7:
                break

            if len(parts) >= 8:
                pid = int(parts[0])
                type_id = int(parts[2])
                diameter = float(parts[3])
                density = float(parts[4])
                radius = 0.5 * diameter
                x = float(parts[5])
                y = float(parts[6])
                z = float(parts[7])
            else:
                pid = int(parts[0])
                type_id = int(parts[1])
                diameter = float(parts[2])
                density = float(parts[3])
                radius = 0.5 * diameter
                x = float(parts[4])
                y = float(parts[5])
                z = float(parts[6])
            particles.append(Particle(pid=pid, type_id=type_id, radius_um=radius, x_um=x, y_um=y, z_um=z))
            total_mass += density * sphere_volume(radius)

    return Snapshot(
        particles=particles,
        x_length_um=x_length,
        y_length_um=y_length,
        z_length_um=z_length,
        total_mass_pg=total_mass,
    )


def parse_single_frame_dump(path: Path) -> Snapshot:
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
        x_length = float(x_bounds[1]) - float(x_bounds[0])
        y_length = float(y_bounds[1]) - float(y_bounds[0])
        z_length = float(z_bounds[1]) - float(z_bounds[0])

        atoms_header = handle.readline().strip()
        if not atoms_header.startswith("ITEM: ATOMS"):
            raise ValueError(f"Unexpected dump format in {path}")

        columns = atoms_header.split()[2:]
        id_idx = columns.index("id")
        type_idx = columns.index("type")
        x_idx = columns.index("x")
        y_idx = columns.index("y")
        z_idx = columns.index("z")
        r_idx = columns.index("radius")
        nbond_idx = columns.index("c_nbond") if "c_nbond" in columns else None

        particles: list[Particle] = []
        for _ in range(n_atoms):
            parts = handle.readline().split()
            particles.append(
                Particle(
                    pid=int(parts[id_idx]),
                    type_id=int(parts[type_idx]),
                    radius_um=float(parts[r_idx]),
                    x_um=float(parts[x_idx]),
                    y_um=float(parts[y_idx]),
                    z_um=float(parts[z_idx]),
                    nbond=int(float(parts[nbond_idx])) if nbond_idx is not None else None,
                )
            )

    return Snapshot(
        particles=particles,
        x_length_um=x_length,
        y_length_um=y_length,
        z_length_um=z_length,
        total_mass_pg=0.0,
    )


def compute_height(particles: list[Particle]) -> tuple[float, float, float]:
    z_min = min(p.z_um - p.radius_um for p in particles)
    z_max = max(p.z_um + p.radius_um for p in particles)
    return z_min, z_max, z_max - z_min


def minimum_image(delta: float, box_length: float) -> float:
    if box_length <= 0.0:
        return delta
    return delta - box_length * round(delta / box_length)


def equivalent_modulus_gpa(type_i: int, type_j: int) -> float:
    # Literature-inspired effective elastic properties for the fracture proxy.
    e_lfp = 47.3
    nu_lfp = 0.23
    e_cbd = 1.73
    nu_cbd = 0.38

    e_i = e_lfp if type_i == 1 else e_cbd
    nu_i = nu_lfp if type_i == 1 else nu_cbd
    e_j = e_lfp if type_j == 1 else e_cbd
    nu_j = nu_lfp if type_j == 1 else nu_cbd
    compliance = ((1.0 - nu_i * nu_i) / e_i) + ((1.0 - nu_j * nu_j) / e_j)
    return 1.0 / compliance


def hertz_max_pressure_gpa(particle_i: Particle, particle_j: Particle, overlap_um: float) -> float:
    if overlap_um <= 0.0:
        return 0.0
    inverse_radius = (1.0 / particle_i.radius_um) + (1.0 / particle_j.radius_um)
    reduced_radius = 1.0 / inverse_radius
    e_star = equivalent_modulus_gpa(particle_i.type_id, particle_j.type_id)
    return (2.0 * e_star / math.pi) * math.sqrt(overlap_um / reduced_radius)


def compute_fracture_metrics(
    snapshot: Snapshot,
    fracture_threshold_gpa: float,
) -> FractureMetrics:
    particles = snapshot.particles
    total_contacts = 0
    am_involving_contacts = 0
    am_am_contacts = 0
    am_cbd_contacts = 0
    max_overlap = 0.0
    max_pressure = 0.0
    am_contacts_over_threshold = 0
    am_particles_at_risk: set[int] = set()
    total_am_particles = sum(1 for particle in particles if particle.type_id == 1)

    for i, particle_i in enumerate(particles):
        for j in range(i + 1, len(particles)):
            particle_j = particles[j]
            dx = minimum_image(particle_i.x_um - particle_j.x_um, snapshot.x_length_um)
            dy = minimum_image(particle_i.y_um - particle_j.y_um, snapshot.y_length_um)
            dz = particle_i.z_um - particle_j.z_um
            center_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            overlap = particle_i.radius_um + particle_j.radius_um - center_distance
            if overlap <= 0.0:
                continue

            total_contacts += 1
            max_overlap = max(max_overlap, overlap)
            pressure = hertz_max_pressure_gpa(particle_i, particle_j, overlap)
            max_pressure = max(max_pressure, pressure)

            am_in_contact = particle_i.type_id == 1 or particle_j.type_id == 1
            if not am_in_contact:
                continue

            am_involving_contacts += 1
            if particle_i.type_id == 1 and particle_j.type_id == 1:
                am_am_contacts += 1
            else:
                am_cbd_contacts += 1

            if pressure >= fracture_threshold_gpa:
                am_contacts_over_threshold += 1
                if particle_i.type_id == 1:
                    am_particles_at_risk.add(particle_i.pid)
                if particle_j.type_id == 1:
                    am_particles_at_risk.add(particle_j.pid)

    am_contact_fraction = (
        am_contacts_over_threshold / am_involving_contacts if am_involving_contacts > 0 else 0.0
    )
    am_particle_fraction = (
        len(am_particles_at_risk) / total_am_particles if total_am_particles > 0 else 0.0
    )

    return FractureMetrics(
        total_contacts=total_contacts,
        am_involving_contacts=am_involving_contacts,
        am_am_contacts=am_am_contacts,
        am_cbd_contacts=am_cbd_contacts,
        max_overlap_um=max_overlap,
        max_hertz_pressure_gpa=max_pressure,
        am_contacts_over_threshold=am_contacts_over_threshold,
        am_contact_over_threshold_fraction=am_contact_fraction,
        am_particles_at_risk=len(am_particles_at_risk),
        am_particle_risk_fraction=am_particle_fraction,
    )


def compute_bond_tracking_metrics(snapshot: Snapshot) -> BondTrackingMetrics:
    available = [particle for particle in snapshot.particles if particle.nbond is not None]
    if not available:
        return BondTrackingMetrics(
            bonded_particles=0,
            bonded_particle_fraction=0.0,
            mean_nbond=0.0,
            mean_am_nbond=0.0,
            am_zero_bond_particles=0,
            am_zero_bond_fraction=0.0,
        )

    bonded_particles = sum(1 for particle in available if (particle.nbond or 0) > 0)
    am_particles = [particle for particle in available if particle.type_id == 1]
    am_zero_bond_particles = sum(1 for particle in am_particles if (particle.nbond or 0) == 0)
    mean_nbond = sum(float(particle.nbond or 0) for particle in available) / len(available)
    mean_am_nbond = (
        sum(float(particle.nbond or 0) for particle in am_particles) / len(am_particles)
        if am_particles
        else 0.0
    )
    return BondTrackingMetrics(
        bonded_particles=bonded_particles,
        bonded_particle_fraction=bonded_particles / len(available),
        mean_nbond=mean_nbond,
        mean_am_nbond=mean_am_nbond,
        am_zero_bond_particles=am_zero_bond_particles,
        am_zero_bond_fraction=am_zero_bond_particles / len(am_particles) if am_particles else 0.0,
    )


def compute_am_network_damage_metrics(
    initial_snapshot: Snapshot,
    snapshot: Snapshot,
    bond_loss_damage_threshold_fraction: float,
) -> AmNetworkDamageMetrics:
    initial_by_pid = {
        particle.pid: particle
        for particle in initial_snapshot.particles
        if particle.type_id == 1 and particle.nbond is not None and (particle.nbond or 0) > 0
    }
    current_by_pid = {particle.pid: particle for particle in snapshot.particles if particle.type_id == 1}

    if not initial_by_pid:
        return AmNetworkDamageMetrics(
            am_particles_with_initial_links=0,
            damaged_am_particles=0,
            damaged_am_particle_fraction=0.0,
            mean_am_bond_loss_fraction=0.0,
            max_am_bond_loss_fraction=0.0,
            top_damaged_am_fraction=0.0,
            middle_damaged_am_fraction=0.0,
            bottom_damaged_am_fraction=0.0,
        )

    z_min, z_max, height = compute_height(snapshot.particles)
    if height <= 0.0:
        height = 1.0

    damaged = 0
    loss_fractions: list[float] = []
    region_totals = {"top": 0, "middle": 0, "bottom": 0}
    region_damaged = {"top": 0, "middle": 0, "bottom": 0}

    for pid, initial_particle in initial_by_pid.items():
        current_particle = current_by_pid.get(pid)
        current_nbond = 0 if current_particle is None or current_particle.nbond is None else max(current_particle.nbond, 0)
        initial_nbond = max(initial_particle.nbond or 0, 0)
        if initial_nbond <= 0:
            continue

        loss_fraction = max((initial_nbond - current_nbond) / initial_nbond, 0.0)
        loss_fractions.append(loss_fraction)
        is_damaged = loss_fraction >= bond_loss_damage_threshold_fraction
        if is_damaged:
            damaged += 1

        z_position = initial_particle.z_um if current_particle is None else current_particle.z_um
        normalized_depth = (z_position - z_min) / height
        if normalized_depth < (1.0 / 3.0):
            region = "bottom"
        elif normalized_depth < (2.0 / 3.0):
            region = "middle"
        else:
            region = "top"
        region_totals[region] += 1
        if is_damaged:
            region_damaged[region] += 1

    count = len(loss_fractions)
    return AmNetworkDamageMetrics(
        am_particles_with_initial_links=count,
        damaged_am_particles=damaged,
        damaged_am_particle_fraction=(damaged / count) if count > 0 else 0.0,
        mean_am_bond_loss_fraction=(sum(loss_fractions) / count) if count > 0 else 0.0,
        max_am_bond_loss_fraction=max(loss_fractions, default=0.0),
        top_damaged_am_fraction=(
            region_damaged["top"] / region_totals["top"] if region_totals["top"] > 0 else 0.0
        ),
        middle_damaged_am_fraction=(
            region_damaged["middle"] / region_totals["middle"] if region_totals["middle"] > 0 else 0.0
        ),
        bottom_damaged_am_fraction=(
            region_damaged["bottom"] / region_totals["bottom"] if region_totals["bottom"] > 0 else 0.0
        ),
    )


def build_state_metrics(
    state: str,
    source_file: Path,
    initial_snapshot: Snapshot,
    snapshot: Snapshot,
    total_mass_pg: float,
    solid_density_g_cm3: float,
    initial_height_um: float,
    areal_loading_mg_cm2: float,
    am_mass_fraction: float,
    q_usable_mAh_g: float,
    average_voltage_V: float,
    fracture_threshold_gpa: float,
    bond_loss_damage_threshold_fraction: float,
) -> StateMetrics:
    z_min, z_max, height = compute_height(snapshot.particles)
    area_um2 = snapshot.x_length_um * snapshot.y_length_um
    bulk_density = total_mass_pg / (area_um2 * height)
    coating_density_from_loading_thickness = (
        10.0 * areal_loading_mg_cm2 / height if height > 0.0 else 0.0
    )
    porosity = 1.0 - (bulk_density / solid_density_g_cm3)
    gravimetric_energy = am_mass_fraction * q_usable_mAh_g * average_voltage_V
    volumetric_energy = gravimetric_energy * bulk_density
    volumetric_capacity_loading_based = (
        10.0 * areal_loading_mg_cm2 * am_mass_fraction * q_usable_mAh_g / height
        if height > 0.0
        else 0.0
    )
    volumetric_energy_loading_based = volumetric_capacity_loading_based * average_voltage_V
    reduction_pct = (initial_height_um - height) / initial_height_um * 100.0
    fracture = compute_fracture_metrics(snapshot, fracture_threshold_gpa)
    bond_tracking = compute_bond_tracking_metrics(snapshot)
    network_damage = compute_am_network_damage_metrics(
        initial_snapshot=initial_snapshot,
        snapshot=snapshot,
        bond_loss_damage_threshold_fraction=bond_loss_damage_threshold_fraction,
    )

    return StateMetrics(
        state=state,
        source_file=source_file.name,
        particle_count=len(snapshot.particles),
        z_min_um=z_min,
        z_max_um=z_max,
        bed_height_um=height,
        coating_thickness_um=height,
        bulk_density_g_cm3=bulk_density,
        coating_density_from_loading_thickness_g_cm3=coating_density_from_loading_thickness,
        porosity=porosity,
        reduction_vs_initial_pct=reduction_pct,
        gravimetric_energy_Wh_kg=gravimetric_energy,
        volumetric_energy_Wh_L=volumetric_energy,
        volumetric_capacity_loading_based_mAh_cm3=volumetric_capacity_loading_based,
        volumetric_energy_loading_based_Wh_L=volumetric_energy_loading_based,
        total_contacts=fracture.total_contacts,
        am_involving_contacts=fracture.am_involving_contacts,
        am_am_contacts=fracture.am_am_contacts,
        am_cbd_contacts=fracture.am_cbd_contacts,
        max_overlap_um=fracture.max_overlap_um,
        max_hertz_pressure_gpa=fracture.max_hertz_pressure_gpa,
        am_contacts_over_threshold=fracture.am_contacts_over_threshold,
        am_contact_over_threshold_fraction=fracture.am_contact_over_threshold_fraction,
        am_particles_at_risk=fracture.am_particles_at_risk,
        am_particle_risk_fraction=fracture.am_particle_risk_fraction,
        bonded_particles=bond_tracking.bonded_particles,
        bonded_particle_fraction=bond_tracking.bonded_particle_fraction,
        mean_nbond=bond_tracking.mean_nbond,
        mean_am_nbond=bond_tracking.mean_am_nbond,
        am_zero_bond_particles=bond_tracking.am_zero_bond_particles,
        am_zero_bond_fraction=bond_tracking.am_zero_bond_fraction,
        am_particles_with_initial_links=network_damage.am_particles_with_initial_links,
        damaged_am_particles=network_damage.damaged_am_particles,
        damaged_am_particle_fraction=network_damage.damaged_am_particle_fraction,
        mean_am_bond_loss_fraction=network_damage.mean_am_bond_loss_fraction,
        max_am_bond_loss_fraction=network_damage.max_am_bond_loss_fraction,
        top_damaged_am_fraction=network_damage.top_damaged_am_fraction,
        middle_damaged_am_fraction=network_damage.middle_damaged_am_fraction,
        bottom_damaged_am_fraction=network_damage.bottom_damaged_am_fraction,
    )


def write_csv(path: Path, rows: list[StateMetrics]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, case_summary: CaseSummary, rows: list[StateMetrics]) -> None:
    validation = {
        "no_lost_atoms": True,
        "pressure_nonnegative": all(row.max_hertz_pressure_gpa >= 0.0 for row in rows),
        "porosity_monotonic_reasonable": all(0.0 < row.porosity < 1.0 for row in rows),
        "active_bonds_nonnegative": case_summary.max_active_bonds >= 0.0
        and case_summary.final_active_bonds >= 0.0,
    }
    validation["pass"] = all(validation.values())
    payload: dict = {
        "case_summary": asdict(case_summary),
        "states": [asdict(row) for row in rows],
        "validation_gates": validation,
    }
    if HAS_BOND_CONNECTIVITY:
        try:
            payload["bond_breakage_connectivity"] = analyze_bonds_ab(path.parent)
        except Exception:
            pass
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare non-calendered, compressed, and final calendered states for one case."
    )
    parser.add_argument(
        "--case-dir",
        required=True,
        help="Path to one simulation case folder, e.g. simulations/lfp_r25_fast",
    )
    parser.add_argument(
        "--q-usable",
        type=float,
        default=150.0,
        help="Usable LFP specific capacity in mAh/g-LFP for the energy proxy.",
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=3.4,
        help="Average LFP discharge voltage in V for the energy proxy.",
    )
    parser.add_argument(
        "--fracture-threshold-gpa",
        "--contact-risk-threshold-gpa",
        type=float,
        default=0.7,
        help=(
            "Critical Hertzian contact pressure threshold for contact-risk proxy. "
            "LFP fracture toughness ~0.7 GPa (vs NMC 1.5 GPa). "
            "This is a coarse contact damage-risk metric, not explicit crack propagation."
        ),
    )
    parser.add_argument(
        "--bond-loss-damage-threshold-fraction",
        type=float,
        default=0.10,
        help=(
            "AM particle is flagged as support-network-damaged if its bond loss fraction exceeds this threshold. "
            "Default 0.10 follows the Xu et al. practical interpretation, but for the current coarse-grained "
            "model this should be interpreted as AM support-network damage rather than intraparticle cracking."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir).resolve()

    required_paths = {
        "data": case_dir / "data.electrode",
        "bonded_initial": case_dir / "snapshot.bonded_initial.lammpstrj",
        "compressed": case_dir / "snapshot.compressed.lammpstrj",
        "final": case_dir / "final.calendered.lammpstrj",
        "config": case_dir / "case_config.json",
        "metadata": case_dir / "structure_metadata.json",
    }
    required_for_run = {
        key: path
        for key, path in required_paths.items()
        if key != "bonded_initial"
    }
    missing = [str(path) for path in required_for_run.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    case_config = load_json(required_paths["config"])
    structure_metadata = load_json(required_paths["metadata"])

    initial_snapshot = (
        parse_single_frame_dump(required_paths["bonded_initial"])
        if required_paths["bonded_initial"].exists()
        else parse_lammps_data_sphere(required_paths["data"])
    )
    compressed_snapshot = parse_single_frame_dump(required_paths["compressed"])
    final_snapshot = parse_single_frame_dump(required_paths["final"])

    initial_height_um = compute_height(initial_snapshot.particles)[2]
    solid_density = float(structure_metadata["solid_density_g_cm3"])
    am_mass_fraction = float(structure_metadata["am_mass_fraction"])
    total_mass_pg = float(structure_metadata["total_mass_pg"])
    areal_loading_mg_cm2 = float(structure_metadata["areal_loading_mg_cm2"])

    states = [
        build_state_metrics(
            state="non_calendered",
            source_file=(
                required_paths["bonded_initial"]
                if required_paths["bonded_initial"].exists()
                else required_paths["data"]
            ),
            initial_snapshot=initial_snapshot,
            snapshot=initial_snapshot,
            total_mass_pg=total_mass_pg,
            solid_density_g_cm3=solid_density,
            initial_height_um=initial_height_um,
            areal_loading_mg_cm2=areal_loading_mg_cm2,
            am_mass_fraction=am_mass_fraction,
            q_usable_mAh_g=args.q_usable,
            average_voltage_V=args.voltage,
            fracture_threshold_gpa=args.fracture_threshold_gpa,
            bond_loss_damage_threshold_fraction=args.bond_loss_damage_threshold_fraction,
        ),
        build_state_metrics(
            state="max_compression",
            source_file=required_paths["compressed"],
            initial_snapshot=initial_snapshot,
            snapshot=compressed_snapshot,
            total_mass_pg=total_mass_pg,
            solid_density_g_cm3=solid_density,
            initial_height_um=initial_height_um,
            areal_loading_mg_cm2=areal_loading_mg_cm2,
            am_mass_fraction=am_mass_fraction,
            q_usable_mAh_g=args.q_usable,
            average_voltage_V=args.voltage,
            fracture_threshold_gpa=args.fracture_threshold_gpa,
            bond_loss_damage_threshold_fraction=args.bond_loss_damage_threshold_fraction,
        ),
        build_state_metrics(
            state="final_recovered",
            source_file=required_paths["final"],
            initial_snapshot=initial_snapshot,
            snapshot=final_snapshot,
            total_mass_pg=total_mass_pg,
            solid_density_g_cm3=solid_density,
            initial_height_um=initial_height_um,
            areal_loading_mg_cm2=areal_loading_mg_cm2,
            am_mass_fraction=am_mass_fraction,
            q_usable_mAh_g=args.q_usable,
            average_voltage_V=args.voltage,
            fracture_threshold_gpa=args.fracture_threshold_gpa,
            bond_loss_damage_threshold_fraction=args.bond_loss_damage_threshold_fraction,
        ),
    ]

    pressure_rows = parse_pressure_log(case_dir / "pressure_log.txt")
    peak_pressure = max((row[3] for row in pressure_rows), default=0.0)
    final_pressure = pressure_rows[-1][3] if pressure_rows else 0.0
    max_active_bonds = max((row[4] for row in pressure_rows), default=0.0)
    final_active_bonds = pressure_rows[-1][4] if pressure_rows else 0.0
    broken_bond_events = parse_broken_bond_dump(case_dir / "dump.broken_bonds.local")
    broken_bond_fraction = (
        broken_bond_events / max_active_bonds if max_active_bonds > 0.0 else 0.0
    )

    h_min = states[1].bed_height_um
    h_final = states[2].bed_height_um
    h0 = states[0].bed_height_um
    springback_um = h_final - h_min
    springback_relative_to_min = springback_um / h_min if h_min > 0.0 else 0.0
    recovery_span = h0 - h_min
    springback_recovery_fraction = springback_um / recovery_span if recovery_span > 0.0 else 0.0

    case_summary = CaseSummary(
        case_label=str(case_config.get("label", case_dir.name)),
        q_usable_mAh_g=args.q_usable,
        average_voltage_V=args.voltage,
        fracture_threshold_gpa=args.fracture_threshold_gpa,
        initial_coating_thickness_um=states[0].coating_thickness_um,
        min_coating_thickness_um=states[1].coating_thickness_um,
        final_coating_thickness_um=states[2].coating_thickness_um,
        peak_pressure=peak_pressure,
        final_pressure=final_pressure,
        springback_um=springback_um,
        springback_relative_to_min=springback_relative_to_min,
        springback_recovery_fraction=springback_recovery_fraction,
        max_active_bonds=max_active_bonds,
        final_active_bonds=final_active_bonds,
        broken_bond_events=broken_bond_events,
        broken_bond_fraction_vs_max_active=broken_bond_fraction,
        bond_loss_damage_threshold_fraction=args.bond_loss_damage_threshold_fraction,
        max_damaged_am_particle_fraction=max(
            (state.damaged_am_particle_fraction for state in states),
            default=0.0,
        ),
        final_damaged_am_particle_fraction=states[-1].damaged_am_particle_fraction,
        damage_interpretation=(
            "coarse_grained_am_support_network_damage"
            if not case_config.get("bond_topology", {}).get("am_am", False)
            else "coarse_grained_mixed_network_damage"
        ),
    )
    # Lost-atoms detection from the simulation log is a hard fail gate.
    log_path = case_dir / "log.lammps"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        if "ERROR: Lost atoms" in text:
            case_summary = CaseSummary(
                **{
                    **asdict(case_summary),
                    "damage_interpretation": case_summary.damage_interpretation
                    + ";lost_atoms_detected",
                }
            )

    csv_path = case_dir / "state_comparison.csv"
    json_path = case_dir / "state_comparison.json"
    write_csv(csv_path, states)
    write_json(json_path, case_summary, states)

    print(f"Wrote state comparison CSV to {csv_path}")
    print(f"Wrote state comparison JSON to {json_path}")
    print("")
    for state in states:
        print(
            f"{state.state}: "
            f"coating_thickness={state.coating_thickness_um:.3f} um, "
            f"coating_density={state.coating_density_from_loading_thickness_g_cm3:.3f} g/cm^3, "
            f"porosity={state.porosity:.3f}, "
            f"mAh/cm^3={state.volumetric_capacity_loading_based_mAh_cm3:.1f}, "
            f"Wh/L={state.volumetric_energy_loading_based_Wh_L:.1f}, "
            f"max_contact_risk_p0={state.max_hertz_pressure_gpa:.3f} GPa, "
            f"am_risk_fraction={state.am_particle_risk_fraction:.3f}, "
            f"support_damage_fraction={state.damaged_am_particle_fraction:.3f}, "
            f"mean_am_nbond={state.mean_am_nbond:.2f}, "
            f"am_zero_bond_fraction={state.am_zero_bond_fraction:.3f}"
        )
    print("")
    print(
        "springback: "
        f"{case_summary.springback_um:.3f} um, "
        f"relative_to_min={case_summary.springback_relative_to_min:.3f}, "
        f"recovery_fraction={case_summary.springback_recovery_fraction:.3f}"
    )
    print(
        "support-network damage: "
        f"max_active_bonds={case_summary.max_active_bonds:.1f}, "
        f"final_active_bonds={case_summary.final_active_bonds:.1f}, "
        f"broken_bond_events={case_summary.broken_bond_events}, "
        f"broken_fraction={case_summary.broken_bond_fraction_vs_max_active:.3f}"
    )
    print(
        "am support-network damage criterion: "
        f"bond_loss_threshold={case_summary.bond_loss_damage_threshold_fraction:.3f}, "
        f"max_damaged_am_fraction={case_summary.max_damaged_am_particle_fraction:.3f}, "
        f"final_damaged_am_fraction={case_summary.final_damaged_am_particle_fraction:.3f}"
    )
    print(
        "note: Hertz-pressure output is a contact-risk proxy. "
        "For the current coarse-grained model, bond-loss output should be interpreted as "
        f"{case_summary.damage_interpretation.replace('_', ' ')} rather than true intraparticle cracking."
    )


if __name__ == "__main__":
    main()
