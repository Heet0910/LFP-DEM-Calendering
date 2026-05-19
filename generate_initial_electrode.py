from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from geometry_materials_config import BondParams, DomainConfig, ParticleTypes
except ModuleNotFoundError:
    from input.geometry_materials_config import BondParams, DomainConfig, ParticleTypes


@dataclass
class Particle:
    pid: int
    type_id: int
    radius: float
    density: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class StructureMetadata:
    """3-phase dry electrode structure metadata.

    Particle types:
      1 = LFP Active Material (AM)
      2 = Carbon Black (CB)  — was: CBD type 2 in 2-phase model
      3 = PTFE Binder        — new in 3-phase dry electrode model

    Legacy backward-compat aliases:
      cbd_count        = cb_count + ptfe_count  (total non-AM particles)
      cbd_mass_fraction = cb_mass_fraction + ptfe_mass_fraction
    """
    am_count: int
    cb_count: int
    ptfe_count: int
    total_mass_pg: float
    areal_loading_mg_cm2: float
    bed_height_um: float
    bulk_density_g_cm3: float
    porosity: float
    solid_density_g_cm3: float
    am_mass_fraction: float
    cb_mass_fraction: float
    ptfe_mass_fraction: float

    # ── Legacy aliases ────────────────────────────────────────────────────────
    @property
    def cbd_count(self) -> int:
        return self.cb_count + self.ptfe_count

    @property
    def cbd_mass_fraction(self) -> float:
        return self.cb_mass_fraction + self.ptfe_mass_fraction


def sphere_volume(radius_um: float) -> float:
    return (4.0 / 3.0) * math.pi * radius_um**3


def areal_loading_to_total_mass_pg(areal_loading_mg_cm2: float) -> float:
    return areal_loading_mg_cm2 * 10.0 * DomainConfig.area_um2()


def build_mass_targets(areal_loading_mg_cm2: float) -> tuple[int, int, int]:
    """Return target particle counts (n_am, n_cb, n_ptfe) for the 3-phase system."""
    total_mass_pg = areal_loading_to_total_mass_pg(areal_loading_mg_cm2)

    am_mass_target   = ParticleTypes.am_wt_frac   * total_mass_pg
    cb_mass_target   = ParticleTypes.cb_wt_frac   * total_mass_pg
    ptfe_mass_target = ParticleTypes.ptfe_wt_frac * total_mass_pg

    mean_am_volume = sum(
        weight * sphere_volume(radius)
        for radius, weight in zip(
            ParticleTypes.am_radii_um,
            ParticleTypes.am_radius_probabilities,
        )
    )
    mean_am_mass   = mean_am_volume * ParticleTypes.am_density_g_cm3
    cb_mass        = sphere_volume(ParticleTypes.cb_radius_um)   * ParticleTypes.cb_density_g_cm3
    ptfe_mass      = sphere_volume(ParticleTypes.ptfe_radius_um) * ParticleTypes.ptfe_density_g_cm3

    n_am   = max(int(round(am_mass_target   / mean_am_mass)), 1)
    n_cb   = max(int(round(cb_mass_target   / cb_mass)),   1)
    n_ptfe = max(int(round(ptfe_mass_target / ptfe_mass)), 1)
    return n_am, n_cb, n_ptfe


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1.0e-9:
        values.append(round(current, 8))
        current += step
    return values


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def minimum_image(dx: float, box_length: float) -> float:
    return dx - box_length * round(dx / box_length)


def overlaps(
    x: float,
    y: float,
    z: float,
    radius: float,
    particles: list[Particle],
    overlap_scale: float,
) -> bool:
    for particle in particles:
        dx = minimum_image(x - particle.x, DomainConfig.lx_um)
        dy = minimum_image(y - particle.y, DomainConfig.ly_um)
        dz = z - particle.z
        cutoff = overlap_scale * (radius + particle.radius)
        if dx * dx + dy * dy + dz * dz < cutoff * cutoff:
            return True
    return False


def build_am_sites() -> list[tuple[float, float, float]]:
    max_r = max(ParticleTypes.am_radii_um)
    spacing_xy = 2.60
    spacing_z  = 2.45
    y_step = spacing_xy * math.sqrt(3.0) / 2.0

    sites: list[tuple[float, float, float]] = []
    z_values = frange(max_r, DomainConfig.initial_bed_height_um - max_r, spacing_z)
    for layer_index, z in enumerate(z_values):
        plane_shift = 0.5 * spacing_xy if layer_index % 2 else 0.0
        y_values = frange(max_r, DomainConfig.ly_um - max_r, y_step)
        for row_index, y in enumerate(y_values):
            row_shift = plane_shift + (0.5 * spacing_xy if row_index % 2 else 0.0)
            x_values = frange(max_r + row_shift, DomainConfig.lx_um - max_r, spacing_xy)
            for x in x_values:
                sites.append((x, y, z))
    return sites


def build_small_particle_sites(
    radius: float,
    spacing_xy: float,
    spacing_z: float,
) -> list[tuple[float, float, float]]:
    """Generic HCP site generator for small particles (CB or PTFE)."""
    y_step = spacing_xy * math.sqrt(3.0) / 2.0
    sites: list[tuple[float, float, float]] = []
    z_values = frange(radius, DomainConfig.initial_bed_height_um - radius, spacing_z)
    for layer_index, z in enumerate(z_values):
        plane_shift = 0.5 * spacing_xy if layer_index % 2 else 0.0
        y_values = frange(radius, DomainConfig.ly_um - radius, y_step)
        for row_index, y in enumerate(y_values):
            row_shift = plane_shift + (0.5 * spacing_xy if row_index % 2 else 0.0)
            x_values = frange(radius + row_shift, DomainConfig.lx_um - radius, spacing_xy)
            for x in x_values:
                sites.append((x, y, z))
    return sites


def build_cb_sites() -> list[tuple[float, float, float]]:
    """HCP site grid for Carbon Black aggregates (r ≈ 0.45 µm)."""
    r = ParticleTypes.cb_radius_um
    # Clearance between adjacent CB particles ≈ 0.15 µm
    return build_small_particle_sites(r, spacing_xy=2*r + 0.15, spacing_z=2*r + 0.10)


def build_ptfe_sites() -> list[tuple[float, float, float]]:
    """HCP site grid for PTFE particles (r ≈ 0.40 µm).

    Placed AFTER CB, so overlap checker handles cross-type conflicts.
    Slightly finer grid than CB to fill remaining interstitial space.
    """
    r = ParticleTypes.ptfe_radius_um
    return build_small_particle_sites(r, spacing_xy=2*r + 0.12, spacing_z=2*r + 0.08)


def place_particles_on_sites(
    type_id: int,
    density: float,
    radii: list[float],
    candidate_sites: list[tuple[float, float, float]],
    particles: list[Particle],
    rng: random.Random,
    jitter_um: float,
    overlap_scale: float,
) -> list[Particle]:
    available = candidate_sites.copy()
    rng.shuffle(available)
    next_pid = len(particles) + 1
    placed: list[Particle] = []

    for radius in sorted(radii, reverse=True):
        found_index = None
        for index, (x0, y0, z0) in enumerate(available):
            x = clamp(x0 + rng.uniform(-jitter_um, jitter_um), radius, DomainConfig.lx_um - radius)
            y = clamp(y0 + rng.uniform(-jitter_um, jitter_um), radius, DomainConfig.ly_um - radius)
            z = clamp(
                z0 + rng.uniform(-jitter_um, jitter_um),
                radius,
                DomainConfig.initial_bed_height_um - radius,
            )
            combined_particles = particles + placed
            if not overlaps(x, y, z, radius, combined_particles, overlap_scale):
                placed.append(
                    Particle(
                        pid=next_pid,
                        type_id=type_id,
                        radius=radius,
                        density=density,
                        x=x,
                        y=y,
                        z=z,
                    )
                )
                next_pid += 1
                found_index = index
                break
        if found_index is None:
            raise RuntimeError(
                f"Could not place all particles of type {type_id} (r={radius:.3f} µm). "
                "Consider enlarging the domain or reducing the target density."
            )
        del available[found_index]

    return placed


def build_particles(rng: random.Random, areal_loading_mg_cm2: float) -> list[Particle]:
    """Place AM (type 1), Carbon Black (type 2), PTFE (type 3) particles.

    Uses a shrink-factor retry loop to reduce particle count if the domain
    cannot accommodate all particles at the target areal loading.
    """
    target_n_am, target_n_cb, target_n_ptfe = build_mass_targets(areal_loading_mg_cm2)
    last_error: RuntimeError | None = None

    for shrink_factor in (1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79):
        n_am   = max(int(round(target_n_am   * shrink_factor)), 1)
        n_cb   = max(int(round(target_n_cb   * shrink_factor)), 1)
        n_ptfe = max(int(round(target_n_ptfe * shrink_factor)), 1)

        am_radii   = list(rng.choices(
            list(ParticleTypes.am_radii_um),
            weights=list(ParticleTypes.am_radius_probabilities),
            k=n_am,
        ))
        cb_radii   = [ParticleTypes.cb_radius_um]   * n_cb
        ptfe_radii = [ParticleTypes.ptfe_radius_um] * n_ptfe

        particles: list[Particle] = []
        try:
            # ── Type 1: LFP Active Material ───────────────────────────────────
            particles.extend(place_particles_on_sites(
                type_id=1,
                density=ParticleTypes.am_density_g_cm3,
                radii=am_radii,
                candidate_sites=build_am_sites(),
                particles=particles,
                rng=rng,
                jitter_um=0.12,
                overlap_scale=0.88,
            ))
            # ── Type 2: Carbon Black (Super-P aggregates) ─────────────────────
            particles.extend(place_particles_on_sites(
                type_id=2,
                density=ParticleTypes.cb_density_g_cm3,
                radii=cb_radii,
                candidate_sites=build_cb_sites(),
                particles=particles,
                rng=rng,
                jitter_um=0.08,
                overlap_scale=0.96,
            ))
            # ── Type 3: PTFE Binder (replaces PVDF, placed last) ─────────────
            particles.extend(place_particles_on_sites(
                type_id=3,
                density=ParticleTypes.ptfe_density_g_cm3,
                radii=ptfe_radii,
                candidate_sites=build_ptfe_sites(),
                particles=particles,
                rng=rng,
                jitter_um=0.06,
                overlap_scale=0.98,
            ))
            return particles
        except RuntimeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not build an initial particle structure.")


def compute_metadata(particles: list[Particle]) -> StructureMetadata:
    total_mass = sum(p.density * sphere_volume(p.radius) for p in particles)
    z_min = min(p.z - p.radius for p in particles)
    z_max = max(p.z + p.radius for p in particles)
    bed_height   = z_max - z_min
    bulk_density = total_mass / (DomainConfig.area_um2() * bed_height)
    porosity     = 1.0 - (bulk_density / ParticleTypes.solid_density_g_cm3())
    areal_loading_mg_cm2 = total_mass / DomainConfig.area_um2() / 10.0

    am_mass   = sum(p.density * sphere_volume(p.radius) for p in particles if p.type_id == 1)
    cb_mass   = sum(p.density * sphere_volume(p.radius) for p in particles if p.type_id == 2)
    ptfe_mass = sum(p.density * sphere_volume(p.radius) for p in particles if p.type_id == 3)

    return StructureMetadata(
        am_count=sum(1 for p in particles if p.type_id == 1),
        cb_count=sum(1 for p in particles if p.type_id == 2),
        ptfe_count=sum(1 for p in particles if p.type_id == 3),
        total_mass_pg=total_mass,
        areal_loading_mg_cm2=areal_loading_mg_cm2,
        bed_height_um=bed_height,
        bulk_density_g_cm3=bulk_density,
        porosity=porosity,
        solid_density_g_cm3=ParticleTypes.solid_density_g_cm3(),
        am_mass_fraction=am_mass / total_mass if total_mass else 0.0,
        cb_mass_fraction=cb_mass / total_mass if total_mass else 0.0,
        ptfe_mass_fraction=ptfe_mass / total_mass if total_mass else 0.0,
    )


def densify_along_z(
    particles: list[Particle],
    target_bulk_density_g_cm3: float,
) -> list[Particle]:
    metadata = compute_metadata(particles)
    if metadata.bulk_density_g_cm3 >= target_bulk_density_g_cm3:
        return particles

    z_bottom = min(p.z - p.radius for p in particles)
    scale = metadata.bulk_density_g_cm3 / target_bulk_density_g_cm3

    densified: list[Particle] = []
    for particle in particles:
        lower = particle.z - particle.radius
        new_lower = z_bottom + scale * (lower - z_bottom)
        densified.append(
            Particle(
                pid=particle.pid,
                type_id=particle.type_id,
                radius=particle.radius,
                density=particle.density,
                x=particle.x,
                y=particle.y,
                z=new_lower + particle.radius,
            )
        )
    return densified


def write_lammps_sphere_data(particles: list[Particle], path: Path) -> None:
    """Write LAMMPS bpm/sphere data file for 3-phase dry electrode (3 atom types, 4 bond types)."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("LAMMPS data file via generate_initial_electrode.py (3-phase dry electrode)\n\n")
        handle.write(f"{len(particles)} atoms\n")
        handle.write("0 bonds\n")
        handle.write("3 atom types\n")   # 1=AM, 2=CB, 3=PTFE
        handle.write("4 bond types\n\n") # 1=AM-CB, 2=AM-PTFE, 3=CB-PTFE, 4=CB-CB
        handle.write(f"0.0 {DomainConfig.lx_um} xlo xhi\n")
        handle.write(f"0.0 {DomainConfig.ly_um} ylo yhi\n")
        handle.write(f"0.0 {DomainConfig.lz_um} zlo zhi\n\n")
        handle.write("Atoms # bpm/sphere\n\n")
        for particle in particles:
            diameter = 2.0 * particle.radius
            handle.write(
                f"{particle.pid} 1 {particle.type_id} {diameter:.6f} {particle.density:.6f} "
                f"{particle.x:.6f} {particle.y:.6f} {particle.z:.6f}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 3-phase LFP/CB/PTFE dry electrode structure for DEM calendering."
    )
    parser.add_argument(
        "--areal-loading-mg-cm2",
        type=float,
        default=ParticleTypes.target_areal_loading_mg_cm2,
        help="Target coating areal loading in mg/cm^2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(7)
    particles = build_particles(rng, args.areal_loading_mg_cm2)
    particles = densify_along_z(particles, ParticleTypes.initial_bulk_density_g_cm3)
    metadata  = compute_metadata(particles)

    out_dir        = Path(__file__).resolve().parent
    data_path      = out_dir / "data.electrode"
    metadata_path  = out_dir / "structure_metadata.json"

    # StructureMetadata has properties (cbd_count, cbd_mass_fraction) which are
    # not fields → exclude from asdict by converting manually
    meta_dict = {
        "am_count":            metadata.am_count,
        "cb_count":            metadata.cb_count,
        "ptfe_count":          metadata.ptfe_count,
        "cbd_count":           metadata.cbd_count,         # legacy
        "total_mass_pg":       metadata.total_mass_pg,
        "areal_loading_mg_cm2": metadata.areal_loading_mg_cm2,
        "bed_height_um":       metadata.bed_height_um,
        "bulk_density_g_cm3":  metadata.bulk_density_g_cm3,
        "porosity":            metadata.porosity,
        "solid_density_g_cm3": metadata.solid_density_g_cm3,
        "am_mass_fraction":    metadata.am_mass_fraction,
        "cb_mass_fraction":    metadata.cb_mass_fraction,
        "ptfe_mass_fraction":  metadata.ptfe_mass_fraction,
        "cbd_mass_fraction":   metadata.cbd_mass_fraction, # legacy
    }

    write_lammps_sphere_data(particles, data_path)
    metadata_path.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")

    print(f"Wrote {len(particles)} particles to {data_path}")
    print(
        f"AM={metadata.am_count}, CB={metadata.cb_count}, PTFE={metadata.ptfe_count}, "
        f"loading={metadata.areal_loading_mg_cm2:.3f} mg/cm^2, "
        f"bulk_density={metadata.bulk_density_g_cm3:.3f} g/cm^3, "
        f"porosity={metadata.porosity:.3f}, bed_height={metadata.bed_height_um:.3f} um"
    )
    print(
        f"Mass fractions: AM={metadata.am_mass_fraction:.3f}, "
        f"CB={metadata.cb_mass_fraction:.3f}, PTFE={metadata.ptfe_mass_fraction:.3f} "
        f"(target 0.94:0.03:0.03)"
    )


if __name__ == "__main__":
    main()
