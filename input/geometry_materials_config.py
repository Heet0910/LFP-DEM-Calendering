from __future__ import annotations

"""
Literature-anchored geometry and material defaults for LFP dry electrode DEM model.
3-phase system: Active Material (LFP) + Carbon Black (CB) + PTFE Binder

Key literature sources
----------------------
[1] Ngandjong et al. (2021), J. Power Sources 485, 229320.
    doi:10.1016/j.jpowsour.2020.229320
[2] Schreiner et al. (2021), Procedia CIRP 104, 91-97.
    doi:10.1016/j.procir.2021.11.016
[3] Ge et al. (2022), Powder Technology 403, 117366.
    doi:10.1016/j.powtec.2022.117366
[4] Sangrós Giménez et al. (2020), Energy Technology 8, 1900180.
    doi:10.1002/ente.201900180
[5] Daikin / Chemours PTFE datasheets — density 2.14-2.20 g/cm³, E ~0.5 GPa
[6] Sigma-Aldrich Super-P carbon black — density 1.8-2.2 g/cm³, BET 60-65 m²/g
[7] Dry LFP electrode 94:3:3 (LFP:CB:PTFE) — RSC Energy & Env. Sci. 2023

Unit system (LAMMPS "units micro")
-----------------------------------
  length   : µm          (1 µm  = 1e-6 m)
  mass     : pg          (1 pg  = 1e-15 kg)
  time     : µs          (1 µs  = 1e-6 s)
  pressure : pg/(µm·µs²) ≈ 1 kPa   (1 sim-unit ≈ 1 kPa)
  velocity : µm/µs = m/s

Calibration notes
-----------------
The SJKR cohesion energy density (CED) is the most important parameter to
calibrate against an experimental compaction curve.  The default value of 80
(~80 kPa) is the mid-range from Schreiner et al. [2] Table 2 (NMC cathode,
binder-mediated contacts).  For LFP you may need 50–300 kPa depending on the
PTFE binder content.  For dry electrodes, the fibrillated PTFE network
    requires higher eps_c (fracture strain) than PVDF — PTFE fibrils stretch
    before failing.  Run the 0 % compression case and check that the settled bed
    maintains its height without creep; then scan K against the measured
    pressure–displacement slope at 15 % compression.
"""


class LiteratureAnchors:
    """Directly extracted values from the primary references."""

    # Reduction cases from [1] SI Table S1 and [2] Fig. 3
    reduction_cases: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35)

    # Target post-calendering bulk density window from [1] and [3]
    target_density_window_g_cm3: tuple[float, float] = (2.5, 2.6)

    # Springback range from [1] SI: 4–57 % relative to minimum height
    springback_relative_to_min_window: tuple[float, float] = (0.04, 0.57)

    # DEM time-step used in [1] (scaled to our unit system)
    timestep_us: float = 1.0e-4

    # Roll speed ~0.002 m/s → 0.002 µm/µs in this unit system ([1] §2.2)
    plane_speed_um_per_us: float = 0.002


class ResearchBaselineProfile:
    """
    Conservative defaults for bonded-fracture production runs.
    Velocity is set 1.5× the literature plane speed to keep run times
    manageable within the quasi-static regime verified in [1] SI Fig. S6.

    LFP-specific calibration (Liu 2023, Bockholt 2016):
      kn = 30,000 kPa — matches LFP porous secondary particle E_eff ~ 47 GPa
      gamma_n = 35   — matched viscous damping for kn=3e4
      dt = 0.003 µs  — numerically stable for kn=3e4 (dt_crit ~ 0.005 µs)
    Target: P_peak = 1–15 MPa for 15–35% CR (vs NMC 50–400 MPa).
    """

    compression_ratio: float = 0.20
    compression_velocity_um_per_us: float = -0.003
    decompression_velocity_um_per_us: float = 0.003
    timestep_us: float = 0.001   # required for stability at kn=5e4
    settling_steps: int = 8_000
    hold_steps: int = 2_000
    thermo_every: int = 1_000
    dump_every: int = 2_000


class DomainConfig:
    """
    Simulation box geometry.

    The 18×18 µm footprint is a minimal representative volume that keeps
    single-case wall-clock times reasonable on a desktop workstation.
    For publication-quality size-effect checks, duplicate the initial
    structure in x and y to reach ≥36×36 µm ([1] SI Fig. S3).
    """

    lx_um: float = 18.0
    ly_um: float = 18.0
    initial_bed_height_um: float = 78.0
    headspace_um: float = 22.0
    lz_um: float = initial_bed_height_um + headspace_um

    periodic_x: bool = True
    periodic_y: bool = True
    periodic_z: bool = False

    @classmethod
    def area_um2(cls) -> float:
        return cls.lx_um * cls.ly_um

    @classmethod
    def initial_volume_um3(cls) -> float:
        return cls.area_um2() * cls.initial_bed_height_um


class CarbonBlack:
    """
    Super-P / C65 conductive carbon black — Type 2 particle in 3-phase model.

    Physical primary particle: 30–50 nm (too small for µm-scale DEM).
    DEM coarse-grained radius chosen so that particle count ~320 per case,
    matching the statistical representativeness of the old CBD phase.
    Density = intrinsic carbon material density 1.8–2.2 g/cm³ [6].
    """
    radius_um: float = 0.75          # coarse-grained aggregate sphere [DEM scale]
    density_g_cm3: float = 1.95      # midpoint 1.8–2.2 g/cm³ [6]
    wt_frac: float = 0.03            # 3 wt% in dry LFP 94:3:3 formulation [7]
    young_modulus_gpa: float = 2.0   # effective aggregate modulus (DEM literature)


class PTFEBinder:
    """
    PTFE binder — Type 3 particle in 3-phase model (replaces PVDF).

    Physical fibril diameter: ~200–50 nm (too small for µm-scale DEM).
    DEM coarse-grained radius chosen so that particle count ~350 per case.
    PTFE is 10× softer than PVDF (E ≈ 0.5 GPa vs 3–8 GPa) and fibrillates
    during calendering rather than fracturing brittlely.

    Sources: Daikin PTFE powder TDS; Chemours Teflon PTFE datasheet [5].
    """
    radius_um: float = 0.70          # coarse-grained binder node [DEM scale]
    density_g_cm3: float = 2.17      # mid 2.14–2.20 g/cm³ [5]
    wt_frac: float = 0.03            # 3 wt% in dry LFP 94:3:3 formulation [7]
    young_modulus_gpa: float = 0.5   # bulk PTFE E [5] — much softer than PVDF
    poisson_ratio: float = 0.46      # nearly incompressible


class ParticleTypes:
    """
    3-phase dry electrode: LFP active material (AM) + Carbon Black (CB) + PTFE.

    AM radii are coarse-grained secondary-particle sizes inferred from
    [1] Fig. 1 and [3] Table 1.  CB and PTFE radii are effective aggregate
    spheres. Formulation: 94 wt% LFP : 3 wt% CB : 3 wt% PTFE [7].

    Initial bulk density
    --------------------
    Raised from 1.80 → 2.10 g/cm³ (porosity ~0.40) to better match the
    uncalendered electrode state reported in [1] (ε₀ ≈ 0.38–0.42) and
    [3] tomography baseline.
    """

    # ── Active Material (LFP) — Type 1 ──────────────────────────────────────
    am_radii_um: tuple[float, ...] = (1.2, 1.4, 1.6, 1.8)
    am_radius_probabilities: tuple[float, ...] = (0.2, 0.3, 0.3, 0.2)
    am_density_g_cm3: float = 3.60   # LFP theoretical density [1]
    am_wt_frac: float = 0.94         # 94 wt% in dry 94:3:3 formulation [7]

    # ── Carbon Black — Type 2 (was: CBD) ────────────────────────────────────
    cb_radius_um: float = CarbonBlack.radius_um
    cb_density_g_cm3: float = CarbonBlack.density_g_cm3
    cb_wt_frac: float = CarbonBlack.wt_frac

    # ── PTFE Binder — Type 3 (new, replaces PVDF) ───────────────────────────
    ptfe_radius_um: float = PTFEBinder.radius_um
    ptfe_density_g_cm3: float = PTFEBinder.density_g_cm3
    ptfe_wt_frac: float = PTFEBinder.wt_frac

    # ── Legacy alias (backward compat with existing scripts) ─────────────────
    # Remove these after updating generate_initial_electrode.py
    cbd_radius_um: float = 0.75      # DEPRECATED — use cb_radius_um
    cbd_density_g_cm3: float = 1.95  # DEPRECATED — use cb_density_g_cm3
    cbd_wt_frac: float = 0.06        # DEPRECATED — cb + ptfe combined

    initial_bulk_density_g_cm3: float = 2.10
    target_areal_loading_mg_cm2: float = 11.36

    @classmethod
    def solid_density_g_cm3(cls) -> float:
        """Harmonic-mean solid density for 3-phase dry electrode (94:3:3)."""
        return 1.0 / (
            (cls.am_wt_frac   / cls.am_density_g_cm3)
            + (cls.cb_wt_frac / cls.cb_density_g_cm3)
            + (cls.ptfe_wt_frac / cls.ptfe_density_g_cm3)
        )

    @classmethod
    def initial_porosity(cls) -> float:
        return 1.0 - (cls.initial_bulk_density_g_cm3 / cls.solid_density_g_cm3())

    @classmethod
    def areal_loading_from_bulk_density(cls, bed_height_um: float) -> float:
        return cls.initial_bulk_density_g_cm3 * bed_height_um * 0.1


class ContactParams:
    """
    DEM contact parameters for the Hertz-Mindlin + SJKR hybrid model.

    Hertz-Mindlin (gran/hertz/history)
    ------------------------------------
    kn is the effective normal stiffness coefficient related to the
    plane-strain modulus.  The value 1e5 kPa ≈ 100 MPa is a numerically
    efficient coarse-grained approximation; real LFP has E ≈ 100 GPa
    but using the true value requires a timestep of ~1e-8 µs.

    SJKR cohesion (pair_style sjkr)
    --------------------------------
    The Simplified Johnson-Kendall-Roberts model adds a short-range
    cohesive force proportional to the contact area.  This represents
    the PVDF binder bridges between particles and is essential for
    reproducing the correct compaction stiffness at low pressures and
    the springback magnitude after unloading.

    Literature values for CED [2] Table 2:
      AM–AM  : 8.0 × 10⁴ J/m³  →  80 kPa  ≈  80 sim-units
      AM–CBD : 5.0 × 10⁴ J/m³  →  50 kPa  ≈  50 sim-units
      CBD–CBD: 3.0 × 10⁴ J/m³  →  30 kPa  ≈  30 sim-units

    A single blended value is used here for simplicity; per-type CEDs
    require pair_style hybrid/overlay with multiple sjkr instances,
    which is left for future calibration work.

    Calibration procedure
    ---------------------
    1. Run the 15 % compression case with CED = 0 (pure Hertz).
    2. Increase CED in steps of 10 until the pressure-at-max-displacement
       matches experimental data (typically 5–50 MPa for LFP, [3] Fig. 6).
    3. Check that springback fraction falls in the 4–57 % window from [1].
    """

    # True material constants (for reporting only)
    lfp_youngs_modulus_gpa: float = 100.0     # [1] Table 1
    lfp_poisson_ratio: float = 0.23
    pvdf_youngs_modulus_gpa: float = 1.73
    pvdf_poisson_ratio: float = 0.38

    # kn = 50,000 kPa — LFP intermediate calibration step.
    # Results viewer diagnosed kn=8e3 as too soft:
    #   peak pressure 0.109 MPa (target 0.5–5 MPa)
    #   max particle overlap 0.594 µm (target <0.1 µm)
    # Direction: increase toward 80,000–120,000 (E_LFP = 100–125 GPa range).
    # Start at 50,000 as a stable intermediate; raise to 80k if pressure still low.
    #
    # History:
    #   kn=1e4: P=0.14 MPa, overlap=0.5 µm (too soft)
    #   kn=8e3: P=0.109 MPa, overlap=0.594 µm (even softer, wrong direction)
    #   kn=3e4: P~130 MPa (too stiff, NMC-range)
    #   kn=5e4: target P~5–20 MPa, overlap <0.1 µm ← this pass
    hertz_kn: float = 5.0e4     # LFP intermediate stiffness (kPa)
    hertz_gamma_n: float = 50.0  # physically motivated (Schreiner 2021): was 15, too low
    hertz_friction: float = 0.5  # particle-particle friction [Ngandjong 2021]

    # SJKR/JKR not used (reverted: JKR unstable with bpm/sphere atoms)
    jkr_gamma: float = 0.0   # disabled; use decompression viscous damping instead

    # Viscous damping for each simulation phase
    settling_viscous_damping: float = 25.0
    # Moderate damping during calendering: damps kinetic energy without
    # over-constraining particle rearrangement.
    process_viscous_damping: float = 20.0
    # HIGH damping during decompression: resists elastic springback.
    # Models PVDF binder viscosity on unloading (Ngandjong 2021 SI).
    # 500 = 25x process damping -> target springback 15-40%.
    # Raise to 1000+ if springback still > 50%.
    decompression_viscous_damping: float = 500.0

    gravity_um_per_us2: float = 9.81e-6   # 9.81 m/s² in µm/µs²


class BondParams:
    """
    Breakable bonded-particle model parameters — 3-phase PTFE dry electrode.

    4 bond types for the 3-phase system:
      Bond 1: AM–CB   (LFP active material ↔ carbon black aggregate)
      Bond 2: AM–PTFE (LFP active material ↔ PTFE binder)
      Bond 3: CB–PTFE (carbon black ↔ PTFE fibril junction)
      Bond 4: CB–CB   (direct carbon black aggregate contact)

    PTFE vs PVDF bond differences
    ------------------------------
    PTFE fibrillates under shear — forms nano-fibrils that stretch before
    fracturing.  This is modelled by eps_c (fracture strain) 3–5× higher
    than PVDF values.  PVDF eps_c ~ 0.20–0.30; PTFE eps_c ~ 0.60–0.80.
    PTFE stiffness K is lower (softer binder, E ≈ 0.5 GPa vs 3–8 GPa).

    Bond coefficients for bond_style bpm/spring/plastic
    ----------------------------------------------------
    Format: (K, eps_c, gamma, eps_p)
      K      : spring stiffness [nN/µm]
      eps_c  : critical strain — bond breaks at strain > eps_c
      gamma  : viscous damping
      eps_p  : plastic yield strain (eps_p < eps_c required)
    """

    extra_bonds_per_atom: int = 24   # raised for 4 bond types
    extra_special_per_atom: int = 60

    create_bonds_after_settling: bool = True
    bond_relax_steps: int = 20_000
    bond_relax_viscous_damping: float = 30.0
    process_contact_clearance_um: float = 0.05
    damage_activation_fraction_of_compression: float = 0.0

    # ── Bond enable flags ───────────────────────────────────────────────────
    enable_am_am_bonds: bool   = False  # no direct AM-AM bonding
    enable_am_cb_bonds: bool   = True   # AM-CB: primary structural contact
    enable_am_ptfe_bonds: bool = True   # AM-PTFE: binder attachment to AM
    enable_cb_ptfe_bonds: bool = True   # CB-PTFE: fibril junction (most ductile)
    enable_cb_cb_bonds: bool   = True   # CB-CB: direct aggregate contact

    # ── Cutoff distances (µm) ───────────────────────────────────────────────
    am_am_cutoff_um:   float = 3.20   # AM r_max=1.8 → r_max+r_max+gap
    am_cb_cutoff_um:   float = 2.35   # AM r_max=1.8 + CB r=0.45 + gap
    am_ptfe_cutoff_um: float = 2.30   # AM r_max=1.8 + PTFE r=0.40 + gap
    cb_ptfe_cutoff_um: float = 1.00   # CB r=0.45 + PTFE r=0.40 + gap
    cb_cb_cutoff_um:   float = 1.05   # CB r=0.45 + CB r=0.45 + gap

    am_am_gap_tolerance_um:   float = 0.5
    am_cb_gap_tolerance_um:   float = 0.5
    am_ptfe_gap_tolerance_um: float = 0.5
    cb_ptfe_gap_tolerance_um: float = 0.3
    cb_cb_gap_tolerance_um:   float = 0.3

    # ── Bond 1: AM–CB ───────────────────────────────────────────────────────
    # Hard AM particle bonded to rigid CB aggregate — moderate stiffness
    # eps_c: CB aggregate pulls off AM surface at ~30% strain
    am_cb_coeffs: tuple[float, ...] = (
        350.0,   # K: moderate (CB-AM interface is rigid)
        0.30,    # eps_c: break at 30% (same order as old PVDF am-cbd)
        0.5,     # gamma
        0.10,    # eps_p: yield at 10%
    )

    # ── Bond 2: AM–PTFE ─────────────────────────────────────────────────────
    # PTFE fibril attached to AM surface — ductile stretching before failure
    # eps_c RAISED 2× vs PVDF: PTFE fibrils stretch significantly [5]
    am_ptfe_coeffs: tuple[float, ...] = (
        200.0,   # K: softer than AM-CB (PTFE E=0.5 GPa vs CB ~2 GPa)
        0.60,    # eps_c: PTFE fibril breaks at 60% strain (vs PVDF 30%)
        0.5,     # gamma
        0.15,    # eps_p: yield at 15% (wide plastic zone before fracture)
    )

    # ── Bond 3: CB–PTFE ─────────────────────────────────────────────────────
    # Fibril junction between CB aggregate and PTFE network — most ductile
    # This represents the load-bearing fibril network of the dry electrode
    cb_ptfe_coeffs: tuple[float, ...] = (
        100.0,   # K: softest bond (fibril network compliant)
        0.75,    # eps_c: break at 75% strain (extreme fibril stretching) [5]
        0.5,     # gamma
        0.20,    # eps_p: yield at 20%
    )

    # ── Bond 4: CB–CB ───────────────────────────────────────────────────────
    # Direct carbon aggregate contact — brittle, no PTFE mediation
    # Similar to old CBD-CBD but slightly higher eps_c (no PVDF embrittlement)
    cb_cb_coeffs: tuple[float, ...] = (
        150.0,   # K: same order as old CBD-CBD network
        0.25,    # eps_c: break at 25% (slightly higher than PVDF 22%)
        0.5,     # gamma
        0.08,    # eps_p: yield at 8%
    )

    # ── Legacy aliases (backward compat — remove after full migration) ───────
    am_am_coeffs:  tuple[float, ...] = (1000.0, 0.35, 0.5, 0.12)
    am_cbd_coeffs: tuple[float, ...] = am_cb_coeffs    # → am_cb_coeffs
    cbd_cbd_coeffs: tuple[float, ...] = cb_cb_coeffs   # → cb_cb_coeffs


def summarize() -> None:
    print("=== Geometry & Materials Config ===")
    print(f"Footprint (µm)        : {DomainConfig.lx_um} × {DomainConfig.ly_um}")
    print(f"Initial bed height    : {DomainConfig.initial_bed_height_um} µm")
    print(f"AM radii (µm)         : {ParticleTypes.am_radii_um}")
    print(f"CBD radius (µm)       : {ParticleTypes.cbd_radius_um}")
    print(f"Solid density         : {ParticleTypes.solid_density_g_cm3():.4f} g/cm³")
    print(f"Initial bulk density  : {ParticleTypes.initial_bulk_density_g_cm3} g/cm³")
    print(f"Initial porosity      : {ParticleTypes.initial_porosity():.4f}")
    print(f"SJKR CED (sim units)  : {ContactParams.sjkr_ced}  (~{ContactParams.sjkr_ced:.0f} kPa)")
    print(f"Reduction cases       : {LiteratureAnchors.reduction_cases}")
    print(f"Target density window : {LiteratureAnchors.target_density_window_g_cm3} g/cm³")


if __name__ == "__main__":
    summarize()