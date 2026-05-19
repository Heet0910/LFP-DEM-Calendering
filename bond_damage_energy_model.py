"""
bond_damage_energy_model.py
===========================
Calculates the BOND-DAMAGE CORRECTED Volumetric Energy Density (VED).

Physics model
-------------
VED_geometric  = (1-ε) × ρ_solid × w_AM × C_g × V_avg   (porosity only)
VED_effective  = VED_geometric × η_connectivity

η_connectivity = connectivity efficiency factor:
  η = 1 - α×f_isolated - β×(f_damaged - f_isolated)×mean_loss

  f_isolated   : fraction of AM particles with ZERO remaining bonds → 0% utilizable
  f_damaged    : fraction of AM particles with >10% bond loss → partial loss
  mean_loss    : mean bond loss fraction for damaged particles
  α = 1.0      : full capacity loss for isolated particles
  β = 0.40     : partial loss for damaged (40% utilization penalty at mean bond loss)

Usage
-----
  python bond_damage_energy_model.py
  python bond_damage_energy_model.py --cases simulations/dry_trial_cr20_ld11

Outputs
-------
  - Terminal report with geometric vs effective VED
  - figures/fig_bond_corrected_ved.png
"""

import json
import argparse
import sys, io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Physical constants ────────────────────────────────────────────────────────
C_GRAVIMETRIC   = 150.0   # usable capacity mAh/g (LFP, conservative)
V_AVERAGE       = 3.4     # V vs Li/Li+
W_AM            = 0.94    # active material mass fraction (94:3:3)
RHO_SOLID       = 3.444   # g/cm³ harmonic mean solid density

# ── Penalty coefficients (literature-anchored) ────────────────────────────────
ALPHA = 1.00   # isolated AM → 100% capacity lost
BETA  = 0.40   # damaged AM  → 40% utilization penalty per unit mean bond loss

# ── VED formulae ─────────────────────────────────────────────────────────────
def ved_geometric(porosity: float) -> float:
    """Wh/L — electrode layer only, porosity-based."""
    return (1 - porosity) * RHO_SOLID * W_AM * C_GRAVIMETRIC * V_AVERAGE / 3.6

def connectivity_efficiency(f_isolated: float, f_damaged: float, mean_loss: float) -> float:
    """Fraction of VED actually deliverable after bond damage."""
    penalty = ALPHA * f_isolated + BETA * max(f_damaged - f_isolated, 0) * mean_loss
    return max(0.0, 1.0 - penalty)

def ved_effective(porosity: float, f_iso: float, f_dmg: float, mean_loss: float) -> float:
    eta = connectivity_efficiency(f_iso, f_dmg, mean_loss)
    return ved_geometric(porosity) * eta

# ── Load one simulation case ──────────────────────────────────────────────────
def load_case(path: Path) -> dict | None:
    sc = path / "state_comparison.json"
    if not sc.exists():
        return None
    data = json.loads(sc.read_text())
    states = {s["state"]: s for s in data["states"]}
    fn = states.get("final_recovered", {})
    nc = states.get("non_calendered", {})
    summary = data.get("case_summary", {})

    cr   = fn.get("reduction_vs_initial_pct", 0) / 100
    eps  = fn.get("porosity", 0)
    rho  = fn.get("bulk_density_g_cm3", 0)
    ved_geo = fn.get("volumetric_energy_Wh_L", ved_geometric(eps))

    # Bond damage metrics from final state
    f_iso   = fn.get("am_zero_bond_fraction", 0)
    f_dmg   = fn.get("damaged_am_particle_fraction", 0)
    ml      = fn.get("mean_am_bond_loss_fraction", 0)

    # Also from initial state to compute newly-isolated fraction
    nc_iso  = nc.get("am_zero_bond_fraction", 0)
    f_iso_new = max(f_iso - nc_iso, 0)   # particles newly isolated by calendering

    eta     = connectivity_efficiency(f_iso_new, f_dmg, ml)
    ved_eff = ved_geo * eta
    ved_loss= ved_geo - ved_eff

    return {
        "label":    path.name,
        "cr":       cr,
        "porosity": eps,
        "rho":      rho,
        "ved_geo":  ved_geo,
        "ved_eff":  ved_eff,
        "ved_loss": ved_loss,
        "eta":      eta,
        "f_iso":    f_iso,
        "f_iso_new":f_iso_new,
        "f_dmg":    f_dmg,
        "mean_loss":ml,
        "peak_p":   summary.get("peak_pressure", 0),
        "springback":summary.get("springback_relative_to_min", 0),
    }

# ── Theoretical curves (analytical model) ────────────────────────────────────
def theoretical_curves():
    """Generate smooth CR sweep for analytical curves."""
    cr = np.linspace(0, 0.40, 200)
    eps0 = 0.442                          # our measured initial porosity

    # Porosity decreases with CR (linear approx — good for LFP 0-35%)
    eps = eps0 * (1 - cr) - 0.05 * cr    # slight over-compression effect

    ved_geo_curve = np.array([ved_geometric(e) for e in eps])

    # Bond fracture grows nonlinearly beyond CR~20% for PTFE
    # Calibrated: low fracture below CR=20%, accelerating above (PTFE fibril network)
    f_iso_curve = np.where(cr < 0.20,
                           0.06 + 0.20 * cr,                    # slow growth
                           0.06 + 0.20 * 0.20 + 1.5 * (cr - 0.20)**1.5)  # accelerating
    f_iso_curve = np.clip(f_iso_curve, 0, 0.80)

    f_dmg_curve  = np.clip(f_iso_curve * 3.5, 0, 1.0)
    mean_l_curve = np.clip(0.10 + 0.8 * cr, 0, 0.9)

    # Only newly-isolated (subtract initial ~6%)
    f_iso_new_curve = np.clip(f_iso_curve - 0.06, 0, 1.0)

    eta_curve    = np.array([connectivity_efficiency(fi, fd, ml)
                             for fi, fd, ml in zip(f_iso_new_curve, f_dmg_curve, mean_l_curve)])
    ved_eff_curve = ved_geo_curve * eta_curve

    return cr, ved_geo_curve, ved_eff_curve, eta_curve, f_iso_new_curve

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(cases: list[dict], save_dir: Path):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), facecolor="#0d1117")
    BG = "#161b22"
    cr_c, ved_geo_c, ved_eff_c, eta_c, fiso_c = theoretical_curves()
    cr_pct = cr_c * 100

    # ── Panel 1: VED vs CR ────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.plot(cr_pct, ved_geo_c, "--", color="#4fc3f7", lw=2.0, label="VED (porosity only)")
    ax.plot(cr_pct, ved_eff_c, "-",  color="#69f0ae", lw=2.5, label="VED (bond-corrected)")
    ax.fill_between(cr_pct, ved_eff_c, ved_geo_c, alpha=0.15, color="#ef5350",
                    label="Bond damage penalty")

    # Optimal CR marker
    opt_idx = np.argmax(ved_eff_c)
    ax.axvline(cr_pct[opt_idx], color="#ffd54f", lw=1.5, ls=":", alpha=0.8)
    ax.annotate(f"Optimal CR\n{cr_pct[opt_idx]:.1f}%",
                xy=(cr_pct[opt_idx], ved_eff_c[opt_idx]),
                xytext=(cr_pct[opt_idx]+2, ved_eff_c[opt_idx]-60),
                color="#ffd54f", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#ffd54f", lw=1.2))

    for c in cases:
        ax.scatter([c["cr"]*100], [c["ved_geo"]], marker="D", s=100,
                   color="#4fc3f7", zorder=10, label=f"Sim geo: {c['label']}")
        ax.scatter([c["cr"]*100], [c["ved_eff"]], marker="*", s=180,
                   color="#69f0ae", zorder=10, label=f"Sim eff: {c['label']}")
        ax.annotate(f"-{c['ved_loss']:.0f} Wh/L\nloss",
                    xy=(c["cr"]*100, c["ved_eff"]),
                    xytext=(c["cr"]*100+1.5, c["ved_eff"]-80),
                    color="#ef5350", fontsize=8.5)

    ax.set_xlabel("Calendering Ratio CR (%)", color="#aaa")
    ax.set_ylabel("Volumetric Energy Density (Wh/L)", color="#aaa")
    ax.set_title("Geometric vs Bond-Corrected VED", color="white", fontweight="bold")
    ax.legend(fontsize=8, facecolor="#1e2430", edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="#aaa")
    ax.grid(color="#222", ls="--", lw=0.6)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")

    # ── Panel 2: Connectivity efficiency η ────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.plot(cr_pct, eta_c * 100, "-", color="#ff8a65", lw=2.5, label="Connectivity η (%)")
    ax2.plot(cr_pct, fiso_c * 100, "--", color="#ef5350", lw=1.8, label="Isolated AM (%)")
    ax2.axhline(90, color="#ffd54f", lw=1, ls=":", alpha=0.6)
    ax2.text(1, 91, "η = 90% threshold", color="#ffd54f", fontsize=8.5)

    for c in cases:
        ax2.scatter([c["cr"]*100], [c["eta"]*100], marker="*", s=180,
                    color="#ff8a65", zorder=10)
        ax2.scatter([c["cr"]*100], [c["f_iso_new"]*100], marker="D", s=100,
                    color="#ef5350", zorder=10)

    ax2.set_xlabel("Calendering Ratio CR (%)", color="#aaa")
    ax2.set_ylabel("Fraction (%)", color="#aaa")
    ax2.set_title("Bond Connectivity vs CR", color="white", fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9, facecolor="#1e2430", edgecolor="#333", labelcolor="white")
    ax2.tick_params(colors="#aaa")
    ax2.grid(color="#222", ls="--", lw=0.6)
    for sp in ax2.spines.values(): sp.set_edgecolor("#333")

    # ── Panel 3: VED loss (penalty) ────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(BG)
    ved_loss_c = ved_geo_c - ved_eff_c
    ax3.fill_between(cr_pct, 0, ved_loss_c, alpha=0.4, color="#ef5350")
    ax3.plot(cr_pct, ved_loss_c, "-", color="#ef5350", lw=2.5, label="VED penalty (bond damage)")
    ax3.plot(cr_pct, ved_geo_c - ved_geo_c[0], "--", color="#4fc3f7", lw=2.0, label="VED gain (porosity)")

    for c in cases:
        ax3.scatter([c["cr"]*100], [c["ved_loss"]], marker="*", s=180,
                    color="#ef5350", zorder=10, label=f"Sim: {c['ved_loss']:.0f} Wh/L lost")

    ax3.set_xlabel("Calendering Ratio CR (%)", color="#aaa")
    ax3.set_ylabel("Wh/L", color="#aaa")
    ax3.set_title("VED Gain vs Bond-Damage Penalty", color="white", fontweight="bold")
    ax3.legend(fontsize=9, facecolor="#1e2430", edgecolor="#333", labelcolor="white")
    ax3.tick_params(colors="#aaa")
    ax3.grid(color="#222", ls="--", lw=0.6)
    for sp in ax3.spines.values(): sp.set_edgecolor("#333")

    fig.suptitle("Bond-Damage Corrected Volumetric Energy Density — LFP PTFE Dry Electrode",
                 fontsize=13, fontweight="bold", color="white", y=1.01)
    fig.text(0.5, -0.03,
             "VED_eff = VED_geo x (1 - alpha*f_isolated - beta*f_damaged*mean_bond_loss) "
             " | alpha=1.0, beta=0.40",
             ha="center", fontsize=8, color="#666")

    plt.tight_layout()
    out = save_dir / "fig_bond_corrected_ved.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.show()

# ── Report ────────────────────────────────────────────────────────────────────
def print_report(cases: list[dict]):
    cr_c, ved_geo_c, ved_eff_c, eta_c, fiso_c = theoretical_curves()
    opt_idx = np.argmax(ved_eff_c)
    print("\n" + "="*70)
    print("  BOND-DAMAGE CORRECTED VED ANALYSIS")
    print("="*70)
    print(f"\n  Model parameters:")
    print(f"    C_gravimetric = {C_GRAVIMETRIC} mAh/g  |  V_avg = {V_AVERAGE} V")
    print(f"    w_AM = {W_AM*100:.0f}%  |  rho_solid = {RHO_SOLID} g/cm3")
    print(f"    alpha (isolated penalty) = {ALPHA}")
    print(f"    beta  (damaged penalty)  = {BETA}")
    print(f"\n  Analytical model optimal CR: {cr_c[opt_idx]*100:.1f}%")
    print(f"  Peak VED_effective:          {ved_eff_c[opt_idx]:.0f} Wh/L")
    print(f"  VED_geometric at same CR:    {ved_geo_c[opt_idx]:.0f} Wh/L")

    if cases:
        print(f"\n  Simulation data points:")
        hdr = f"  {'Case':<28} {'CR':>6} {'eps':>7} {'VED_geo':>9} {'VED_eff':>9} {'Loss':>8} {'eta':>7} {'f_iso_new':>10}"
        print(hdr)
        print("  " + "-"*90)
        for c in cases:
            print(f"  {c['label']:<28} {c['cr']*100:>5.1f}% {c['porosity']:>7.4f} "
                  f"{c['ved_geo']:>9.1f} {c['ved_eff']:>9.1f} "
                  f"{c['ved_loss']:>7.1f} {c['eta']*100:>6.1f}% {c['f_iso_new']*100:>9.1f}%")

    print("\n  Interpretation:")
    print("    - VED_geo measures porosity-driven density gain")
    print("    - VED_eff = real deliverable energy after bond damage")
    print("    - Difference = capacity lost due to AM isolation")
    print("    - Optimal CR = peak of VED_eff curve")
    print("    - Beyond optimal CR: bond damage > porosity gain")
    print("="*70)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*",
                        default=["simulations/dry_trial_cr20_ld11"],
                        help="One or more simulation case paths")
    args = parser.parse_args()

    cases = []
    for path_str in args.cases:
        p = Path(path_str)
        if not p.exists():
            print(f"  [skip] {p} not found")
            continue
        # Handle both direct case paths and CR*/case_name/ nesting
        if (p / "state_comparison.json").exists():
            c = load_case(p)
            if c:
                cases.append(c)
        else:
            # Search subdirectories
            for sub in sorted(p.rglob("state_comparison.json")):
                c = load_case(sub.parent)
                if c:
                    cases.append(c)

    if not cases:
        print("  No simulation data found — running analytical model only.")

    print_report(cases)
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    plot(cases, fig_dir)

if __name__ == "__main__":
    main()
