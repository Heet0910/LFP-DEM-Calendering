"""
pvdf_bondloss_ved_analysis.py
===============================
Bond-loss vs. VED analysis for PVDF (2-phase AM+CBD) calendering cases.
Cases: CR = 15, 20, 25, 30  |  Loading = 8, 11 mg/cm²  |  Speed = 1, 3, 6, 10 mm/s
Reads: simulations/CR{cr}/cal_cr{cr}_ld{ld}_sp{sp}/state_comparison.json

Key metrics extracted per case
--------------------------------
  bond_loss_pct   = (max_bonds - final_bonds) / max_bonds × 100
  VED_initial     = volumetric_energy_Wh_L  [non_calendered state]
  VED_final       = volumetric_energy_Wh_L  [final_recovered state]
  VED_gain_pct    = (VED_final - VED_initial) / VED_initial × 100
  penalty         = damaged_am_frac × mean_am_bond_loss_frac
  VED_eff         = VED_final × (1 - penalty)
  VED_eff_gain    = (VED_eff - VED_initial) / VED_initial × 100
  porosity_final  = porosity at final_recovered state
  springback_frac = springback / (initial_h - min_h)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import json
from pathlib import Path
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
ROOT   = Path("simulations")
OUTDIR = Path("figures_ptfe")
OUTDIR.mkdir(exist_ok=True)

CRS    = [15, 20, 25, 30]
LDS    = [8, 11]           # areal loading labels
SPS    = [1, 3, 6, 10]     # speed labels

CR_COLORS  = {15: "#2196F3", 20: "#4CAF50", 25: "#FF9800", 30: "#F44336"}
LD_MARKERS = {8: "o", 11: "s"}
DARK = "#0d1e5e"

# ── Parse all cases ────────────────────────────────────────────────────────────
records = []

for cr in CRS:
    cr_dir = ROOT / f"CR{cr}"
    if not cr_dir.exists():
        print(f"  [SKIP] {cr_dir} not found")
        continue
    for ld in LDS:
        for sp in SPS:
            case_name = f"cal_cr{cr}_ld{ld}_sp{sp}"
            sj = cr_dir / case_name / "state_comparison.json"
            if not sj.exists():
                print(f"  [MISS] {sj}")
                continue
            try:
                d = json.loads(sj.read_text())
                cs = d["case_summary"]
                st = d["states"]          # 0=initial 1=max_comp 2=final

                # locate states by label
                state_map = {s["state"]: s for s in st}
                s0 = state_map.get("non_calendered",    st[0])
                s2 = state_map.get("final_recovered",   st[-1])

                max_bonds    = cs["max_active_bonds"]
                final_bonds  = cs["final_active_bonds"]
                bond_loss    = (max_bonds - final_bonds) / max_bonds * 100 if max_bonds else 0

                VED_i = s0["volumetric_energy_Wh_L"]
                VED_f = s2["volumetric_energy_Wh_L"]
                VED_gain = (VED_f - VED_i) / VED_i * 100

                dam_frac  = s2.get("damaged_am_particle_fraction", 0)
                bond_loss_frac = s2.get("mean_am_bond_loss_fraction", 0)
                penalty   = dam_frac * bond_loss_frac
                VED_eff   = VED_f * (1.0 - penalty)
                VED_eff_gain = (VED_eff - VED_i) / VED_i * 100

                por_i = s0["porosity"]
                por_f = s2["porosity"]
                t_i   = s0["coating_thickness_um"]
                t_f   = s2["coating_thickness_um"]
                actual_cr = (t_i - t_f) / t_i * 100

                springback = cs.get("springback_relative_to_min", 0) * 100

                records.append(dict(
                    cr=cr, ld=ld, sp=sp, case=case_name,
                    max_bonds=max_bonds, final_bonds=final_bonds,
                    bond_loss=bond_loss,
                    VED_i=VED_i, VED_f=VED_f, VED_gain=VED_gain,
                    penalty=penalty, VED_eff=VED_eff, VED_eff_gain=VED_eff_gain,
                    dam_frac=dam_frac, bond_loss_frac=bond_loss_frac,
                    por_i=por_i, por_f=por_f,
                    actual_cr=actual_cr,
                    springback=springback,
                    peak_pressure=cs.get("peak_pressure", 0),
                ))
            except Exception as e:
                print(f"  [ERR] {sj}: {e}")

print(f"\nLoaded {len(records)} PVDF cases\n")

# ── CR-level summary ──────────────────────────────────────────────────────────
print(f"{'CR':>4} {'N':>3} {'BondLoss%':>10} {'VED_gain%':>10} {'VED_eff_gain%':>14} "
      f"{'Porosity_f':>11} {'Penalty':>8} {'Springback%':>12}")
print("-" * 80)
cr_summary = {}
for cr in CRS:
    sub = [r for r in records if r["cr"] == cr]
    if not sub: continue
    bl   = np.mean([r["bond_loss"]     for r in sub])
    vg   = np.mean([r["VED_gain"]      for r in sub])
    veg  = np.mean([r["VED_eff_gain"]  for r in sub])
    pf   = np.mean([r["por_f"]         for r in sub])
    pen  = np.mean([r["penalty"]       for r in sub])
    sb   = np.mean([r["springback"]    for r in sub])
    cr_summary[cr] = dict(bl=bl, vg=vg, veg=veg, pf=pf, pen=pen, sb=sb, n=len(sub))
    print(f"{cr:>4} {len(sub):>3} {bl:>10.2f} {vg:>10.2f} {veg:>14.2f} "
          f"{pf:>11.4f} {pen:>8.4f} {sb:>12.2f}")

# Pearson r: bond_loss vs VED_gain and vs VED_eff_gain
all_bl  = [r["bond_loss"]    for r in records]
all_vg  = [r["VED_gain"]     for r in records]
all_veg = [r["VED_eff_gain"] for r in records]
r1, p1  = stats.pearsonr(all_bl, all_vg)
r2, p2  = stats.pearsonr(all_bl, all_veg)
print(f"\nPearson r (bond_loss vs VED_gain):     r={r1:.4f}  p={p1:.4e}")
print(f"Pearson r (bond_loss vs VED_eff_gain): r={r2:.4f}  p={p2:.4e}")

# ── FIGURE 1: Bond-loss vs VED gain scatter (4-panel by CR) ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=160)
fig.suptitle("PVDF 2-Phase DEM — Bond Loss vs. VED Gain by Calendering Ratio",
             fontsize=14, fontweight="bold", color=DARK, y=0.98)

for ax, cr in zip(axes.flatten(), CRS):
    sub = [r for r in records if r["cr"] == cr]
    col = CR_COLORS[cr]

    # scatter by loading
    for ld in LDS:
        pts = [r for r in sub if r["ld"] == ld]
        xs  = [r["bond_loss"]   for r in pts]
        ys  = [r["VED_eff_gain"] for r in pts]
        sps_list = [r["sp"]    for r in pts]
        sc = ax.scatter(xs, ys, c=[r["sp"] for r in pts],
                        cmap="cool", vmin=1, vmax=10,
                        marker=LD_MARKERS[ld], s=90, edgecolors=col, lw=1.5, zorder=5)
        for x, y, sp in zip(xs, ys, sps_list):
            ax.annotate(f"{sp}", (x, y), fontsize=6.5, ha="left", va="bottom",
                        color="#555", xytext=(3, 3), textcoords="offset points")

    # trend line
    if len(sub) >= 3:
        xall = [r["bond_loss"]    for r in sub]
        yall = [r["VED_eff_gain"] for r in sub]
        m, b, rv, pv, _ = stats.linregress(xall, yall)
        xs_line = np.linspace(min(xall)-1, max(xall)+1, 80)
        ax.plot(xs_line, m*xs_line + b, "--", color=col, lw=1.4, alpha=0.7)
        ax.text(0.97, 0.05, f"r = {rv:.3f}\np = {pv:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=col,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=col, alpha=0.8))

    ax.axhline(0, color="#aaa", lw=0.8, ls=":")
    ax.set_title(f"CR = {cr}%", fontsize=11, fontweight="bold", color=col)
    ax.set_xlabel("Bond Loss  (%)", fontsize=9)
    ax.set_ylabel("VED_eff Gain  (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.4, alpha=0.5)

    # patch legend for loading
    p1_ = mpatches.Patch(edgecolor=col, facecolor="white",
                          label="ld = 8 mg/cm²  (○)", lw=1.5)
    p2_ = mpatches.Patch(edgecolor=col, facecolor=col,
                          label="ld = 11 mg/cm²  (□)", lw=1.5)
    ax.legend(handles=[p1_, p2_], fontsize=7, loc="upper left",
              framealpha=0.85, edgecolor="#ccc")

plt.colorbar(sc, ax=axes.flatten(), label="Calendering speed  (mm/s)",
             orientation="vertical", shrink=0.6, pad=0.02)
plt.tight_layout(rect=[0, 0, 0.93, 0.97])
out1 = OUTDIR / "pvdf_bondloss_vs_VED_scatter.png"
plt.savefig(out1, dpi=160, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out1}")

# ── FIGURE 2: CR summary — bar chart tradeoff ─────────────────────────────────
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5), dpi=160)
fig2.suptitle("PVDF DEM — Mean Metrics vs Calendering Ratio  (all speeds & loadings)",
              fontsize=13, fontweight="bold", color=DARK)

cr_vals  = list(cr_summary.keys())
cols     = [CR_COLORS[c] for c in cr_vals]
x        = np.arange(len(cr_vals))
w        = 0.55

# --- ax1: Bond Loss ---
bl_vals  = [cr_summary[c]["bl"]  for c in cr_vals]
bl_std   = [np.std([r["bond_loss"]    for r in records if r["cr"]==c]) for c in cr_vals]
bars1 = ax1.bar(x, bl_vals, w, color=cols, edgecolor="white", lw=1.2,
                yerr=bl_std, capsize=4, error_kw=dict(lw=1.5, capthick=1.5))
for b, v in zip(bars1, bl_vals):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels([f"CR={c}%" for c in cr_vals], fontsize=9)
ax1.set_ylabel("Mean Bond Loss  (%)", fontsize=10)
ax1.set_title("Bond Loss", fontsize=11, fontweight="bold")
ax1.set_ylim(0, max(bl_vals)*1.25)
ax1.grid(axis="y", lw=0.4, alpha=0.5)

# --- ax2: VED gain vs VED_eff gain ---
vg_vals  = [cr_summary[c]["vg"]  for c in cr_vals]
veg_vals = [cr_summary[c]["veg"] for c in cr_vals]
vg_std   = [np.std([r["VED_gain"]     for r in records if r["cr"]==c]) for c in cr_vals]
veg_std  = [np.std([r["VED_eff_gain"] for r in records if r["cr"]==c]) for c in cr_vals]
xi = x - w/4; xii = x + w/4
b2a = ax2.bar(xi, vg_vals,  w/2, color=cols, alpha=0.55, edgecolor="white", lw=1,
              yerr=vg_std, capsize=3, error_kw=dict(lw=1.2), label="VED_geo gain")
b2b = ax2.bar(xii, veg_vals, w/2, color=cols, alpha=0.95, edgecolor="white", lw=1,
              yerr=veg_std, capsize=3, error_kw=dict(lw=1.2), label="VED_eff gain")
for b, v in zip(b2b, veg_vals):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels([f"CR={c}%" for c in cr_vals], fontsize=9)
ax2.set_ylabel("Mean VED Gain  (%)", fontsize=10)
ax2.set_title("VED Gain (geo vs. eff)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.85)
ax2.grid(axis="y", lw=0.4, alpha=0.5)
ax2.set_ylim(0, max(vg_vals)*1.3)

# --- ax3: Porosity & Penalty ---
pf_vals  = [cr_summary[c]["pf"]  * 100 for c in cr_vals]
pen_vals = [cr_summary[c]["pen"] * 100 for c in cr_vals]
ax3b = ax3.twinx()
b3a = ax3.bar(x - w/4, pf_vals,  w/2, color=cols, alpha=0.7, edgecolor="white", lw=1,
              label="Final porosity (%)")
b3b = ax3b.bar(x + w/4, pen_vals, w/2, color=cols, alpha=0.95, edgecolor="white",
               hatch="//", lw=1, label="Damage penalty (%)")
for b, v in zip(b3a, pf_vals):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
             f"{v:.1f}", ha="center", va="bottom", fontsize=8)
for b, v in zip(b3b, pen_vals):
    ax3b.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
              f"{v:.1f}", ha="center", va="bottom", fontsize=8)
ax3.set_xticks(x); ax3.set_xticklabels([f"CR={c}%" for c in cr_vals], fontsize=9)
ax3.set_ylabel("Final Porosity  (%)", fontsize=10, color="#333")
ax3b.set_ylabel("Damage Penalty  (%)", fontsize=10, color="#555")
ax3.set_title("Porosity  &  Damage Penalty", fontsize=11, fontweight="bold")
lines = [mpatches.Patch(facecolor="#888", label="Final porosity"),
         mpatches.Patch(facecolor="#888", hatch="//", label="Damage penalty")]
ax3.legend(handles=lines, fontsize=8, framealpha=0.85)
ax3.grid(axis="y", lw=0.4, alpha=0.5)

plt.tight_layout()
out2 = OUTDIR / "pvdf_CR_summary_bars.png"
plt.savefig(out2, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ── FIGURE 3: Bond-loss vs VED_eff scatter — all cases, colored by CR ─────────
fig3, ax = plt.subplots(figsize=(9, 7), dpi=160)
ax.set_facecolor("#f5f8ff")

for cr in CRS:
    sub = [r for r in records if r["cr"] == cr]
    xs  = [r["bond_loss"]    for r in sub]
    ys  = [r["VED_eff_gain"] for r in sub]
    mks = [LD_MARKERS[r["ld"]] for r in sub]
    col = CR_COLORS[cr]
    for x_, y_, mk in zip(xs, ys, mks):
        ax.scatter(x_, y_, marker=mk, color=col, s=70,
                   edgecolors="white", lw=0.8, zorder=5, alpha=0.88)
    # mean point
    ax.scatter(np.mean(xs), np.mean(ys), marker="D", color=col,
               s=180, edgecolors="black", lw=1.5, zorder=10)
    ax.annotate(f"CR={cr}%\n(mean)",
                (np.mean(xs), np.mean(ys)),
                xytext=(8, 6), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color=col)

# Global trend line
slope, intercept, rv, pv, se = stats.linregress(all_bl, all_veg)
xs_line = np.linspace(min(all_bl)-2, max(all_bl)+2, 100)
ax.plot(xs_line, slope*xs_line + intercept, "--", color="#444", lw=1.8,
        label=f"Global trend  (r = {rv:.3f}, p = {pv:.4f})")

# Quadrant shading
ax.axhline(0, color="#aaa", lw=1, ls=":")
ax.axvline(np.mean(all_bl), color="#aaa", lw=1, ls=":")
ax.fill_between([min(all_bl)-3, np.mean(all_bl)],
                [np.mean(all_veg), np.mean(all_veg)],
                [max(all_veg)+3, max(all_veg)+3],
                color="#4CAF50", alpha=0.07, label="Low loss / High gain (optimal)")
ax.fill_between([np.mean(all_bl), max(all_bl)+3],
                [min(all_veg)-3, min(all_veg)-3],
                [np.mean(all_veg), np.mean(all_veg)],
                color="#F44336", alpha=0.07, label="High loss / Low gain (avoid)")

ax.set_xlabel("Bond Loss  (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("VED_eff Gain  (%)", fontsize=12, fontweight="bold")
ax.set_title("PVDF 2-Phase DEM — Bond Loss vs. VED\u2091\u2092\u2092 Gain\n"
             "All Calendering Conditions  (CR = 15–30%)",
             fontsize=12, fontweight="bold", color=DARK)

# Legend
legend_patches = [mpatches.Patch(facecolor=CR_COLORS[c], label=f"CR = {c}%") for c in CRS]
ld_patches = [mpatches.Patch(facecolor="#888", label="○ = ld 8 mg/cm²"),
              mpatches.Patch(facecolor="#888", label="□ = ld 11 mg/cm²")]
ax.legend(handles=legend_patches + ld_patches,
          loc="upper right", fontsize=8.5, framealpha=0.9, edgecolor="#ccc", ncol=2)
ax.grid(True, lw=0.4, alpha=0.4)
plt.tight_layout()
out3 = OUTDIR / "pvdf_bondloss_vs_VEDeff_allcases.png"
plt.savefig(out3, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

# ── FIGURE 4: Speed sensitivity — Bond-loss vs speed, facet by CR ─────────────
fig4, axes4 = plt.subplots(1, 4, figsize=(16, 5), dpi=160, sharey=True)
fig4.suptitle("PVDF — Bond Loss & VED_eff Gain vs. Calendering Speed  (by CR)",
              fontsize=13, fontweight="bold", color=DARK)

for ax4, cr in zip(axes4, CRS):
    col = CR_COLORS[cr]
    for ld in LDS:
        pts = sorted([r for r in records if r["cr"]==cr and r["ld"]==ld],
                     key=lambda r: r["sp"])
        sps_  = [r["sp"]         for r in pts]
        bls_  = [r["bond_loss"]  for r in pts]
        vegs_ = [r["VED_eff_gain"] for r in pts]
        ls_ = "-" if ld == 11 else "--"
        ax4.plot(sps_, bls_, ls_, marker=LD_MARKERS[ld], color=col,
                 lw=1.8, ms=7, label=f"BondLoss ld={ld}")
        ax4t = ax4.twinx() if cr == CRS[-1] else ax4.twinx()
        ax4t.plot(sps_, vegs_, ls_, marker=LD_MARKERS[ld], color="#888",
                  lw=1.4, ms=6, alpha=0.7)
        ax4t.set_ylabel("VED_eff Gain (%)", fontsize=8, color="#666")
        ax4t.tick_params(labelsize=7)

    ax4.set_xlabel("Speed  (mm/s)", fontsize=9)
    ax4.set_title(f"CR = {cr}%", fontsize=11, fontweight="bold", color=col)
    ax4.set_xticks(SPS)
    ax4.tick_params(labelsize=8)
    ax4.grid(True, lw=0.4, alpha=0.4)

axes4[0].set_ylabel("Bond Loss  (%)", fontsize=10)
plt.tight_layout()
out4 = OUTDIR / "pvdf_speed_sensitivity_bondloss.png"
plt.savefig(out4, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved: {out4}")

# ── Print detailed results table ───────────────────────────────────────────────
print("\n" + "="*110)
print(f"{'Case':<25} {'CR%':>4} {'Ld':>4} {'Sp':>4} {'BondLoss%':>10} "
      f"{'VED_i':>8} {'VED_f':>8} {'VEDgain%':>9} {'VED_eff':>8} {'VEDeff%':>9} "
      f"{'DamFrac':>8} {'Por_f':>7}")
print("-"*110)
for r in sorted(records, key=lambda x: (x["cr"], x["ld"], x["sp"])):
    print(f"{r['case']:<25} {r['cr']:>4} {r['ld']:>4} {r['sp']:>4} "
          f"{r['bond_loss']:>10.2f} {r['VED_i']:>8.1f} {r['VED_f']:>8.1f} "
          f"{r['VED_gain']:>9.2f} {r['VED_eff']:>8.1f} {r['VED_eff_gain']:>9.2f} "
          f"{r['dam_frac']:>8.4f} {r['por_f']:>7.4f}")

# ── Optimal case identification ────────────────────────────────────────────────
print("\n=== OPTIMAL CASE (max VED_eff_gain) ===")
best = max(records, key=lambda r: r["VED_eff_gain"])
print(f"  Case: {best['case']}")
print(f"  CR={best['cr']}%  ld={best['ld']} mg/cm²  sp={best['sp']} mm/s")
print(f"  Bond Loss     = {best['bond_loss']:.2f}%")
print(f"  VED_geo gain  = {best['VED_gain']:.2f}%")
print(f"  VED_eff gain  = {best['VED_eff_gain']:.2f}%")
print(f"  Damage penalty= {best['penalty']*100:.2f}%")
print(f"  Final porosity= {best['por_f']:.4f}")

print(f"\n=== LEAST DAMAGE CASE (min bond_loss) ===")
safest = min(records, key=lambda r: r["bond_loss"])
print(f"  Case: {safest['case']}")
print(f"  CR={safest['cr']}%  ld={safest['ld']} mg/cm²  sp={safest['sp']} mm/s")
print(f"  Bond Loss     = {safest['bond_loss']:.2f}%")
print(f"  VED_eff gain  = {safest['VED_eff_gain']:.2f}%")

print(f"\nAll figures saved to: {OUTDIR.resolve()}")
