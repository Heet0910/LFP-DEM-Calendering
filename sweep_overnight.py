"""
sweep_overnight.py
==================
Overnight parametric sweep for LFP 3-phase PTFE dry electrode.

Design:
  CR        : 15%, 20%, 25%, 30%
  Loading   : 8 mg/cm², 11.36 mg/cm²
  Speed     : 0.001, 0.003, 0.006 µm/µs
  Total     : 4 × 2 × 3 = 24 cases

Usage:
  python sweep_overnight.py              # run all 24 cases
  python sweep_overnight.py --dry-run    # print commands only, no execution
  python sweep_overnight.py --skip-done  # skip cases that already have state_comparison.json

Outputs per case:
  simulations/<label>/case_config.json
  simulations/<label>/pressure_log.txt
  simulations/<label>/state_comparison.json
  simulations/<label>/state_comparison.csv

After all cases:
  figures/fig_bond_corrected_ved.png  (updated with all 24 points)
  sweep_results_summary.csv           (consolidated table)
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent
SIM_ROOT       = PROJECT_ROOT / "simulations"
POSTPROC       = PROJECT_ROOT / "postproc" / "compare_calendered_states.py"
BOND_MODEL     = PROJECT_ROOT / "bond_damage_energy_model.py"
LOG_FILE       = PROJECT_ROOT / "sweep_overnight.log"

# ── Sweep matrix ─────────────────────────────────────────────────────────────
CRS      = [0.15, 0.20, 0.25, 0.30]
LOADINGS = [8.0, 11.36]
SPEEDS   = [0.001, 0.003, 0.006]

# Speed label suffix: 0.001→sp1, 0.003→sp3, 0.006→sp6
SPEED_LABEL = {0.001: "sp1", 0.003: "sp3", 0.006: "sp6"}
# Loading label suffix: 8→ld8, 11.36→ld11
LOADING_LABEL = {8.0: "ld8", 11.36: "ld11"}

TIMESTEP_US   = 0.001   # stable for all three speeds (matches verified old cases)
SETTLING      = 8_000
HOLD          = 2_000
THERMO_EVERY  = 1_000
DUMP_EVERY    = 2_000


def make_label(cr: float, loading: float, speed: float) -> str:
    cr_int = int(round(cr * 100))
    return f"dry_cr{cr_int}_{LOADING_LABEL[loading]}_{SPEED_LABEL[speed]}"


def estimate_steps(cr: float, loading: float, speed: float) -> int:
    """Rough total step count for time estimate."""
    rho_bulk = 2.10  # g/cm3
    bed_h = loading / (rho_bulk * 0.1)   # µm
    disp = bed_h * cr
    comp_steps = int(disp / (speed * TIMESTEP_US))
    return SETTLING + comp_steps + HOLD + comp_steps + 10_000  # +free relax


def build_sim_cmd(cr: float, loading: float, speed: float) -> list[str]:
    label = make_label(cr, loading, speed)
    return [
        sys.executable, str(PROJECT_ROOT / "run_calendering.py"),
        "--label",                label,
        "--compression-ratio",    str(cr),
        "--areal-loading-mg-cm2", str(loading),
        "--compression-velocity", str(-speed),
        "--decompression-velocity", str(speed),
        "--timestep-us",          str(TIMESTEP_US),
        "--settling-steps",       str(SETTLING),
        "--hold-steps",           str(HOLD),
        "--thermo-every",         str(THERMO_EVERY),
        "--dump-every",           str(DUMP_EVERY),
        "--regenerate-structure",
        "--allow-high-risk",
    ]


def build_postproc_cmd(case_dir: Path) -> list[str]:
    return [
        sys.executable, str(POSTPROC),
        "--case-dir",                          str(case_dir),
        "--q-usable",                          "150",
        "--voltage",                           "3.4",
        "--fracture-threshold-gpa",            "1.5",
        "--bond-loss-damage-threshold-fraction", "0.10",
    ]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_done(label: str) -> bool:
    for p in SIM_ROOT.rglob(f"{label}/state_comparison.json"):
        return True
    return False


def find_case_dir(label: str) -> Path | None:
    for p in SIM_ROOT.rglob(f"{label}/case_config.json"):
        return p.parent
    return SIM_ROOT / label


def run_cmd(cmd: list[str], desc: str, dry_run: bool) -> bool:
    log(f"  START: {desc}")
    if dry_run:
        log(f"  [DRY-RUN] {' '.join(cmd)}")
        return True
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    log(f"  END: {desc} — {status} in {elapsed:.0f}s")
    return ok


def build_all_cases() -> list[dict]:
    cases = []
    for cr in CRS:
        for loading in LOADINGS:
            for speed in SPEEDS:
                cases.append({"cr": cr, "loading": loading, "speed": speed,
                               "label": make_label(cr, loading, speed)})
    return cases


def print_sweep_plan(cases: list[dict]) -> None:
    print("\n" + "="*72)
    print("  OVERNIGHT SWEEP PLAN — LFP 3-Phase PTFE Dry Electrode")
    print("="*72)
    print(f"  Total cases : {len(cases)}")
    print(f"  Timestep    : {TIMESTEP_US} µs")
    print(f"  Bond fracture: enabled (4 bond types)")
    print()
    total_est = 0
    hdr = f"  {'Label':<28} {'CR':>5} {'Load':>6} {'Speed':>7} {'~Steps':>12} {'~Time':>8}"
    print(hdr)
    print("  " + "-"*70)
    lammps_rate = 8_000  # steps/sec estimate
    for c in cases:
        steps = estimate_steps(c["cr"], c["loading"], c["speed"])
        secs  = steps / lammps_rate
        total_est += secs
        print(f"  {c['label']:<28} {c['cr']*100:>4.0f}% {c['loading']:>6.2f} "
              f"{c['speed']:>7.3f} {steps:>12,} {secs/60:>7.1f}m")
    print()
    print(f"  Estimated total run time: {total_est/3600:.1f} hours")
    print("="*72 + "\n")


def write_summary_csv(cases: list[dict]) -> None:
    """Collect results from all state_comparison.json files into one CSV."""
    rows = []
    for c in cases:
        case_dir = find_case_dir(c["label"])
        if case_dir is None:
            continue
        sc_path = case_dir / "state_comparison.json"
        if not sc_path.exists():
            continue
        try:
            sc = json.loads(sc_path.read_text(encoding="utf-8"))
            states = {s["state"]: s for s in sc.get("states", [])}
            fn = states.get("final_recovered", {})
            nc = states.get("non_calendered", {})
            # Bond-corrected VED
            ved_geo = float(fn.get("volumetric_energy_loading_based_Wh_L") or 0)
            f_iso_i = float(nc.get("am_zero_bond_fraction", 0))
            f_iso_f = float(fn.get("am_zero_bond_fraction", 0))
            f_iso_n = max(f_iso_f - f_iso_i, 0)
            f_dmg   = float(fn.get("damaged_am_particle_fraction", 0))
            ml      = float(fn.get("mean_am_bond_loss_fraction", 0))
            penalty = 1.0 * f_iso_n + 0.4 * max(f_dmg - f_iso_n, 0) * ml
            eta     = max(0.0, 1.0 - penalty)
            ved_eff = ved_geo * eta if ved_geo > 0 else 0.0
            csm = sc.get("case_summary", {})
            rows.append({
                "label":         c["label"],
                "cr_target":     c["cr"],
                "loading_mg_cm2": c["loading"],
                "speed_um_us":   c["speed"],
                "cr_actual":     round(fn.get("reduction_vs_initial_pct", 0) / 100, 4),
                "porosity":      round(float(fn.get("porosity", 0)), 4),
                "bulk_density":  round(float(fn.get("bulk_density_g_cm3", 0)), 4),
                "peak_pressure_mpa": round(float(csm.get("peak_pressure", 0)) / 1000, 3),
                "springback_pct":round(float(csm.get("springback_relative_to_min", 0)) * 100, 2),
                "ved_geo_whl":   round(ved_geo, 1),
                "ved_eff_whl":   round(ved_eff, 1),
                "ved_loss_whl":  round(ved_geo - ved_eff, 1),
                "eta_pct":       round(eta * 100, 1),
                "f_iso_new":     round(f_iso_n, 4),
                "f_damaged":     round(f_dmg, 4),
                "mean_bond_loss": round(ml, 4),
            })
        except Exception as e:
            log(f"  [WARN] Could not parse {sc_path}: {e}")

    if not rows:
        log("  No completed cases found for summary CSV.")
        return

    out = PROJECT_ROOT / "sweep_results_summary.csv"
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log(f"  Summary CSV written: {out}  ({len(rows)} rows)")

    # Print table to terminal
    print("\n" + "="*80)
    print("  SWEEP RESULTS SUMMARY")
    print("="*80)
    hdr = (f"  {'Label':<28} {'CR%':>5} {'LD':>5} {'ε':>7} "
           f"{'VED_geo':>8} {'VED_eff':>8} {'η%':>6} {'Loss':>7}")
    print(hdr)
    print("  " + "-"*78)
    for r in sorted(rows, key=lambda x: (x["loading_mg_cm2"], x["speed_um_us"], x["cr_target"])):
        print(f"  {r['label']:<28} {r['cr_actual']*100:>4.1f}% "
              f"{r['loading_mg_cm2']:>5.1f} {r['porosity']:>7.4f} "
              f"{r['ved_geo_whl']:>8.1f} {r['ved_eff_whl']:>8.1f} "
              f"{r['eta_pct']:>5.1f}% {r['ved_loss_whl']:>7.1f}")
    print("="*80)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip cases that already have state_comparison.json")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only write summary CSV from completed cases")
    args = parser.parse_args()

    cases = build_all_cases()
    print_sweep_plan(cases)

    if args.summary_only:
        write_summary_csv(cases)
        return

    t_sweep_start = time.time()
    log(f"Sweep started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Total cases: {len(cases)}")

    passed = failed = skipped = 0

    for i, c in enumerate(cases, 1):
        label    = c["label"]
        cr       = c["cr"]
        loading  = c["loading"]
        speed    = c["speed"]

        log(f"\n[{i:02d}/{len(cases)}] {label}  CR={cr*100:.0f}%  ld={loading:.2f}  v={speed}")

        if args.skip_done and is_done(label):
            log(f"  SKIP — state_comparison.json already exists")
            skipped += 1
            continue

        # ── 1. Run simulation ──────────────────────────────────────────────
        sim_cmd = build_sim_cmd(cr, loading, speed)
        ok = run_cmd(sim_cmd, f"LAMMPS simulation {label}", args.dry_run)

        if not ok:
            log(f"  [FAIL] Simulation failed for {label} — skipping postproc")
            failed += 1
            continue

        # ── 2. Run postprocessing ──────────────────────────────────────────
        case_dir = find_case_dir(label)
        if case_dir and (case_dir.exists() or args.dry_run):
            pp_cmd = build_postproc_cmd(case_dir or SIM_ROOT / label)
            run_cmd(pp_cmd, f"Postprocessing {label}", args.dry_run)
        else:
            log(f"  [WARN] Case directory not found for {label} — skipping postproc")

        passed += 1

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t_sweep_start
    log(f"\n{'='*60}")
    log(f"Sweep complete in {elapsed/3600:.2f} hours")
    log(f"Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    log(f"{'='*60}")

    # ── Bond-corrected VED analysis for all cases ──────────────────────────
    if not args.dry_run and passed > 0:
        all_case_dirs = [str(find_case_dir(c["label"])) for c in cases
                         if is_done(c["label"]) and find_case_dir(c["label"])]
        if all_case_dirs:
            log("Running bond-damage corrected VED analysis on all completed cases...")
            run_cmd([sys.executable, str(BOND_MODEL), "--cases"] + all_case_dirs,
                    "Bond-corrected VED (all cases)", False)

    # ── Write consolidated results CSV ──────────────────────────────────────
    write_summary_csv(cases)


if __name__ == "__main__":
    main()
