"""
monitor_sweep.py
================
Checks all 24 sweep cases every 60 minutes.
Verifies parameters, tracks progress, alerts on problems.

Usage (run in a SECOND terminal while sweep runs):
  python monitor_sweep.py               # check every 60 min
  python monitor_sweep.py --interval 15 # check every 15 min
  python monitor_sweep.py --once        # check right now and exit
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SIM_ROOT     = PROJECT_ROOT / "simulations"
MONITOR_LOG  = PROJECT_ROOT / "monitor_sweep.log"

# ── Expected sweep spec (must match sweep_overnight.py) ──────────────────────
CRS           = [0.15, 0.20, 0.25, 0.30]
LOADINGS      = {8.0: "ld8", 11.36: "ld11"}
SPEEDS        = {0.001: "sp1", 0.003: "sp3", 0.006: "sp6"}
TIMESTEP_US   = 0.001
SETTLING      = 8_000
HOLD          = 2_000

EXPECTED_BOND_TOPOLOGY = {
    "am_am": False, "am_cb": True, "am_ptfe": True, "cb_ptfe": True, "cb_cb": True
}
EXPECTED_BOND_COEFFS = {
    "am_cb":   [350.0, 0.30, 0.5, 0.10],
    "am_ptfe": [200.0, 0.60, 0.5, 0.15],
    "cb_ptfe": [100.0, 0.75, 0.5, 0.20],
    "cb_cb":   [150.0, 0.25, 0.5, 0.08],
}
EXPECTED_KN = 50_000.0

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_label(cr: float, ld_key: float, sp_key: float) -> str:
    cr_int = int(round(cr * 100))
    return f"dry_cr{cr_int}_{LOADINGS[ld_key]}_{SPEEDS[sp_key]}"


def all_expected_cases() -> list[dict]:
    cases = []
    for cr in CRS:
        for ld in LOADINGS:
            for sp in SPEEDS:
                cases.append({"label": make_label(cr, ld, sp),
                               "cr": cr, "loading": ld, "speed": sp})
    return cases


def find_case_dir(label: str) -> Path | None:
    for p in SIM_ROOT.rglob(f"{label}/case_config.json"):
        return p.parent
    return None


def read_progress(case_dir: Path, config: dict) -> dict:
    """Read pressure_log.txt and return progress info."""
    pl = case_dir / "pressure_log.txt"
    if not pl.exists():
        return {"status": "NO_LOG", "pct": 0, "timestep": 0, "bonds": "-"}
    try:
        lines = [l for l in pl.read_text(encoding="utf-8", errors="replace").splitlines()
                 if l.strip() and not l.startswith("#")]
        if len(lines) < 2:
            return {"status": "STARTING", "pct": 0, "timestep": 0, "bonds": "-"}
        last = lines[-1].split()
        ts = int(float(last[0]))
        bonds = last[4] if len(last) > 4 else "-"
        total = (config.get("settling_steps", 0) + config.get("compression_steps", 0) +
                 config.get("hold_steps", 0) + config.get("decompression_steps", 0))
        pct = ts / total * 100 if total > 0 else 0
        status = "RUNNING" if pct < 99 else "LAMMPS_DONE"
        return {"status": status, "pct": pct, "timestep": ts, "bonds": bonds}
    except Exception as e:
        return {"status": f"READ_ERR:{e}", "pct": 0, "timestep": 0, "bonds": "-"}


def check_config(config: dict, expected: dict) -> list[str]:
    """Return list of problems found. Empty list = all good."""
    problems = []

    # CR
    got_cr = config.get("compression_ratio", -1)
    if abs(got_cr - expected["cr"]) > 0.001:
        problems.append(f"CR={got_cr} expected {expected['cr']}")

    # Speed
    got_v = abs(config.get("compression_velocity", 0))
    if abs(got_v - expected["speed"]) > 1e-6:
        problems.append(f"speed={got_v} expected {expected['speed']}")

    # Loading
    got_ld = config.get("areal_loading_mg_cm2", 0)
    tol = 0.6  # ±0.6 mg/cm2 tolerance (actual particle count varies slightly)
    if abs(got_ld - expected["loading"]) > tol:
        problems.append(f"loading={got_ld:.2f} expected {expected['loading']:.2f}")

    # Timestep
    got_dt = config.get("timestep_us", -1)
    if abs(got_dt - TIMESTEP_US) > 1e-9:
        problems.append(f"timestep={got_dt} expected {TIMESTEP_US}")

    # Bonded fracture
    if not config.get("bonded_fracture", False):
        problems.append("bonded_fracture=False (must be True)")

    # Bond topology
    bt = config.get("bond_topology", {})
    for k, ev in EXPECTED_BOND_TOPOLOGY.items():
        if bt.get(k) != ev:
            problems.append(f"bond_topology.{k}={bt.get(k)} expected {ev}")

    # Bond coefficients
    bc = config.get("bond_coefficients", {})
    for k, exp_coeffs in EXPECTED_BOND_COEFFS.items():
        got = bc.get(k, [])
        if not got or any(abs(g - e) > 1e-9 for g, e in zip(got, exp_coeffs)):
            problems.append(f"bond_coeffs.{k}={got} expected {exp_coeffs}")

    # kn
    kn = config.get("contact_model", {}).get("hertz_kn", 0)
    if abs(kn - EXPECTED_KN) > 1.0:
        problems.append(f"hertz_kn={kn} expected {EXPECTED_KN}")

    return problems


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with MONITOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── Main check ────────────────────────────────────────────────────────────────
def run_check(cases: list[dict]) -> dict:
    """Run one full check cycle. Returns summary counts."""
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print("=" * 76)
    print(f"  SWEEP MONITOR CHECK — {now}")
    print("=" * 76)

    col_hdr = (f"  {'Label':<28} {'Status':<12} {'Prog':>6} "
               f"{'TS':>10} {'Bonds':>6} {'Problems'}")
    print(col_hdr)
    print("  " + "-" * 74)

    counts = {"done": 0, "running": 0, "pending": 0, "failed": 0, "config_err": 0}

    for c in cases:
        label     = c["label"]
        case_dir  = find_case_dir(label)

        # ── Case not started yet ───────────────────────────────────────────
        if case_dir is None:
            print(f"  {label:<28} {'PENDING':<12} {'':>6} {'':>10} {'':>6} —")
            counts["pending"] += 1
            continue

        # ── Load config ────────────────────────────────────────────────────
        cfg_path = case_dir / "case_config.json"
        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  {label:<28} {'CFG_ERR':<12} — cannot read config: {e}")
            counts["config_err"] += 1
            continue

        # ── Parameter check ────────────────────────────────────────────────
        problems = check_config(config, c)

        # ── Progress ───────────────────────────────────────────────────────
        pp_done = (case_dir / "state_comparison.json").exists()
        if pp_done:
            status = "COMPLETE"
            prog   = "100.0%"
            ts_str = "-"
            bonds  = "-"
            counts["done"] += 1
        else:
            prog_info = read_progress(case_dir, config)
            status = prog_info["status"]
            prog   = f"{prog_info['pct']:.1f}%"
            ts_str = str(prog_info["timestep"])
            bonds  = str(prog_info["bonds"])
            if status in ("RUNNING", "STARTING"):
                counts["running"] += 1
            elif status == "LAMMPS_DONE":
                counts["running"] += 1  # postproc in progress
            else:
                counts["pending"] += 1

        # ── Print row ──────────────────────────────────────────────────────
        prob_str = "" if not problems else "⚠ " + " | ".join(problems)
        if problems:
            counts["config_err"] += 1
        print(f"  {label:<28} {status:<12} {prog:>6} {ts_str:>10} {bonds:>6}  {prob_str}")

    print()
    print(f"  Summary: DONE={counts['done']}  RUNNING={counts['running']}  "
          f"PENDING={counts['pending']}  ERRORS={counts['config_err'] + counts['failed']}")
    print("=" * 76)

    # ── Bond health for running cases ──────────────────────────────────────
    for c in cases:
        label    = c["label"]
        case_dir = find_case_dir(label)
        if case_dir is None:
            continue
        pp_done = (case_dir / "state_comparison.json").exists()
        if pp_done:
            continue
        pl = case_dir / "pressure_log.txt"
        if not pl.exists():
            continue
        try:
            lines = [l for l in pl.read_text(encoding="utf-8", errors="replace").splitlines()
                     if l.strip() and not l.startswith("#")]
            if len(lines) < 5:
                continue
            init_bonds = int(float(lines[1].split()[4])) if len(lines[1].split()) > 4 else 0
            last_bonds = int(float(lines[-1].split()[4])) if len(lines[-1].split()) > 4 else 0
            if init_bonds > 0:
                surv = last_bonds / init_bonds * 100
                flag = "" if surv > 60 else "  ← LOW BOND SURVIVAL"
                print(f"  Bond health [{label}]: {last_bonds}/{init_bonds} = {surv:.1f}%{flag}")
        except Exception:
            pass

    return counts


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60,
                        help="Check interval in minutes (default 60)")
    parser.add_argument("--once", action="store_true",
                        help="Run one check and exit")
    args = parser.parse_args()

    cases = all_expected_cases()

    log(f"Monitor started. {len(cases)} cases to track. "
        f"Interval: {'once' if args.once else str(args.interval) + ' min'}")

    check_num = 0
    while True:
        check_num += 1
        log(f"Check #{check_num}")
        counts = run_check(cases)

        if args.once:
            break

        if counts["done"] == len(cases):
            log("All 24 cases COMPLETE. Monitor exiting.")
            break

        wait_sec = args.interval * 60
        next_check = datetime.now().strftime
        log(f"Next check in {args.interval} min. Sleeping...")
        print(f"  (Press Ctrl+C to stop monitoring)")
        try:
            time.sleep(wait_sec)
        except KeyboardInterrupt:
            log("Monitor stopped by user.")
            break


if __name__ == "__main__":
    main()
