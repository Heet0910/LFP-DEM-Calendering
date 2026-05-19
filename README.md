# LFP-DEM-Calendering

B.Tech Thesis Code — Heet Vinodbhai Patel (22BME103D)
Pandit Deendayal Energy University | Supervisor: Dr. Ramesh Guduru

## Files

| File | Description |
|------|-------------|
| `run_calendering.py` | LAMMPS simulation orchestrator (1,465 lines) |
| `generate_initial_electrode.py` | Stochastic 3-phase electrode geometry generator |
| `bond_damage_energy_model.py` | Bond-damage VED_eff model — original contribution |
| `postproc/` | Post-processing: porosity, pressure, tortuosity, bond stats |

## Simulation Parameters

- Composition: LFP:CB:PTFE = 94:3:3 wt%
- Areal loading: 8.00 and 11.36 mg/cm2
- Compression ratios: 15%, 20%, 25%, 30%
- Calendering speeds: 0.001, 0.003, 0.006 um/us
- Total cases: 24 (PTFE) + 12 (PVDF comparison)
- Simulator: LAMMPS 22 Jul 2025
- Contact: Hertz-Mindlin | Bond: bpm/spring/plastic (Clemmer 2024)

## Key Equations

Geometric VED:
    VED_geom = (1 - eps) x rho_solid x w_AM x C_g x V_avg   [Wh/L]

Bond-Damage-Corrected VED (original contribution):
    VED_eff = eta x VED_geom
    eta = 1 - alpha*f_isolated - beta*(f_damaged - f_isolated)*mean_loss
    alpha = 1.0, beta = 0.40

## Key Results

- Optimal CR: 15-20% at 11.36 mg/cm2
- Peak VED_eff: 940-973 Wh/L with bond survival > 80%
- PTFE vs PVDF: 35-45% wider stable calendering window
- Manufacturing savings: $3.9-6.1M / GWh/yr (dry vs wet route)

## Requirements

    pip install numpy matplotlib pandas scipy

LAMMPS 22 Jul 2025 (64-bit, OpenMP) required for simulations.

## Citation

H. V. Patel, DEM-BPM Simulation of LiFePO4 Battery Electrode Calendering,
B.Tech Thesis, PDEU, Gandhinagar, 2025.