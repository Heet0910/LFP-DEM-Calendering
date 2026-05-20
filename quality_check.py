import json
from pathlib import Path

SIM = Path('simulations')
CRS = [0.15, 0.20, 0.25, 0.30]
LDS = {8.0:'ld8', 11.36:'ld11'}
SPS = {0.001:'sp1', 0.003:'sp3', 0.006:'sp6'}

rows = []

for cr in CRS:
    for ld, ls in LDS.items():
        for sp, ss in SPS.items():
            label = f'dry_cr{int(cr*100)}_{ls}_{ss}'
            matches = list(SIM.rglob(f'{label}/state_comparison.json'))
            if not matches:
                print(f'MISSING: {label}')
                continue
            sc = json.loads(matches[0].read_text())
            states = {s['state']:s for s in sc.get('states',[])}
            fn  = states.get('final_recovered',{})
            nc  = states.get('non_calendered',{})
            csm = sc.get('case_summary',{})

            eps   = float(fn.get('porosity',0))
            sb    = float(csm.get('springback_relative_to_min',0))
            pk_p  = float(csm.get('peak_pressure',0))/1000
            cr_act= float(fn.get('reduction_vs_initial_pct',0))/100
            ved_g = float(fn.get('volumetric_energy_loading_based_Wh_L') or 0)

            # Bond survival
            pl_path = matches[0].parent / 'pressure_log.txt'
            bond_surv = None
            if pl_path.exists():
                lines = [l for l in pl_path.read_text(errors='replace').splitlines()
                         if l.strip() and not l.startswith('#')]
                if len(lines) > 2:
                    try:
                        b0 = float(lines[1].split()[4])
                        b1 = float(lines[-1].split()[4])
                        bond_surv = b1/b0*100 if b0 > 0 else None
                    except Exception:
                        pass

            # Bond-corrected VED
            f_iso_i = float(nc.get('am_zero_bond_fraction',0))
            f_iso_f = float(fn.get('am_zero_bond_fraction',0))
            f_iso_n = max(f_iso_f - f_iso_i, 0)
            f_dmg   = float(fn.get('damaged_am_particle_fraction',0))
            ml      = float(fn.get('mean_am_bond_loss_fraction',0))
            penalty = 1.0*f_iso_n + 0.4*max(f_dmg-f_iso_n,0)*ml
            eta     = max(0.0, 1.0 - penalty)
            ved_eff = ved_g * eta

            # Flag out-of-range values
            flags = []
            if not (0.25 <= eps <= 0.48):    flags.append(f'EPS={eps:.3f}')
            if not (0.04 <= sb <= 0.57):     flags.append(f'SB={sb*100:.1f}pct')
            if pk_p < 0.1:                   flags.append(f'P={pk_p:.3f}MPa_LOW')
            if pk_p > 20.0:                  flags.append(f'P={pk_p:.1f}MPa_HIGH')
            if bond_surv and bond_surv < 60: flags.append(f'BONDS={bond_surv:.1f}pct_CRIT')

            rows.append(dict(
                label=label, cr_t=cr, ld=ld, sp=sp,
                cr_act=cr_act, eps=eps, sb_pct=sb*100,
                pk_p=pk_p, bond_surv=bond_surv,
                ved_g=ved_g, ved_eff=ved_eff, eta=eta,
                flags=flags, status='OK' if not flags else 'WARN'
            ))

SEP = '=' * 94
sep = '-' * 92

print(SEP)
print('  FULL QUALITY REPORT -- 24 cases')
print(SEP)
hdr = (f"  {'Label':<26} {'CRact':>5} {'eps':>6} {'SB%':>5} {'P_MPa':>6} "
       f"{'Bond%':>6} {'VED_g':>6} {'VED_e':>6} {'eta%':>5}  Status")
print(hdr)
print('  ' + sep)

ok = warn = 0
for r in sorted(rows, key=lambda x: (x['ld'], x['sp'], x['cr_t'])):
    bs = f"{r['bond_surv']:.1f}" if r['bond_surv'] is not None else '-'
    fl = ' [' + ', '.join(r['flags']) + ']' if r['flags'] else ''
    line = (f"  {r['label']:<26} {r['cr_act']*100:>4.1f}% "
            f"{r['eps']:>6.4f} {r['sb_pct']:>5.1f} {r['pk_p']:>6.3f} "
            f"{bs:>6} {r['ved_g']:>6.1f} {r['ved_eff']:>6.1f} "
            f"{r['eta']*100:>5.1f}  {r['status']}{fl}")
    print(line)
    if r['status'] == 'OK':
        ok += 1
    else:
        warn += 1

print()
print(f'  RESULT: {ok} PASS  /  {warn} WARN  /  0 FAIL')
print()

all_eps  = [r['eps']      for r in rows]
all_sb   = [r['sb_pct']   for r in rows]
all_pk   = [r['pk_p']     for r in rows]
all_bs   = [r['bond_surv'] for r in rows if r['bond_surv'] is not None]
all_vg   = [r['ved_g']    for r in rows if r['ved_g'] > 0]
all_ve   = [r['ved_eff']  for r in rows if r['ved_eff'] > 0]

print(f'  Porosity range     : {min(all_eps):.4f} to {max(all_eps):.4f}   lit 0.25-0.45')
print(f'  Springback range   : {min(all_sb):.1f}% to {max(all_sb):.1f}%         lit 4-57%')
print(f'  Peak pressure range: {min(all_pk):.3f} to {max(all_pk):.3f} MPa  lit 0.6-15 MPa')
print(f'  Bond survival range: {min(all_bs):.1f}% to {max(all_bs):.1f}%     threshold >60%')
print(f'  VED_geo range      : {min(all_vg):.1f} to {max(all_vg):.1f} Wh/L')
print(f'  VED_eff range      : {min(all_ve):.1f} to {max(all_ve):.1f} Wh/L')
print(SEP)
