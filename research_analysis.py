"""
research_analysis.py — Publication-quality parametric analysis
Generates Fig R1: 6-panel figure (Ngandjong/Meyer/Schreiner style)
"""
import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

SIM = Path('simulations')
OUT = Path('figures_ptfe'); OUT.mkdir(exist_ok=True)
OUTLIERS = {'dry_cr25_ld8_sp3','dry_cr30_ld8_sp1','dry_cr30_ld11_sp3'}
LDS = {8.0:'ld8', 11.36:'ld11'}
SPS = {0.001:'sp1', 0.003:'sp3', 0.006:'sp6'}
NAVY="#0d1e5e"; GOLD="#c8941a"; RED="#c0392b"
COLOR_LD={8.0:"#2166ac", 11.36:"#b2182b"}
MARK_LD={8.0:"o", 11.36:"s"}

matplotlib.rcParams.update({
    "font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,
    "axes.labelsize":9.5,"xtick.labelsize":8.5,"ytick.labelsize":8.5,
    "legend.fontsize":8,"legend.framealpha":0.9,"axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
    "grid.linestyle":"--","grid.linewidth":0.6,"lines.linewidth":1.9,
    "lines.markersize":6,"figure.dpi":300,
})

# ── Load all clean data ───────────────────────────────────────────────────────
data = []
for cr in [0.15,0.20,0.25,0.30]:
    for ld,ls in LDS.items():
        for sp,ss in SPS.items():
            label=f'dry_cr{int(cr*100)}_{ls}_{ss}'
            if label in OUTLIERS: continue
            m=list(SIM.rglob(f'{label}/state_comparison.json'))
            if not m: continue
            sc=json.loads(m[0].read_text())
            states={s['state']:s for s in sc.get('states',[])}
            fn=states.get('final_recovered',{})
            nc=states.get('non_calendered',{})
            mc=states.get('max_compression',{})
            csm=sc.get('case_summary',{})
            eps_nc=float(nc.get('porosity',0))
            eps_mc=float(mc.get('porosity',0))
            eps_fin=float(fn.get('porosity',0))
            ved_g=float(fn.get('volumetric_energy_loading_based_Wh_L') or 0)
            f_iso_i=float(nc.get('am_zero_bond_fraction',0))
            f_iso_f=float(fn.get('am_zero_bond_fraction',0))
            f_dmg=float(fn.get('damaged_am_particle_fraction',0))
            ml=float(fn.get('mean_am_bond_loss_fraction',0))
            penalty=max(f_iso_f-f_iso_i,0)+0.4*max(f_dmg-max(f_iso_f-f_iso_i,0),0)*ml
            eta=max(0.0,1.0-penalty)
            ved_eff=ved_g*eta
            sb=float(csm.get('springback_relative_to_min',0))*100
            pk_p=float(csm.get('peak_pressure',0))/1000
            pl=list(SIM.rglob(f'{label}/pressure_log.txt'))
            bs=None
            if pl:
                lines=[l for l in pl[0].read_text(errors='replace').splitlines()
                       if l.strip() and not l.startswith('#')]
                if len(lines)>2:
                    try:
                        b0=float(lines[1].split()[4]); b1=float(lines[-1].split()[4])
                        bs=b1/b0*100 if b0>0 else None
                    except: pass
            data.append(dict(label=label,cr=cr,ld=ld,sp=sp,
                eps_nc=eps_nc,eps_mc=eps_mc,eps_fin=eps_fin,
                sb=sb,pk_p=pk_p,bs=bs or 0,
                ved_g=ved_g,ved_eff=ved_eff,eta=eta*100,f_dmg=f_dmg))

CRS=[0.15,0.20,0.25,0.30]; cr_pct=[15,20,25,30]

def get_stats(cr,ld,key):
    vals=[d[key] for d in data if d['cr']==cr and d['ld']==ld and d[key]!=0]
    if not vals: return None,None
    return np.mean(vals),np.std(vals)

def powlaw(x,a,n): return a*x**n
def linfit(x,a,b): return a*x+b
def expfit(x,a,b,c): return a*np.exp(b*x)+c

# ── 6-panel figure ────────────────────────────────────────────────────────────
fig,axes=plt.subplots(2,3,figsize=(14,8.5),constrained_layout=True)
axs=axes.flatten()

# Panel A: Porosity vs CR (mean±std across 3 speeds)
ax=axs[0]
for ld in [8.0,11.36]:
    means,stds,xs=[],[],[]
    for cr in CRS:
        m,s=get_stats(cr,ld,'eps_fin')
        if m: means.append(m); stds.append(s); xs.append(cr*100)
    m_arr,s_arr,x_arr=np.array(means),np.array(stds),np.array(xs)
    ax.errorbar(x_arr,m_arr,yerr=s_arr,marker=MARK_LD[ld],color=COLOR_LD[ld],
                lw=2,capsize=4,capthick=1.2,label=f'{ld} mg/cm²',zorder=4)
    ax.fill_between(x_arr,m_arr-s_arr,m_arr+s_arr,alpha=0.10,color=COLOR_LD[ld])
    # Linear fit
    if len(x_arr)>=3:
        try:
            popt,_=curve_fit(linfit,x_arr,m_arr)
            xfit=np.linspace(14,31,50)
            ax.plot(xfit,linfit(xfit,*popt),ls=':',color=COLOR_LD[ld],lw=1.0,alpha=0.7)
        except: pass
ax.axhspan(0.30,0.38,alpha=0.07,color='green')
ax.text(14.2,0.350,'Target\nwindow',fontsize=7,color='darkgreen')
ax.plot([20],[0.373],marker='*',ms=16,color=GOLD,zorder=10,linestyle='None',label='Optimal (CR=20%)')
ax.set_xlabel('Compression Ratio (%)'); ax.set_ylabel('Final Porosity ε (–)')
ax.set_title('(A)  Porosity vs. CR\nmean ± s.d. across 3 speeds',fontweight='bold',color=NAVY)
ax.set_xticks(cr_pct); ax.legend(fontsize=7.5)

# Panel B: Springback vs CR
ax=axs[1]
for ld in [8.0,11.36]:
    means,stds,xs=[],[],[]
    for cr in CRS:
        m,s=get_stats(cr,ld,'sb')
        if m and m>0: means.append(m); stds.append(s); xs.append(cr*100)
    if means:
        m_arr,s_arr,x_arr=np.array(means),np.array(stds),np.array(xs)
        ax.errorbar(x_arr,m_arr,yerr=s_arr,marker=MARK_LD[ld],color=COLOR_LD[ld],
                    lw=2,capsize=4,capthick=1.2,label=f'{ld} mg/cm²')
        ax.fill_between(x_arr,np.clip(m_arr-s_arr,0,None),m_arr+s_arr,
                        alpha=0.10,color=COLOR_LD[ld])
# Literature range
ax.axhspan(4,57,alpha=0.05,color='#555'); ax.text(14.2,50,'Ngandjong 2021\nlit. range',fontsize=6.5,color='#777')
ax.plot([20],[7.3],marker='*',ms=16,color=GOLD,zorder=10,linestyle='None')
ax.set_xlabel('Compression Ratio (%)'); ax.set_ylabel('Springback (%)')
ax.set_title('(B)  Elastic Springback vs. CR\nHigher CR → greater residual springback',fontweight='bold',color=NAVY)
ax.set_xticks(cr_pct); ax.legend(fontsize=7.5)

# Panel C: Peak pressure vs CR — power-law fit
ax=axs[2]
for ld in [8.0,11.36]:
    means,stds,xs=[],[],[]
    for cr in CRS:
        m,s=get_stats(cr,ld,'pk_p')
        if m: means.append(m); stds.append(s); xs.append(cr)
    if means:
        m_arr,s_arr,x_arr=np.array(means),np.array(stds),np.array(xs)
        ax.errorbar(x_arr*100,m_arr,yerr=s_arr,marker=MARK_LD[ld],color=COLOR_LD[ld],
                    lw=2,capsize=4,capthick=1.2,label=f'{ld} mg/cm²')
        try:
            popt,_=curve_fit(powlaw,x_arr,m_arr,p0=[1.0,2.0],maxfev=5000)
            xfit=np.linspace(0.14,0.31,60)
            lbl=f'P={popt[0]:.2f}·CR^{popt[1]:.2f}'
            ax.plot(xfit*100,powlaw(xfit,*popt),ls='--',color=COLOR_LD[ld],lw=1.2,
                    alpha=0.8,label=lbl)
        except: pass
ax.axhspan(0.6,15,alpha=0.04,color='purple')
ax.text(14,13,'Meyer 2018\n0.6–15 MPa',fontsize=6.5,color='purple',va='top')
ax.plot([20],[0.885],marker='*',ms=16,color=GOLD,zorder=10,linestyle='None')
ax.set_xlabel('Compression Ratio (%)'); ax.set_ylabel('Peak Pressure (MPa)')
ax.set_title('(C)  Calendering Pressure vs. CR\nPower-law fit: P = a·CR^n',fontweight='bold',color=NAVY)
ax.set_xticks(cr_pct); ax.legend(fontsize=6.5,ncol=2)

# Panel D: Bond Survival vs CR — exponential decay
ax=axs[3]
for ld in [8.0,11.36]:
    means,stds,xs=[],[],[]
    for cr in CRS:
        m,s=get_stats(cr,ld,'bs')
        if m and m>0: means.append(m); stds.append(s); xs.append(cr)
    if means:
        m_arr,s_arr,x_arr=np.array(means),np.array(stds),np.array(xs)
        ax.errorbar(x_arr*100,m_arr,yerr=s_arr,marker=MARK_LD[ld],color=COLOR_LD[ld],
                    lw=2,capsize=4,capthick=1.2,label=f'{ld} mg/cm²')
        try:
            popt,_=curve_fit(expfit,x_arr,m_arr,p0=[-50,-2,100],maxfev=5000)
            xfit=np.linspace(0.14,0.31,60)
            ax.plot(xfit*100,expfit(xfit,*popt),ls='--',color=COLOR_LD[ld],lw=1.2,alpha=0.8)
        except: pass
ax.axhline(60,color=RED,lw=1.2,ls='--'); ax.text(30.5,61,'Min. 60%',fontsize=7.5,color=RED)
ax.axhline(80,color='green',lw=0.9,ls=':'); ax.text(30.5,81,'Safe 80%',fontsize=7.5,color='darkgreen')
ax.fill_between([13.5,32.5],[80,80],[100,100],alpha=0.05,color='green')
ax.fill_between([13.5,32.5],[60,60],[80,80],alpha=0.05,color='orange')
ax.fill_between([13.5,32.5],[0,0],[60,60],alpha=0.06,color='red')
ax.plot([20],[76.2],marker='*',ms=16,color=GOLD,zorder=10,linestyle='None')
ax.set_xlabel('Compression Ratio (%)'); ax.set_ylabel('Bond Survival (%)')
ax.set_title('(D)  PTFE Bond Network Survival vs. CR\nExponential decay; ε_c=0.60–0.75',fontweight='bold',color=NAVY)
ax.set_xticks(cr_pct); ax.set_ylim(50,110); ax.legend(fontsize=7.5)

# Panel E: VED_geo vs VED_eff — shaded loss band + loading effect
ax=axs[4]
for ld in [8.0,11.36]:
    mg,sg,xe=[],[],[]
    me,se=[],[]
    for cr in CRS:
        mv,sv=get_stats(cr,ld,'ved_g')
        me2,se2=get_stats(cr,ld,'ved_eff')
        if mv and me2:
            mg.append(mv); sg.append(sv); xe.append(cr*100)
            me.append(me2); se.append(se2)
    if mg:
        mg_a,sg_a,me_a,se_a=np.array(mg),np.array(sg),np.array(me),np.array(se)
        x_a=np.array(xe)
        ax.plot(x_a,mg_a,marker=MARK_LD[ld],color=COLOR_LD[ld],lw=2.0,
                ls='--',alpha=0.6,label=f'{ld} mg/cm² VED_geo')
        ax.plot(x_a,me_a,marker=MARK_LD[ld],color=COLOR_LD[ld],lw=2.0,
                label=f'{ld} mg/cm² VED_eff')
        ax.fill_between(x_a,me_a,mg_a,alpha=0.12,color=COLOR_LD[ld],
                        label='Bond-damage loss' if ld==11.36 else '')
ax.plot([20],[953.1],marker='*',ms=16,color=GOLD,zorder=10,linestyle='None',label='Optimal')
ax.set_xlabel('Compression Ratio (%)'); ax.set_ylabel('Volumetric Energy Density (Wh/L)')
ax.set_title('(E)  VED_geo vs VED_eff — Bond-Damage Penalty\nShaded = energy lost to network damage',
             fontweight='bold',color=NAVY)
ax.set_xticks(cr_pct); ax.legend(fontsize=6.5,ncol=2)

# Panel F: Sensitivity — effect of each factor on VED_eff
ax=axs[5]
# Compute effect sizes: range of VED_eff when varying each factor
# Factor 1: CR (fix ld=11.36, sp=baseline, vary CR)
cr_effect=[]
for cr in CRS:
    m,s=get_stats(cr,11.36,'ved_eff')
    if m: cr_effect.append(m)
# Factor 2: Loading (fix cr=0.20, sp=baseline, vary ld)
ld_effect=[]
for ld in [8.0,11.36]:
    m,s=get_stats(0.20,ld,'ved_eff')
    if m: ld_effect.append(m)
# Factor 3: Speed (all speeds at cr=0.20, ld=11.36)
sp_effect=[d['ved_eff'] for d in data if d['cr']==0.20 and d['ld']==11.36]

factors=['CR\n(15-30%)\nfixed ld=11.36','Areal Loading\n(8 vs 11.36 mg/cm²)\nfixed CR=20%','Speed\n(0.001-0.006 um/us)\nfixed CR=20%, ld=11.36']
ranges=[max(cr_effect)-min(cr_effect) if len(cr_effect)>1 else 0,
        max(ld_effect)-min(ld_effect) if len(ld_effect)>1 else 0,
        max(sp_effect)-min(sp_effect) if len(sp_effect)>1 else 0]
colors=['#e08214','#4dac26','#2166ac']
bars=ax.barh(factors,ranges,color=colors,alpha=0.82,edgecolor='white',height=0.5)
for bar,val in zip(bars,ranges):
    ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
            f'{val:.1f} Wh/L',va='center',fontsize=8.5,fontweight='bold')
ax.set_xlabel('Effect Size on VED_eff (Wh/L)\n[max - min across factor levels]')
ax.set_title('(F)  Factor Sensitivity Analysis\nCR has dominant effect on VED_eff',
             fontweight='bold',color=NAVY)
ax.set_xlim(0,max(ranges)*1.35)
ax.invert_yaxis()

fig.suptitle('PTFE Dry Electrode Calendering — Parametric Analysis\n'
             '94:3:3 LFP:CB:PTFE  |  DEM-BPM Simulation  |  21 cases (3 outliers excluded)',
             fontweight='bold',color=NAVY,fontsize=11)

out=OUT/'figR1_full_parametric_analysis.png'
fig.savefig(out,dpi=300,bbox_inches='tight',pad_inches=0.1)
plt.close(fig)
print(f'Saved: {out.name} ({out.stat().st_size//1024} KB)')

# ── Statistical summary table ─────────────────────────────────────────────────
print('\n=== STATISTICAL SUMMARY (mean ± s.d. across 3 speeds) ===')
print(f"{'Case':<18} {'eps_fin':>12} {'SB%':>10} {'P(MPa)':>10} {'Bond%':>10} {'VED_geo':>10} {'VED_eff':>10} {'eta%':>8}")
print('-'*95)
for cr in CRS:
    for ld in [8.0,11.36]:
        metrics=['eps_fin','sb','pk_p','bs','ved_g','ved_eff','eta']
        vals={}
        for k in metrics:
            v=[d[k] for d in data if d['cr']==cr and d['ld']==ld and d[k]!=0]
            vals[k]=(np.mean(v),np.std(v)) if v else (0,0)
        opt_flag=' <-- OPTIMAL' if cr==0.20 and ld==11.36 else ''
        print(f"CR={int(cr*100)}% ld={ld:<5} "
              f"  {vals['eps_fin'][0]:.4f}±{vals['eps_fin'][1]:.4f}"
              f"  {vals['sb'][0]:6.1f}±{vals['sb'][1]:.1f}"
              f"  {vals['pk_p'][0]:6.3f}±{vals['pk_p'][1]:.3f}"
              f"  {vals['bs'][0]:6.1f}±{vals['bs'][1]:.1f}"
              f"  {vals['ved_g'][0]:7.1f}±{vals['ved_g'][1]:.1f}"
              f"  {vals['ved_eff'][0]:7.1f}±{vals['ved_eff'][1]:.1f}"
              f"  {vals['eta'][0]:6.1f}±{vals['eta'][1]:.1f}"
              f"{opt_flag}")

# ── Pearson correlations ──────────────────────────────────────────────────────
print('\n=== PEARSON CORRELATIONS (all 21 cases) ===')
cr_arr=np.array([d['cr'] for d in data])
pairs=[('CR','eps_fin',cr_arr,[d['eps_fin'] for d in data]),
       ('CR','pk_p',cr_arr,[d['pk_p'] for d in data]),
       ('CR','bs',cr_arr,[d['bs'] for d in data]),
       ('CR','ved_eff',cr_arr,[d['ved_eff'] for d in data]),
       ('eps_fin','ved_eff',[d['eps_fin'] for d in data],[d['ved_eff'] for d in data]),
       ('bs','ved_eff',[d['bs'] for d in data],[d['ved_eff'] for d in data])]
for x_name,y_name,x,y in pairs:
    r,p=pearsonr(x,y)
    sig='***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  r({x_name}, {y_name}) = {r:+.3f}  p={p:.4f} {sig}")
print()
print('Done.')
