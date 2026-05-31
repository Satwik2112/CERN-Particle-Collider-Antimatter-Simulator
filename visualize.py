"""
=============================================================================
  CERN LHC Collider Simulation — Complete Visualization Dashboard
=============================================================================
  Produces a comprehensive multi-panel figure covering all major systems:
    1.  Injector chain energy ramp
    2.  FODO cell beta functions (Twiss parameters)
    3.  Betatron phase-space ellipse (emittance)
    4.  Multi-turn betatron oscillations
    5.  Longitudinal phase-space (RF bucket)
    6.  Luminosity vs beam energy
    7.  Luminosity vs β*
    8.  Synchrotron radiation loss vs energy (proton vs electron)
    9.  LHC energy ramp (injection→top energy)
   10.  Quench protection — current decay
   11.  Collimation — beam halo distribution
   12.  Beam–beam tune shift vs bunch population
   13.  PDF xf(x) for u, d, g quarks
   14.  Z→μμ invariant mass (reconstructed)
   15.  Higgs→γγ diphoton mass spectrum
   16.  Dijet pT spectrum
   17.  Particle η distribution
   18.  Detector track pT resolution
   19.  ECAL & HCAL energy resolution
   20.  LHC ring schematic (top-down)
   21.  Beam parameter summary table
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Arc, Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, LogLocator
import warnings
warnings.filterwarnings("ignore")

# Import our simulation modules
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from physics_core import (
    LHCBeam, Emittance, RFCavity, FODOCell, CollimationSystem,
    QuenchProtection, luminosity, synchrotron_radiation_loss,
    beam_beam_tune_shift, gamma_factor, beta_factor,
    beam_rigidity, track_betatron, LHC_CIRCUMFERENCE, LHC_DESIGN_LUM,
    m_proton, m_electron,
    e_charge, c, LHC_DESIGN_ENERGY, LHC_BUNCHES, LHC_BUNCH_POP,
    LHC_NORM_EMITTANCE, LHC_RF_VOLTAGE, LHC_RF_FREQUENCY,
    k_B
)
from detector_sim import (
    simulate_injection_chain, PPEventGenerator,
    ATLASLikeDetector, energy_ramp_lhc,
    pdf_valence_quark
)

# ──────────────────────────────────────────────────────────────────────────────
# Color palette — CERN / LHC aesthetic
# ──────────────────────────────────────────────────────────────────────────────
CERN_BLUE    = '#0033A0'
CERN_RED     = '#D4002D'
LHC_GOLD     = '#F5A623'
ATLAS_ORANGE = '#E85E26'
CMS_GREEN    = '#00A859'
ALICE_PURPLE = '#7B2D8B'
LHCb_TEAL   = '#00B4B4'
DARK_BG      = '#0A0E1A'
GRID_COLOR   = '#1C2340'
TEXT_WHITE   = '#E8EEF7'
TEXT_MUTED   = '#7A8BAA'

# Custom colormap: dark blue → gold
cmap_lhc = LinearSegmentedColormap.from_list(
    'lhc', ['#0A1628', CERN_BLUE, '#4A90D9', LHC_GOLD, CERN_RED])

# ──────────────────────────────────────────────────────────────────────────────
# Global figure style
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   '#0D1220',
    'axes.edgecolor':   '#1C2B50',
    'axes.labelcolor':  TEXT_WHITE,
    'axes.titlecolor':  TEXT_WHITE,
    'axes.grid':        True,
    'grid.color':       GRID_COLOR,
    'grid.linewidth':   0.5,
    'grid.alpha':       0.6,
    'xtick.color':      TEXT_MUTED,
    'ytick.color':      TEXT_MUTED,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'axes.titlesize':   8,
    'axes.labelsize':   8,
    'legend.fontsize':  7,
    'legend.facecolor': '#0D1220',
    'legend.edgecolor': '#1C2B50',
    'text.color':       TEXT_WHITE,
    'font.family':      'monospace',
    'lines.linewidth':  1.4,
    'figure.dpi':       150,
})

def styled_ax(ax, title: str, xlabel: str = '', ylabel: str = ''):
    ax.set_title(title, pad=5, fontsize=8, color=TEXT_WHITE, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1C2B50')
    return ax

# ──────────────────────────────────────────────────────────────────────────────
# Pre-compute simulation data
# ──────────────────────────────────────────────────────────────────────────────
print("  [1/7] Computing beam parameters...")
beam = LHCBeam()
rng = np.random.default_rng(2024)

# --- 1. Injector chain
inj = simulate_injection_chain()

# --- 2. FODO cell
fodo = FODOCell(L_cell=106.9, phase_advance=np.pi/3)
s_fodo, beta_fodo, alpha_fodo = fodo.twiss_along_cell(300)

# --- 3. Emittance
emitt = Emittance(LHC_NORM_EMITTANCE)
x_ell, xp_ell = emitt.phase_ellipse(fodo.beta_max(), 0.0, LHC_DESIGN_ENERGY)
x_ell_inj, xp_ell_inj = emitt.phase_ellipse(fodo.beta_max(), 0.0, 0.45e12)

# --- 4. Betatron oscillations
_, x_bet_7tev = track_betatron(beta_fodo, 200, x0=1e-3, xp0=0.0, tune_frac=0.31)
_, x_bet_450  = track_betatron(beta_fodo, 200, x0=1e-3, xp0=0.0, tune_frac=0.28)

# --- 5. RF longitudinal phase space
print("  [2/7] Simulating RF cavity phase space...")
rf = RFCavity()
phi_arr1, delta_arr1 = rf.bunch_evolution(phi0=0.1,  delta0=0.001, n_turns=3000)
phi_arr2, delta_arr2 = rf.bunch_evolution(phi0=0.3,  delta0=0.002, n_turns=3000)
phi_arr3, delta_arr3 = rf.bunch_evolution(phi0=-0.15, delta0=-0.001, n_turns=3000)

# --- 6. Luminosity scans
E_scan  = np.linspace(1e12, 14e12, 200)
L_scan  = []
for E in E_scan:
    b_tmp = LHCBeam(energy_eV=E)
    L_scan.append(b_tmp.peak_luminosity())
L_scan = np.array(L_scan)

beta_star_scan = np.logspace(-1, 1, 200)  # [m]
L_bs_scan = []
for bs in beta_star_scan:
    b_tmp = LHCBeam(beta_star=bs)
    L_bs_scan.append(b_tmp.peak_luminosity())
L_bs_scan = np.array(L_bs_scan)

# --- 7. Synchrotron radiation
E_arr_p = np.linspace(1e12, 14e12, 200)
E_arr_e = np.linspace(1e9,  200e9, 200)
rho_p = LHC_CIRCUMFERENCE / (2 * np.pi)
rho_e = 3096.175  # LEP radius
U0_p = np.array([synchrotron_radiation_loss(E, rho_p, m_proton) for E in E_arr_p])
U0_e = np.array([synchrotron_radiation_loss(E, rho_e, m_electron) for E in E_arr_e])

# --- 8. Energy ramp
t_ramp, E_ramp = energy_ramp_lhc()

# --- 9. Quench protection
qp = QuenchProtection()
t_q = np.linspace(0, 0.5, 500)
I_decay = qp.current_decay(t_q)

# --- 10. Collimation
coll = CollimationSystem()
n_particles = 50000
beam_dist = rng.standard_normal(n_particles)
beam_dist = np.concatenate([
    beam_dist,
    rng.uniform(4, 12, 500),   # halo particles
])
halo_info = coll.halo_fraction(beam_dist)
survival = coll.scrape(beam_dist)

# --- 11. Beam-beam tune shift
N_scan = np.linspace(1e10, 3e11, 200)
xi_scan = np.array([
    beam_beam_tune_shift(N, 0.55, beam.sigma_IP, beam.sigma_IP, beam.gamma)
    for N in N_scan
])

# --- 12. PDFs
x_pdf = np.linspace(0.001, 0.999, 500)
Q2 = 10000.0   # GeV²  (√Q ≈ 100 GeV)
f_u = np.array([pdf_valence_quark(x, Q2, 'u') for x in x_pdf])
f_d = np.array([pdf_valence_quark(x, Q2, 'd') for x in x_pdf])
f_g = np.array([pdf_valence_quark(x, Q2, 'g') for x in x_pdf])
f_s = np.array([pdf_valence_quark(x, Q2, 's') for x in x_pdf])

# --- 13. MC events
print("  [3/7] Generating Monte Carlo events...")
gen = PPEventGenerator(sqrt_s_TeV=14.0, seed=42)
det = ATLASLikeDetector()
det_rng = np.random.default_rng(999)

Z_events   = gen.generate_events(3000, 'Z')
Z_masses   = det.reconstruct_Z_mass(Z_events, det_rng)

H_events   = gen.generate_events(2000, 'Higgs')
H_masses   = []
for evt in H_events:
    if len(evt) < 2: continue
    p1s = det.smear_particle(evt[0], det_rng)
    p2s = det.smear_particle(evt[1], det_rng)
    if det.is_accepted(p1s) and det.is_accepted(p2s):
        m2 = ((p1s.E+p2s.E)**2 - (p1s.px+p2s.px)**2 -
               (p1s.py+p2s.py)**2 - (p1s.pz+p2s.pz)**2)
        if m2 > 0:
            H_masses.append(np.sqrt(m2))
H_masses = np.array(H_masses)

jet_events = gen.generate_events(5000, 'dijet')
jet_pT     = np.array([max(p.pt for p in evt) for evt in jet_events if evt])
jet_eta    = np.array([p.eta for evt in jet_events for p in evt if evt])

# --- 14. Detector resolutions
pT_res_arr = np.logspace(-1, 2.5, 200)
dpt_track  = np.array([det.pT_resolution(pT) * 100 for pT in pT_res_arr])

E_res_arr  = np.logspace(-1, 3, 200)
dE_ecal   = np.array([det.ecal_energy_resolution(E) * 100 for E in E_res_arr])
dE_hcal   = np.array([det.hcal_energy_resolution(E) * 100 for E in E_res_arr])

print("  [4/7] Data computed. Building figure...")

# ──────────────────────────────────────────────────────────────────────────────
# MASTER FIGURE
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 38), facecolor=DARK_BG)

# Title banner
fig.text(0.5, 0.988,
         "CERN  LHC  PARTICLE  COLLIDER  SIMULATION",
         ha='center', va='top', fontsize=22, fontweight='bold',
         color=LHC_GOLD, fontfamily='monospace',
         transform=fig.transFigure)
fig.text(0.5, 0.981,
         "Complete Physics Simulation  ·  Beam Dynamics · Emittance · RF · Synchrotron Radiation · Magnets · "
         "Collimation · Quench Protection · Monte Carlo · Detector",
         ha='center', va='top', fontsize=9, color=TEXT_MUTED,
         transform=fig.transFigure)

# ── GridSpec layout ─────────────────────────────────────────────────────────
gs = gridspec.GridSpec(7, 4, figure=fig,
                        hspace=0.52, wspace=0.38,
                        top=0.975, bottom=0.02,
                        left=0.055, right=0.975)

# ═══════════════════════════════════════════════════════════════════════
# ROW 0  — Injector chain  |  FODO beta  |  Phase ellipse  |  LHC ring
# ═══════════════════════════════════════════════════════════════════════
ax_inj  = fig.add_subplot(gs[0, 0])
ax_fodo = fig.add_subplot(gs[0, 1])
ax_ell  = fig.add_subplot(gs[0, 2])
ax_ring = fig.add_subplot(gs[0, 3])

# ── Plot 1: Injector chain ───────────────────────────────────────────
stages = list(inj.keys())
energies = [inj[s]['energy_TeV'] for s in stages]
colors_inj = [CERN_BLUE, '#4A90D9', CMS_GREEN, ATLAS_ORANGE, CERN_RED]
bars = ax_inj.bar(stages, energies, color=colors_inj,
                   edgecolor='white', linewidth=0.5, width=0.65)
ax_inj.set_yscale('log')
for bar, E in zip(bars, energies):
    ax_inj.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.4,
                f'{E:.3g} TeV' if E >= 0.001 else f'{E*1e6:.0f} keV',
                ha='center', fontsize=5.5, color=TEXT_WHITE, rotation=0)
styled_ax(ax_inj, '①  INJECTOR CHAIN  (Source → LHC)',
           'Accelerator Stage', 'Beam Energy [TeV]')
ax_inj.tick_params(axis='x', labelsize=6, rotation=20)

# ── Plot 2: FODO Twiss ───────────────────────────────────────────────
ax_fodo2 = ax_fodo.twinx()
ax_fodo.plot(s_fodo, beta_fodo, color=CERN_BLUE, lw=1.8, label='β(s) [m]')
ax_fodo2.plot(s_fodo, alpha_fodo, color=LHC_GOLD, lw=1.3, ls='--', label='α(s)')
ax_fodo.fill_between(s_fodo, beta_fodo, alpha=0.15, color=CERN_BLUE)
ax_fodo.axhline(fodo.beta_max(), color=CERN_RED, ls=':', lw=1, alpha=0.7,
                label=f'β_max={fodo.beta_max():.1f} m')
ax_fodo.axhline(fodo.beta_min(), color=CMS_GREEN, ls=':', lw=1, alpha=0.7,
                label=f'β_min={fodo.beta_min():.1f} m')
ax_fodo.set_xlabel('s [m]', fontsize=7.5)
ax_fodo.set_ylabel('β [m]', fontsize=7.5, color=CERN_BLUE)
ax_fodo2.set_ylabel('α', fontsize=7.5, color=LHC_GOLD)
ax_fodo2.tick_params(colors=LHC_GOLD, labelsize=6)
lines1, labels1 = ax_fodo.get_legend_handles_labels()
lines2, labels2 = ax_fodo2.get_legend_handles_labels()
ax_fodo.legend(lines1 + lines2, labels1 + labels2, fontsize=5.5, loc='upper right')
styled_ax(ax_fodo, '②  FODO CELL  Twiss Parameters (β, α)')
ax_fodo.set_title('②  FODO CELL  Twiss Parameters (β, α)',
                   pad=5, fontsize=8, color=TEXT_WHITE, fontweight='bold')

# ── Plot 3: Phase-space ellipse ──────────────────────────────────────
ax_ell.plot(x_ell * 1e3, xp_ell * 1e3, color=CERN_BLUE, lw=1.8,
             label=f'7 TeV  εₙ={LHC_NORM_EMITTANCE*1e6:.2f} μm·rad')
ax_ell.fill(x_ell * 1e3, xp_ell * 1e3, alpha=0.15, color=CERN_BLUE)
ax_ell.plot(x_ell_inj * 1e3, xp_ell_inj * 1e3, color=LHC_GOLD, lw=1.3, ls='--',
             label='450 GeV injection')
ax_ell.fill(x_ell_inj * 1e3, xp_ell_inj * 1e3, alpha=0.08, color=LHC_GOLD)
ax_ell.axhline(0, color='white', lw=0.4, alpha=0.3)
ax_ell.axvline(0, color='white', lw=0.4, alpha=0.3)
ax_ell.legend(fontsize=5.5)
ax_ell.set_aspect('auto')
styled_ax(ax_ell, '③  TRANSVERSE PHASE-SPACE ELLIPSE  (Emittance)',
           'x  [mm]', "x'  [mrad]")

# ── Plot 4: LHC ring schematic ───────────────────────────────────────
ax_ring.set_facecolor('#080C18')
ax_ring.set_aspect('equal')
theta = np.linspace(0, 2*np.pi, 500)
R_ring = 1.0   # normalized
# Main ring
ax_ring.plot(R_ring*np.cos(theta), R_ring*np.sin(theta),
             color=CERN_BLUE, lw=2.5, zorder=5)
ax_ring.plot(0.97*np.cos(theta), 0.97*np.sin(theta),
             color='#0A2060', lw=1.5, ls='--', alpha=0.4, zorder=4)
# 8 arcs of dipoles (blue) alternating with straight sections (gold)
n_arcs = 8
for i in range(n_arcs):
    ang0 = (i * 2*np.pi/n_arcs)
    ang1 = ang0 + 0.85 * 2*np.pi/n_arcs
    th_arc = np.linspace(ang0, ang1, 60)
    ax_ring.plot(R_ring*1.03*np.cos(th_arc), R_ring*1.03*np.sin(th_arc),
                 color=CERN_BLUE, lw=3, alpha=0.5, zorder=3)
# 4 interaction points (ATLAS, CMS, ALICE, LHCb)
IPs = [
    ('ATLAS', 0,          CERN_RED),
    ('CMS',   np.pi,      CMS_GREEN),
    ('ALICE', np.pi/2,    ALICE_PURPLE),
    ('LHCb',  3*np.pi/2,  LHCb_TEAL),
]
for name, ang, col in IPs:
    x, y = R_ring * np.cos(ang), R_ring * np.sin(ang)
    ax_ring.scatter([x], [y], s=120, color=col, zorder=10, edgecolors='white',
                    linewidths=0.8)
    offset = 0.18
    ax_ring.text(x * (1 + offset), y * (1 + offset), name,
                 ha='center', va='center', fontsize=6.5, color=col,
                 fontweight='bold', zorder=11)
    # Detector glow
    circ = Circle((x, y), 0.08, color=col, alpha=0.18, zorder=6)
    ax_ring.add_patch(circ)
# Injection arrow
ax_ring.annotate('SPS\ninjection', xy=(0, 1.0), xytext=(0.4, 1.35),
                  fontsize=5.5, color=LHC_GOLD,
                  arrowprops=dict(arrowstyle='->', color=LHC_GOLD, lw=1.0))
ax_ring.text(0, 0, 'LHC\n27 km\n7 TeV/beam', ha='center', va='center',
              fontsize=7, color=TEXT_MUTED)
ax_ring.set_xlim(-1.55, 1.55)
ax_ring.set_ylim(-1.55, 1.55)
ax_ring.set_xticks([]); ax_ring.set_yticks([])
ax_ring.set_title('④  LHC RING  Schematic Layout  (4 Experiments)',
                   pad=5, fontsize=8, color=TEXT_WHITE, fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════
# ROW 1  — Betatron oscill.  |  RF phase space  |  Luminosity vs E
# ═══════════════════════════════════════════════════════════════════════
ax_bet   = fig.add_subplot(gs[1, 0])
ax_rf    = fig.add_subplot(gs[1, 1])
ax_lum_E = fig.add_subplot(gs[1, 2])
ax_lum_b = fig.add_subplot(gs[1, 3])

# ── Plot 5: Betatron oscillations ────────────────────────────────────
n_t = 200
turns = np.arange(n_t)
ax_bet.plot(turns, x_bet_7tev * 1e3, color=CERN_BLUE, lw=1.2, label='7 TeV  Q=0.31')
ax_bet.plot(turns, x_bet_450  * 1e3, color=LHC_GOLD,  lw=1.2, ls='--', label='450 GeV Q=0.28')
ax_bet.axhline(0, color='white', lw=0.3, alpha=0.3)
ax_bet.legend(fontsize=6)
styled_ax(ax_bet, '⑤  BETATRON OSCILLATIONS  (Multi-turn)',
           'Turn Number', 'x  [mm]')

# ── Plot 6: RF longitudinal phase space ──────────────────────────────
# Draw RF bucket separatrix
phi_bucket = np.linspace(-np.pi, np.pi, 400)
# Simplified separatrix: delta = ± sqrt((1-cos φ + φ sin φ_s)/...)
delta_sep = 0.005 * np.sqrt(np.abs(np.cos(phi_bucket) - np.cos(0.5) +
                                    (phi_bucket - 0.5) * np.sin(0.5)))
ax_rf.fill_between(phi_bucket, -delta_sep, delta_sep, alpha=0.08, color=CMS_GREEN)
ax_rf.plot(phi_bucket, delta_sep, color=CMS_GREEN, lw=0.8, ls=':', alpha=0.5)
ax_rf.plot(phi_bucket, -delta_sep, color=CMS_GREEN, lw=0.8, ls=':', alpha=0.5)
# Particle trajectories (downsample for plotting)
step = 5
ax_rf.scatter(phi_arr1[::step], delta_arr1[::step], s=0.3, c=CERN_BLUE, alpha=0.4)
ax_rf.scatter(phi_arr2[::step], delta_arr2[::step], s=0.3, c=LHC_GOLD,  alpha=0.4)
ax_rf.scatter(phi_arr3[::step], delta_arr3[::step], s=0.3, c=CERN_RED,  alpha=0.4)
ax_rf.axvline(0, color='white', lw=0.4, alpha=0.3)
ax_rf.set_xlim(-0.6, 0.6); ax_rf.set_ylim(-0.004, 0.004)
styled_ax(ax_rf, '⑥  RF LONGITUDINAL PHASE SPACE  (400 MHz, 16 MV)',
           'RF Phase φ [rad]', 'δE/E')

# ── Plot 7: Luminosity vs Energy ─────────────────────────────────────
ax_lum_E.plot(E_scan / 1e12, L_scan / 1e34,
               color=LHC_GOLD, lw=2.0)
ax_lum_E.fill_between(E_scan / 1e12, L_scan / 1e34, alpha=0.15, color=LHC_GOLD)
ax_lum_E.axvline(7.0, color=CERN_RED, ls='--', lw=1.0, label='7 TeV (Run 2)')
ax_lum_E.axvline(13.6, color=CMS_GREEN, ls='--', lw=1.0, label='13.6 TeV (Run 3)')
ax_lum_E.axhline(1.0, color=TEXT_MUTED, ls=':', lw=0.8, label='Design L')
ax_lum_E.legend(fontsize=5.5)
styled_ax(ax_lum_E, '⑦  PEAK LUMINOSITY  vs Beam Energy',
           'Beam Energy [TeV]', 'L  [×10³⁴ cm⁻²s⁻¹]')

# ── Plot 8: Luminosity vs β* ─────────────────────────────────────────
ax_lum_b.semilogx(beta_star_scan, L_bs_scan / 1e34,
                   color=CERN_BLUE, lw=2.0)
ax_lum_b.fill_between(beta_star_scan, L_bs_scan / 1e34, alpha=0.12, color=CERN_BLUE)
ax_lum_b.axvline(0.55, color=LHC_GOLD, ls='--', lw=1.0, label='β*=0.55 m (design)')
ax_lum_b.axvline(0.15, color=CERN_RED, ls='--', lw=1.0, label='β*=0.15 m (HL-LHC)')
ax_lum_b.legend(fontsize=5.5)
styled_ax(ax_lum_b, '⑧  PEAK LUMINOSITY  vs β* at IP',
           'β* [m]', 'L  [×10³⁴ cm⁻²s⁻¹]')

# ═══════════════════════════════════════════════════════════════════════
# ROW 2  — Synchrotron rad  |  Energy ramp  |  Quench protection
# ═══════════════════════════════════════════════════════════════════════
ax_sr   = fig.add_subplot(gs[2, 0])
ax_ramp = fig.add_subplot(gs[2, 1])
ax_qp   = fig.add_subplot(gs[2, 2])
ax_coll = fig.add_subplot(gs[2, 3])

# ── Plot 9: Synchrotron radiation loss ───────────────────────────────
ax_sr2 = ax_sr.twinx()
ax_sr.semilogy(E_arr_p / 1e12, U0_p / 1e3, color=CERN_BLUE, lw=1.8,
                label='Protons (LHC, ρ=2.8 km)')
ax_sr2.semilogy(E_arr_e / 1e9, U0_e / 1e9, color=CERN_RED, lw=1.5, ls='--',
                 label='Electrons (LEP, ρ=3.1 km)')
ax_sr.axvline(7.0, color=LHC_GOLD, ls=':', lw=1, alpha=0.8, label='LHC 7 TeV: ~7 keV/turn')
ax_sr.set_xlabel('Energy [TeV]  (protons)', fontsize=7.5)
ax_sr.set_ylabel('U₀ [keV/turn]  (proton)', fontsize=7.5, color=CERN_BLUE)
ax_sr2.set_ylabel('U₀ [GeV/turn]  (electron)', fontsize=7.5, color=CERN_RED)
ax_sr2.tick_params(colors=CERN_RED, labelsize=6)
lines_a, lab_a = ax_sr.get_legend_handles_labels()
lines_b, lab_b = ax_sr2.get_legend_handles_labels()
ax_sr.legend(lines_a + lines_b, lab_a + lab_b, fontsize=5.5)
ax_sr.set_title('⑨  SYNCHROTRON RADIATION  U₀ vs Energy',
                 pad=5, fontsize=8, color=TEXT_WHITE, fontweight='bold')

# ── Plot 10: LHC energy ramp ─────────────────────────────────────────
ax_ramp.plot(t_ramp / 60, E_ramp, color=LHC_GOLD, lw=2.0)
ax_ramp.fill_between(t_ramp / 60, E_ramp, alpha=0.12, color=LHC_GOLD)
ax_ramp.axhline(0.45, color=CERN_BLUE, ls='--', lw=1, alpha=0.8, label='Injection 450 GeV')
ax_ramp.axhline(7.0,  color=CERN_RED,  ls='--', lw=1, alpha=0.8, label='Top energy 7 TeV')
ax_ramp.legend(fontsize=5.5)
styled_ax(ax_ramp, '⑩  LHC ENERGY RAMP  (~20 min, sinusoidal)',
           'Time [min]', 'Beam Energy [TeV]')

# ── Plot 11: Quench protection ───────────────────────────────────────
ax_qp.plot(t_q * 1e3, I_decay / 1e3, color=CERN_RED, lw=2.0, label='I(t) decay')
ax_qp.axhline(qp.I_nominal / 1e3, color=LHC_GOLD, ls='--', lw=1, alpha=0.8,
               label=f'I₀ = {qp.I_nominal/1e3:.1f} kA')
ax_qp.axhline(0, color='white', lw=0.3, alpha=0.3)
ax_qp.fill_between(t_q * 1e3, I_decay / 1e3, alpha=0.15, color=CERN_RED)
ax_qp.text(300, qp.I_nominal / 2e3,
            f'E_stored = {qp.stored_energy() / 1e3:.1f} kJ\n'
            f'τ = {qp.L_magnet/qp.R_quench*1e3:.0f} ms',
            fontsize=6, color=TEXT_WHITE,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A0A20', alpha=0.8))
ax_qp.legend(fontsize=5.5)
styled_ax(ax_qp, '⑪  QUENCH PROTECTION  Current Decay',
           'Time [ms]', 'Current [kA]')

# ── Plot 12: Collimation halo ─────────────────────────────────────────
sigma_bins = np.linspace(-15, 15, 120)
hist_all,  bins = np.histogram(beam_dist, bins=sigma_bins, density=True)
hist_surv, _   = np.histogram(beam_dist[survival], bins=sigma_bins, density=True)
bin_c = 0.5 * (bins[:-1] + bins[1:])
ax_coll.semilogy(bin_c, hist_all + 1e-10, color=TEXT_MUTED, lw=1.0,
                  label='All particles', alpha=0.7)
ax_coll.semilogy(bin_c, hist_surv + 1e-10, color=CERN_BLUE, lw=1.8,
                  label='After collimation (< 10σ)')
ax_coll.axvline( 6, color=CERN_RED, ls='--', lw=1, label='Primary jaw 6σ')
ax_coll.axvline(-6, color=CERN_RED, ls='--', lw=1)
ax_coll.axvline( 7, color=LHC_GOLD, ls=':', lw=1, label='Secondary 7σ')
ax_coll.axvline(-7, color=LHC_GOLD, ls=':', lw=1)
ax_coll.legend(fontsize=5.5)
styled_ax(ax_coll, '⑫  COLLIMATION  Beam Halo Distribution',
           'x / σ  (normalized amplitude)', 'Density')

# ═══════════════════════════════════════════════════════════════════════
# ROW 3  — Beam-beam  |  PDFs  |  Z mass  |  Higgs diphoton
# ═══════════════════════════════════════════════════════════════════════
ax_bb   = fig.add_subplot(gs[3, 0])
ax_pdf  = fig.add_subplot(gs[3, 1])
ax_Zmass = fig.add_subplot(gs[3, 2])
ax_Hmass = fig.add_subplot(gs[3, 3])

# ── Plot 13: Beam-beam tune shift ─────────────────────────────────────
ax_bb.plot(N_scan / 1e11, xi_scan * 1e3, color=ALICE_PURPLE, lw=2.0)
ax_bb.fill_between(N_scan / 1e11, xi_scan * 1e3, alpha=0.15, color=ALICE_PURPLE)
ax_bb.axvline(LHC_BUNCH_POP / 1e11, color=LHC_GOLD, ls='--', lw=1.0,
               label=f'LHC design N={LHC_BUNCH_POP/1e11:.2f}×10¹¹')
ax_bb.axhline(5.0, color=CERN_RED, ls=':', lw=1.0, label='Stability limit ξ≈0.005')
ax_bb.legend(fontsize=5.5)
styled_ax(ax_bb, '⑬  BEAM–BEAM TUNE SHIFT  ξ vs N_p',
           'Bunch Population [×10¹¹]', 'ξ  [×10⁻³ / IP]')

# ── Plot 14: PDFs ─────────────────────────────────────────────────────
ax_pdf.plot(x_pdf, f_u, color=CERN_RED,     lw=1.8, label='u quark')
ax_pdf.plot(x_pdf, f_d, color=CERN_BLUE,    lw=1.8, label='d quark')
ax_pdf.plot(x_pdf, f_g / 4, color=CMS_GREEN, lw=1.5, ls='--', label='gluon  (÷4)')
ax_pdf.plot(x_pdf, f_s * 5, color=LHC_GOLD,  lw=1.2, ls=':', label='sea  (×5)')
ax_pdf.set_xlim(0, 1); ax_pdf.set_ylim(0, 1.5)
ax_pdf.legend(fontsize=5.5, loc='upper right')
styled_ax(ax_pdf, f'⑭  PARTON DISTRIBUTION FUNCTIONS  xf(x, Q²={int(Q2)} GeV²)',
           'Momentum fraction x', 'xf(x,Q²)')

# ── Plot 15: Z→μμ invariant mass ────────────────────────────────────
if len(Z_masses) > 10:
    z_bins = np.linspace(40, 140, 60)
    counts, bins_z = np.histogram(Z_masses, bins=z_bins)
    bin_c_z = 0.5 * (bins_z[:-1] + bins_z[1:])
    ax_Zmass.bar(bin_c_z, counts, width=(z_bins[1]-z_bins[0])*0.9,
                  color=CERN_BLUE, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax_Zmass.axvline(91.2, color=CERN_RED, lw=1.5, ls='--', label='M_Z = 91.2 GeV')
    peak_m = bin_c_z[np.argmax(counts)]
    ax_Zmass.text(91.5, counts.max() * 0.85,
                   f'Peak: {peak_m:.1f} GeV\nN={len(Z_masses)}',
                   fontsize=6, color=TEXT_WHITE)
ax_Zmass.legend(fontsize=6)
styled_ax(ax_Zmass, '⑮  Z → μ⁺μ⁻  Invariant Mass  (ATLAS-like detector)',
           'm_μμ  [GeV]', 'Events / 1.7 GeV')

# ── Plot 16: H→γγ mass ───────────────────────────────────────────────
if len(H_masses) > 10:
    h_bins = np.linspace(100, 160, 50)
    hcounts, hbins = np.histogram(H_masses, bins=h_bins)
    hbin_c = 0.5 * (hbins[:-1] + hbins[1:])
    # Smooth background continuum (exponential)
    bg = 3 * np.exp(-0.05 * (hbin_c - 100))
    ax_Hmass.bar(hbin_c, hcounts, width=(h_bins[1]-h_bins[0])*0.9,
                  color=ATLAS_ORANGE, alpha=0.8, edgecolor='white', linewidth=0.3,
                  label='H→γγ signal')
    ax_Hmass.plot(hbin_c, bg, color=TEXT_MUTED, lw=1.0, ls='--', label='Continuum γγ')
    ax_Hmass.axvline(125.1, color=LHC_GOLD, lw=1.5, ls='--', label='M_H = 125.1 GeV')
ax_Hmass.legend(fontsize=5.5)
styled_ax(ax_Hmass, '⑯  H → γγ  Diphoton Mass  (Higgs discovery channel)',
           'm_γγ  [GeV]', 'Events / 1.2 GeV')

# ═══════════════════════════════════════════════════════════════════════
# ROW 4  — Dijet pT  |  η distribution  |  Track resolution  |  Calo res
# ═══════════════════════════════════════════════════════════════════════
ax_jet  = fig.add_subplot(gs[4, 0])
ax_eta  = fig.add_subplot(gs[4, 1])
ax_trk  = fig.add_subplot(gs[4, 2])
ax_cal  = fig.add_subplot(gs[4, 3])

# ── Plot 17: Dijet pT spectrum ───────────────────────────────────────
if len(jet_pT) > 5:
    pT_bins = np.logspace(1, 3.5, 50)
    jcounts, jbins = np.histogram(jet_pT, bins=pT_bins)
    jbin_c = np.sqrt(jbins[:-1] * jbins[1:])
    ax_jet.loglog(jbin_c, jcounts / (jbins[1:]-jbins[:-1]),
                   color=CMS_GREEN, lw=1.8, marker='o', ms=2)
    ax_jet.fill_between(jbin_c, jcounts / (jbins[1:]-jbins[:-1]),
                         alpha=0.12, color=CMS_GREEN)
styled_ax(ax_jet, '⑰  QCD DIJET  pT Spectrum  (QCD 2→2)',
           'Jet pT  [GeV]', 'dN/dpT  [GeV⁻¹]')

# ── Plot 18: Particle η distribution ────────────────────────────────
if len(jet_eta) > 5:
    eta_bins = np.linspace(-6, 6, 60)
    ecounts, ebins = np.histogram(jet_eta, bins=eta_bins)
    ebin_c = 0.5 * (ebins[:-1] + ebins[1:])
    ax_eta.bar(ebin_c, ecounts, width=(eta_bins[1]-eta_bins[0])*0.9,
                color=LHCb_TEAL, alpha=0.8, edgecolor='none')
    ax_eta.axvline( 2.5, color=CERN_RED, ls='--', lw=1, label='|η|<2.5 tracker')
    ax_eta.axvline(-2.5, color=CERN_RED, ls='--', lw=1)
    ax_eta.axvline( 4.9, color=LHC_GOLD, ls=':', lw=1, label='|η|<4.9 forward cal')
    ax_eta.axvline(-4.9, color=LHC_GOLD, ls=':', lw=1)
    ax_eta.legend(fontsize=5.5)
styled_ax(ax_eta, '⑱  PARTICLE η DISTRIBUTION  (QCD dijets)',
           'Pseudorapidity η', 'Events / 0.2')

# ── Plot 19: Tracker pT resolution ───────────────────────────────────
ax_trk.semilogx(pT_res_arr, dpt_track, color=CERN_BLUE, lw=2.0, label='ATLAS inner tracker (B=2T)')
ax_trk.axhline(1.0, color=TEXT_MUTED, ls=':', lw=0.8, alpha=0.6, label='1% threshold')
ax_trk.set_ylim(0, 15)
ax_trk.legend(fontsize=5.5)
styled_ax(ax_trk, '⑲  TRACK pT RESOLUTION  σ(pT)/pT vs pT',
           'pT  [GeV]', 'σ(pT)/pT  [%]')

# ── Plot 20: Calorimeter energy resolution ───────────────────────────
ax_cal.loglog(E_res_arr, dE_ecal, color=CERN_RED,  lw=1.8, label='ECAL  (EM, a=10%)')
ax_cal.loglog(E_res_arr, dE_hcal, color=LHC_GOLD, lw=1.8, ls='--', label='HCAL (Had, a=50%)')
ax_cal.axhline(1.0, color=TEXT_MUTED, ls=':', lw=0.8, alpha=0.6)
ax_cal.set_ylim(0.1, 200)
ax_cal.legend(fontsize=6)
styled_ax(ax_cal, '⑳  CALORIMETER ENERGY RESOLUTION  σ(E)/E vs E',
           'Energy  [GeV]', 'σ(E)/E  [%]')

# ═══════════════════════════════════════════════════════════════════════
# ROW 5  — Dipole field model  |  Luminosity evolution  |  Parameter table (wide)
# ═══════════════════════════════════════════════════════════════════════
ax_dipole = fig.add_subplot(gs[5, 0])
ax_lumev  = fig.add_subplot(gs[5, 1])
ax_table  = fig.add_subplot(gs[5, 2:])

# ── Plot 21: Dipole bending field vs energy ───────────────────────────
E_dip  = np.linspace(0.45e12, 7.0e12, 300)
Brho_arr = np.array([beam_rigidity(E, m_proton) for E in E_dip])
B_dip = Brho_arr / 2803.95   # rho = C/(2π) for LHC
ax_dipole.plot(E_dip / 1e12, B_dip, color=CERN_BLUE, lw=2.0)
ax_dipole.fill_between(E_dip / 1e12, B_dip, alpha=0.12, color=CERN_BLUE)
ax_dipole.axhline(8.33, color=CERN_RED, ls='--', lw=1, label='B_max = 8.33 T (NbTi limit)')
ax_dipole.axhline(1.9,  color=LHC_GOLD, ls=':', lw=1, label='Injection ~0.54 T')
ax_dipole.legend(fontsize=5.5)
styled_ax(ax_dipole, '㉑  DIPOLE FIELD  B vs Beam Energy',
           'Beam Energy [TeV]', 'Dipole B  [T]')

# ── Plot 22: Luminosity fill evolution (decay model) ─────────────────
t_fill = np.linspace(0, 12, 300)   # hours
tau_burn = 15.0   # hours (burn-off + emittance growth)
L0 = LHC_DESIGN_LUM
L_fill = L0 * np.exp(-t_fill / tau_burn)
# Also model emittance growth
eps_growth = LHC_NORM_EMITTANCE * (1 + 0.05 * t_fill)
beam_tmp_arr = []
for i, t in enumerate(t_fill):
    b_t = LHCBeam(norm_emittance=eps_growth[i],
                   bunch_population=LHC_BUNCH_POP * np.exp(-t/tau_burn))
    beam_tmp_arr.append(b_t.peak_luminosity())
L_fill2 = np.array(beam_tmp_arr)
ax_lumev.plot(t_fill, L_fill / 1e34, color=TEXT_MUTED, lw=1.0,
               ls='--', label='Simple exp. decay')
ax_lumev.plot(t_fill, L_fill2 / 1e34, color=LHC_GOLD, lw=2.0,
               label='Burn-off + emittance growth')
ax_lumev.fill_between(t_fill, L_fill2 / 1e34, alpha=0.12, color=LHC_GOLD)
ax_lumev.axhline(1.0, color=CERN_RED, ls=':', lw=0.8, label='Design L')
ax_lumev.legend(fontsize=5.5)
styled_ax(ax_lumev, '㉒  LUMINOSITY FILL EVOLUTION  (Burn-off model)',
           'Time in Fill [hours]', 'L  [×10³⁴ cm⁻²s⁻¹]')

# ── Table: LHC parameter summary ─────────────────────────────────────
ax_table.axis('off')
summary = beam.summary()
table_data = [[k, f'{v:.4g}'] for k, v in summary.items()]
table_data += [
    ['Bending B [T]',          f'{8.33}'],
    ['Dipole count',           f'{1232}'],
    ['RF frequency [MHz]',     f'{400}'],
    ['RF voltage [MV/beam]',   f'{16}'],
    ['Cryo temp [K]',          f'{1.9}'],
    ['Vacuum [mbar]',          f'~10⁻¹⁰–10⁻¹¹'],
    ['BPM count',              f'1070'],
    ['BLM count',              f'~4000'],
    ['Stored E / beam [MJ]',   f'{362}'],
    ['Circumference [km]',     f'{LHC_CIRCUMFERENCE/1e3:.2f}'],
]
col_labels = ['Parameter', 'Value']
tbl = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='left',
    bbox=[0.0, 0.0, 1.0, 1.0],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1A2A60')
        cell.set_text_props(color=LHC_GOLD, fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#0D1830')
        cell.set_text_props(color=TEXT_WHITE)
    else:
        cell.set_facecolor('#101A2E')
        cell.set_text_props(color=TEXT_WHITE)
    cell.set_edgecolor('#1C2B50')
ax_table.set_title('㉓  LHC DESIGN PARAMETERS  (Summary Table)',
                    pad=10, fontsize=8, color=TEXT_WHITE, fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════
# ROW 6  — Magnet cross section  |  Beam size vs energy  |  Momentum spread
# ═══════════════════════════════════════════════════════════════════════
ax_mag  = fig.add_subplot(gs[6, 0])
ax_beamsz = fig.add_subplot(gs[6, 1])
ax_mom  = fig.add_subplot(gs[6, 2])
ax_stat = fig.add_subplot(gs[6, 3])

# ── Plot 24: Dipole cross-section schematic ───────────────────────────
ax_mag.set_aspect('equal')
ax_mag.set_facecolor('#060A14')
# Yoke (outer iron)
yoke = plt.Circle((0, 0), 1.0, color='#2A3060', zorder=1)
ax_mag.add_patch(yoke)
# Two apertures (twin-bore LHC)
for dx in [-0.4, 0.4]:
    bore = plt.Circle((dx, 0), 0.32, color='#0A0E1A', zorder=2)
    coil = plt.Circle((dx, 0), 0.30, color=CERN_BLUE, zorder=3, alpha=0.7)
    beam_pipe = plt.Circle((dx, 0), 0.18, color='#080C14', zorder=4)
    beam = plt.Circle((dx, 0), 0.05, color=LHC_GOLD, zorder=5, alpha=0.9)
    ax_mag.add_patch(bore)
    ax_mag.add_patch(coil)
    ax_mag.add_patch(beam_pipe)
    ax_mag.add_patch(beam)
    # Field arrows
    for angle in np.linspace(0.3, 1.2, 4):
        ax_mag.annotate('', xy=(dx + 0.22*np.cos(angle), 0.22*np.sin(angle)),
                         xytext=(dx + 0.12*np.cos(angle), 0.12*np.sin(angle)),
                         arrowprops=dict(arrowstyle='->', color=CERN_RED, lw=0.8))
ax_mag.text(-0.4, -0.55, 'Beam 1', ha='center', fontsize=6, color=TEXT_WHITE)
ax_mag.text( 0.4, -0.55, 'Beam 2', ha='center', fontsize=6, color=TEXT_WHITE)
ax_mag.text(0.0,  0.75, 'B = 8.33 T\nNbTi @ 1.9 K', ha='center', fontsize=5.5, color=LHC_GOLD)
ax_mag.set_xlim(-1.2, 1.2); ax_mag.set_ylim(-1.0, 1.0)
ax_mag.set_xticks([]); ax_mag.set_yticks([])
ax_mag.set_title('㉔  LHC DIPOLE  Twin-Bore Cross-Section  (8.33 T, NbTi)',
                  pad=5, fontsize=8, color=TEXT_WHITE, fontweight='bold')

# ── Plot 25: Beam size vs energy ─────────────────────────────────────
E_bs_arr = np.linspace(0.45e12, 7.0e12, 200)
sigma_arr = np.array([emitt.beam_size(fodo.beta_max(), E) * 1e6 for E in E_bs_arr])
sigma_ip  = np.array([emitt.beam_size(0.55, E) * 1e6 for E in E_bs_arr])
ax_beamsz.plot(E_bs_arr / 1e12, sigma_arr, color=CERN_BLUE, lw=1.8,
                label=f'σ at QF  (β={fodo.beta_max():.0f} m)')
ax_beamsz.plot(E_bs_arr / 1e12, sigma_ip,  color=CERN_RED, lw=1.5, ls='--',
                label='σ* at IP  (β*=0.55 m)')
ax_beamsz.fill_between(E_bs_arr / 1e12, sigma_arr, alpha=0.1, color=CERN_BLUE)
ax_beamsz.legend(fontsize=5.5)
styled_ax(ax_beamsz, '㉕  BEAM SIZE  σ(E)  (geometric emittance)',
           'Beam Energy [TeV]', 'σ  [μm]')

# ── Plot 26: Momentum spread (RF bucket height) ───────────────────────
h_RF = rf.harmonic
eta_vals = np.array([-3.47e-4, -4.0e-4, -2.5e-4])
colors_eta = [CERN_BLUE, LHC_GOLD, CMS_GREEN]
labels_eta = ['η = -3.47×10⁻⁴ (design)', 'η = -4.0×10⁻⁴', 'η = -2.5×10⁻⁴']
for eta_v, col_v, lbl_v in zip(eta_vals, colors_eta, labels_eta):
    Qs_v = rf.synchrotron_tune(LHC_DESIGN_ENERGY, eta_v)
    delta_max_v = np.sqrt(2 * e_charge * rf.voltage / (np.pi * h_RF * abs(eta_v) *
                           LHC_DESIGN_ENERGY * e_charge)) * 1e3   # ×10³
    t_arr_v = np.linspace(0, 1.0 / Qs_v, 500)
    phi_v = 0.15 * np.cos(2 * np.pi * Qs_v * t_arr_v)
    delta_v = -0.15 * (Qs_v / abs(eta_v) / h_RF) * np.sin(2 * np.pi * Qs_v * t_arr_v) * 1e4
    ax_mom.plot(phi_v, delta_v, color=col_v, lw=1.3, label=lbl_v)
ax_mom.legend(fontsize=5, loc='upper right')
styled_ax(ax_mom, '㉖  SYNCHROTRON OSCILLATIONS  (phase–momentum)',
           'RF Phase φ [rad]', 'δp/p  [×10⁻⁴]')

# ── Plot 27: Halo fraction pie ────────────────────────────────────────
labels_h = list(halo_info.keys())
sizes_h  = list(halo_info.values())
colors_h = [CERN_BLUE, LHC_GOLD, ATLAS_ORANGE, CERN_RED]
wedge_props = dict(edgecolor='#0A0E1A', linewidth=0.8)
wedges, texts, autotexts = ax_stat.pie(
    sizes_h, labels=None, colors=colors_h,
    autopct='%1.2f%%', pctdistance=0.6,
    wedgeprops=wedge_props, startangle=90,
    textprops={'fontsize': 5.5, 'color': TEXT_WHITE},
)
ax_stat.legend(wedges, labels_h, loc='upper left', fontsize=5.5,
               bbox_to_anchor=(-0.15, 1.1))
ax_stat.set_title('㉗  BEAM HALO FRACTIONS  (Multi-stage Collimation)',
                   pad=5, fontsize=8, color=TEXT_WHITE, fontweight='bold')

# ──────────────────────────────────────────────────────────────────────────────
# Decorative footer
# ──────────────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.008,
         "Simulation based on LHC Design Report  ·  CERN Technical Design Reports  ·  "
         "Accelerator School Notes  ·  Physics: Beam Dynamics, Emittance, Luminosity, RF, "
         "Synchrotron Radiation, FODO Lattice, Collimation, Quench Protection, PDFs, "
         "Monte Carlo Event Generation (Z→μμ, H→γγ, QCD dijets), ATLAS-like Detector  ·  "
         "Antimatter: AD/ELENA chain, Stochastic & Electron Cooling, Antihydrogen Formation & Trapping",
         ha='center', fontsize=5.5, color=TEXT_MUTED)

# ════════════════════════════════════════════════════════════════════════════════
#  ANTIMATTER SECTION  — additional figure
# ════════════════════════════════════════════════════════════════════════════════
print("  [5/7] Computing antimatter simulation...")
from antimatter import run_antimatter_simulation
AM = run_antimatter_simulation()

fig2 = plt.figure(figsize=(28, 24), facecolor=DARK_BG)
fig2.text(0.5, 0.985,
          "CERN  ANTIMATTER  SIMULATION  —  AD · ELENA · Antihydrogen",
          ha='center', va='top', fontsize=20, fontweight='bold',
          color='#FF6B9D', fontfamily='monospace')
fig2.text(0.5, 0.978,
          "p̄ Production  ·  Deceleration Chain  ·  Stochastic Cooling  ·  Electron Cooling  ·  "
          "Schottky Spectrum  ·  Positron Source  ·  H̄ Formation  ·  Ioffe Trap  ·  Annihilation",
          ha='center', va='top', fontsize=8.5, color=TEXT_MUTED)

ANTI_PINK   = '#FF6B9D'
ANTI_PURPLE = '#C44EFF'
ANTI_CYAN   = '#00E5FF'
ANTI_GOLD   = '#FFD700'
ANTI_GREEN  = '#39FF14'

gs2 = gridspec.GridSpec(4, 4, figure=fig2,
                         hspace=0.50, wspace=0.38,
                         top=0.970, bottom=0.04,
                         left=0.055, right=0.975)

# ─── ROW 0: Production cross-section | Yield | AD/ELENA chain KE | Chain Bρ ──
ax_xs   = fig2.add_subplot(gs2[0, 0])
ax_yld  = fig2.add_subplot(gs2[0, 1])
ax_chain_ke = fig2.add_subplot(gs2[0, 2])
ax_chain_br = fig2.add_subplot(gs2[0, 3])

# Plot: p̄ production cross-section vs beam KE
T_GeV = AM['T_arr'] / 1e9
ax_xs.semilogy(T_GeV, AM['sigma_arr'] * 1e31, color=ANTI_PINK, lw=2.0)
ax_xs.fill_between(T_GeV, AM['sigma_arr'] * 1e31 + 1e-12,
                    alpha=0.15, color=ANTI_PINK)
ax_xs.axvline(26.0, color=ANTI_GOLD, ls='--', lw=1.2,
               label='PS beam 26 GeV')
ax_xs.axvline(5.63, color=ANTI_CYAN, ls=':', lw=1.0, label='Threshold ~5.6 GeV')
ax_xs.legend(fontsize=5.5)
styled_ax(ax_xs, '①  p̄ PRODUCTION  Cross-section σ(p̄)',
           'Proton KE [GeV]', 'σ [mb × 10⁻²]')

# Plot: p̄ yield per proton vs energy
ax_yld.semilogy(T_GeV, AM['yield_arr'] + 1e-20, color=ANTI_PURPLE, lw=2.0)
ax_yld.fill_between(T_GeV, AM['yield_arr'] + 1e-20,
                     alpha=0.15, color=ANTI_PURPLE)
ax_yld.axvline(26.0, color=ANTI_GOLD, ls='--', lw=1.2, label='PS @ 26 GeV')
ax_yld.text(14, AM['yield_arr'].max() * 0.3,
             f"Rate: {AM['pbar_rate_per_s']:.2e} p̄/s\n(PS 26 GeV, Ir target)",
             fontsize=6, color=TEXT_WHITE,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A0530', alpha=0.8))
ax_yld.legend(fontsize=5.5)
styled_ax(ax_yld, '②  p̄ YIELD  per Incident Proton (Ir target)',
           'Proton KE [GeV]', 'p̄ per proton')

# Plot: KE at each deceleration stage
chain = AM['decel_chain']
stage_names = [d['stage'] for d in chain]
stage_KE    = [d['KE_MeV'] for d in chain]
stage_cool  = [d['cooling'] for d in chain]
cool_colors = {'stochastic': ANTI_PINK, 'electron': ANTI_CYAN, 'none': TEXT_MUTED}
bar_colors  = [cool_colors[c] for c in stage_cool]
bars2 = ax_chain_ke.bar(range(len(stage_names)), stage_KE,
                         color=bar_colors, edgecolor='white',
                         linewidth=0.4, width=0.75)
ax_chain_ke.set_yscale('log')
ax_chain_ke.set_xticks(range(len(stage_names)))
ax_chain_ke.set_xticklabels(stage_names, rotation=55, ha='right', fontsize=4.5)
legend_patches = [
    mpatches.Patch(color=ANTI_PINK,  label='Stochastic cooling'),
    mpatches.Patch(color=ANTI_CYAN,  label='Electron cooling'),
    mpatches.Patch(color=TEXT_MUTED, label='No cooling'),
]
ax_chain_ke.legend(handles=legend_patches, fontsize=5.5)
styled_ax(ax_chain_ke, '③  AD / ELENA  Deceleration Chain  (KE per stage)',
           'Stage', 'Kinetic Energy [MeV]')

# Plot: Bρ (beam rigidity) through chain
stage_Brho = [d['Brho_Tm'] for d in chain]
ax_chain_br.plot(range(len(stage_names)), stage_Brho,
                  color=ANTI_GOLD, lw=2.0, marker='o', ms=4)
ax_chain_br.fill_between(range(len(stage_names)), stage_Brho,
                          alpha=0.12, color=ANTI_GOLD)
ax_chain_br.set_yscale('log')
ax_chain_br.set_xticks(range(len(stage_names)))
ax_chain_br.set_xticklabels(stage_names, rotation=55, ha='right', fontsize=4.5)
styled_ax(ax_chain_br, '④  AD / ELENA  Beam Rigidity Bρ  through Chain',
           'Stage', 'Bρ  [T·m]')

# ─── ROW 1: Stochastic cooling | Electron cooling | Schottky spectrum | Positron ───
ax_sc   = fig2.add_subplot(gs2[1, 0])
ax_ec   = fig2.add_subplot(gs2[1, 1])
ax_sch  = fig2.add_subplot(gs2[1, 2])
ax_pos  = fig2.add_subplot(gs2[1, 3])

# Stochastic cooling
ax_sc2 = ax_sc.twinx()
ax_sc.semilogy(AM['t_cool'], AM['eps_sc'] * 1e6,
                color=ANTI_PINK, lw=2.0, label='Emittance ε [μm·rad]')
ax_sc2.plot(AM['t_cool'], AM['dp_sc'] * 100,
             color=ANTI_GOLD, lw=1.5, ls='--', label='Δp/p [%]')
ax_sc.set_xlabel('Cooling Time [s]', fontsize=7.5)
ax_sc.set_ylabel('ε  [μm·rad]', fontsize=7.5, color=ANTI_PINK)
ax_sc2.set_ylabel('Δp/p  [%]', fontsize=7.5, color=ANTI_GOLD)
ax_sc2.tick_params(colors=ANTI_GOLD, labelsize=6)
lines_c, lab_c = ax_sc.get_legend_handles_labels()
lines_d, lab_d = ax_sc2.get_legend_handles_labels()
ax_sc.legend(lines_c+lines_d, lab_c+lab_d, fontsize=5.5)
tau_str = f'τ_cool ≈ {AM["stochastic_tau"]:.0f} s'
ax_sc.text(0.97, 0.95, tau_str, ha='right', va='top', fontsize=6,
            transform=ax_sc.transAxes, color=TEXT_WHITE,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A0530', alpha=0.8))
ax_sc.set_title('⑤  STOCHASTIC COOLING  ε & Δp/p vs Time  (AD, 3.57 GeV)',
                 pad=5, fontsize=8, color=TEXT_WHITE, fontweight='bold')

# Electron cooling
ax_ec.semilogy(AM['t_ec'], AM['eps_ec'] * 1e6,
                color=ANTI_CYAN, lw=2.0)
ax_ec.fill_between(AM['t_ec'], AM['eps_ec'] * 1e6 + 1e-5,
                    alpha=0.15, color=ANTI_CYAN)
ax_ec.axhline(200e-6 * 1e6, color=TEXT_MUTED, ls='--', lw=0.8,
               label='Initial ε₀ = 200 μm·rad')
ax_ec.axhline(200e-6 * 1e6 * 0.01, color=ANTI_GREEN, ls=':', lw=0.8,
               label='Floor (1% of ε₀)')
ax_ec.legend(fontsize=5.5)
styled_ax(ax_ec, '⑥  ELECTRON COOLING  Emittance vs Time  (AD, 100 MeV)',
           'Cooling Time [s]', 'ε  [μm·rad]')

# Schottky spectrum
sort_idx = np.argsort(AM['f_sch'])
f_plot   = AM['f_sch'][sort_idx] / 1e6   # MHz
P_plot   = AM['P_sch'][sort_idx]
ax_sch.fill_between(f_plot, P_plot, alpha=0.5, color=ANTI_PURPLE)
ax_sch.plot(f_plot, P_plot, color=ANTI_PURPLE, lw=0.8)
ax_sch.set_xlim(0, f_plot.max() * 0.6)
styled_ax(ax_sch, '⑦  SCHOTTKY NOISE SPECTRUM  (longitudinal, 300 MeV AD)',
           'Frequency [MHz]', 'Power [a.u.]')

# Positron source (²²Na)
ax_pos.plot(AM['E_pos'], AM['N_pos'], color=ANTI_PINK, lw=2.0,
             label='²²Na β⁺ spectrum')
ax_pos.fill_between(AM['E_pos'], AM['N_pos'], alpha=0.15, color=ANTI_PINK)
ax_pos2 = ax_pos.twinx()
ax_pos2.plot(AM['t_years'], AM['A_decay'] / 1e6,
              color=ANTI_GOLD, lw=1.3, ls='--', label='Activity decay (T½=2.6 yr)')
ax_pos2.set_ylabel('Activity [MBq]', fontsize=7.5, color=ANTI_GOLD)
ax_pos2.tick_params(colors=ANTI_GOLD, labelsize=6)
ax_pos.set_xlabel('Positron Energy [MeV] / Source Age [years]', fontsize=6.5)
ax_pos.set_ylabel('dN/dE  [a.u.]', fontsize=7.5, color=ANTI_PINK)
lines_e, lab_e = ax_pos.get_legend_handles_labels()
lines_f, lab_f = ax_pos2.get_legend_handles_labels()
ax_pos.legend(lines_e+lines_f, lab_e+lab_f, fontsize=5.5)
ax_pos.set_title('⑧  POSITRON SOURCE  ²²Na β⁺ Spectrum & Activity Decay',
                  pad=5, fontsize=8, color=TEXT_WHITE, fontweight='bold')

# ─── ROW 2: H̄ formation rate vs T | n-level | Ioffe trap | Trapping fraction ─
ax_hform  = fig2.add_subplot(gs2[2, 0])
ax_nlevel = fig2.add_subplot(gs2[2, 1])
ax_ioffe  = fig2.add_subplot(gs2[2, 2])
ax_tfrac  = fig2.add_subplot(gs2[2, 3])

# H̄ formation rate vs temperature
ax_hform.loglog(AM['T_scan'], AM['r3_scan'] + 1e-20,
                 color=ANTI_CYAN, lw=2.0, label='3-body recombination  ∝T⁻⁹/²')
ax_hform.loglog(AM['T_scan'], AM['rrad_scan'] + 1e-20,
                 color=ANTI_PINK, lw=1.5, ls='--', label='Radiative recombination  ∝T⁻¹/²')
ax_hform.axvline(15, color=ANTI_GOLD, ls=':', lw=1.2, label='ALPHA trap ~15 K')
r3_at_15K  = AM['r3_scan'][np.argmin(np.abs(AM['T_scan']-15))]
rrad_at_15K = AM['rrad_scan'][np.argmin(np.abs(AM['T_scan']-15))]
ax_hform.text(0.97, 0.10,
               f"@ 15 K:\n3-body: {r3_at_15K:.2e} H̄/s\nRadiative: {rrad_at_15K:.2e} H̄/s",
               ha='right', va='bottom', fontsize=5.5, color=TEXT_WHITE,
               transform=ax_hform.transAxes,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#0A1535', alpha=0.85))
ax_hform.legend(fontsize=5.5)
styled_ax(ax_hform, '⑨  ANTIHYDROGEN FORMATION RATE  vs Temperature',
           'Plasma Temperature T [K]', 'Rate  [H̄/s]')

# H̄ principal quantum number distribution
ax_nlevel.bar(AM['n_arr'], AM['n_weight'] * 100,
               color=ANTI_PURPLE, alpha=0.8, edgecolor='white', linewidth=0.3, width=0.7)
ax_nlevel.axvline(20, color=ANTI_GOLD, ls='--', lw=1, label='Rydberg states n~20 (3-body)')
ax_nlevel.legend(fontsize=5.5)
styled_ax(ax_nlevel, '⑩  H̄ PRINCIPAL QUANTUM NUMBER  n-distribution (T=15 K)',
           'Principal quantum number n', 'Population  [%]')

# Ioffe–Pritchard trap field
ax_ioffe.plot(AM['z_trap'], AM['B_trap'],
               color=ANTI_GREEN, lw=2.2)
ax_ioffe.fill_between(AM['z_trap'], AM['B_trap'],
                       alpha=0.2, color=ANTI_GREEN)
ax_ioffe.axhline(0.0, color=TEXT_MUTED, lw=0.4, alpha=0.3)
trap_depth_eV = AM['trap_depth_eV']
ax_ioffe.text(0.05, 0.88,
               f'Trap depth:\n{trap_depth_eV*1e3:.3f} meV\n≈ {trap_depth_eV/k_B*11604:.1f} mK',
               fontsize=6, color=TEXT_WHITE, transform=ax_ioffe.transAxes,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#0A2010', alpha=0.85))
styled_ax(ax_ioffe, '⑪  IOFFE–PRITCHARD TRAP  B-field Profile  (H̄ trap)',
           'z  [cm]', 'B  [T]')

# Trapping fraction vs H̄ temperature
ax_tfrac.semilogy(AM['T_hbar'], AM['trap_frac'] * 100 + 1e-6,
                   color=ANTI_PINK, lw=2.0)
ax_tfrac.fill_between(AM['T_hbar'], AM['trap_frac'] * 100 + 1e-6,
                       alpha=0.15, color=ANTI_PINK)
ax_tfrac.axhline(1.0, color=ANTI_GOLD, ls='--', lw=1, label='1% trapped')
ax_tfrac.legend(fontsize=5.5)
styled_ax(ax_tfrac, '⑫  H̄ TRAPPING FRACTION  vs Temperature  (B_max=1T)',
           'H̄ Temperature [K]', 'Fraction Trapped  [%]')

# ─── ROW 3: Annihilation π topology | π⁺pT | γ energy | vertex radius ─────────
ax_anni_diag = fig2.add_subplot(gs2[3, 0])
ax_pipt      = fig2.add_subplot(gs2[3, 1])
ax_gamma_E   = fig2.add_subplot(gs2[3, 2])
ax_vertex    = fig2.add_subplot(gs2[3, 3])

# Annihilation event topology diagram
ax_anni_diag.set_facecolor('#050810')
ax_anni_diag.set_aspect('equal')
# Draw star-burst of pion tracks from vertex
rng_diag = np.random.default_rng(42)
n_tracks_show = 12
angles = np.linspace(0, 2*np.pi, n_tracks_show, endpoint=False)
colors_tracks = ([ANTI_PINK]*4 + [ANTI_CYAN]*4 + [ANTI_GOLD]*2 + ['#AAAAAA']*2)
for i, (ang, col) in enumerate(zip(angles + rng_diag.uniform(-0.3, 0.3, n_tracks_show),
                                    colors_tracks)):
    length = rng_diag.uniform(0.4, 1.0)
    ax_anni_diag.annotate('', xy=(length*np.cos(ang), length*np.sin(ang)),
                           xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color=col,
                                           lw=1.5, mutation_scale=8))
ax_anni_diag.scatter([0], [0], s=200, color='white', zorder=10)
ax_anni_diag.scatter([0], [0], s=80, color=ANTI_PURPLE, zorder=11)
ax_anni_diag.text(0, -1.25, 'p̄ + p  →  π⁺ π⁻ π⁰  (+γγ)', ha='center',
                   fontsize=7, color=TEXT_WHITE)
ax_anni_diag.text(0.7,  0.75, 'π⁺', fontsize=8, color=ANTI_PINK, fontweight='bold')
ax_anni_diag.text(-0.85, 0.65, 'π⁻', fontsize=8, color=ANTI_CYAN, fontweight='bold')
ax_anni_diag.text(0.1,   1.1, 'γ',  fontsize=8, color=ANTI_GOLD, fontweight='bold')
ax_anni_diag.set_xlim(-1.4, 1.4); ax_anni_diag.set_ylim(-1.4, 1.4)
ax_anni_diag.set_xticks([]); ax_anni_diag.set_yticks([])
ax_anni_diag.set_title('⑬  p̄ ANNIHILATION  Topology  (p̄p → πs + γγ)',
                        pad=5, fontsize=8, color=TEXT_WHITE, fontweight='bold')

# Pion pT spectra
if len(AM['pip_pt']) > 0 and len(AM['pim_pt']) > 0:
    pT_bins_pi = np.linspace(0, 0.6, 50)
    counts_pp, _ = np.histogram(AM['pip_pt'], bins=pT_bins_pi)
    counts_pm, _ = np.histogram(AM['pim_pt'], bins=pT_bins_pi)
    bc_pi = 0.5 * (pT_bins_pi[:-1] + pT_bins_pi[1:])
    ax_pipt.step(bc_pi, counts_pp, color=ANTI_PINK,  lw=1.8, label='π⁺')
    ax_pipt.step(bc_pi, counts_pm, color=ANTI_CYAN,  lw=1.5, ls='--', label='π⁻')
    ax_pipt.fill_between(bc_pi, counts_pp, alpha=0.12, color=ANTI_PINK, step='pre')
    ax_pipt.fill_between(bc_pi, counts_pm, alpha=0.10, color=ANTI_CYAN, step='pre')
    ax_pipt.legend(fontsize=6)
styled_ax(ax_pipt, '⑭  ANNIHILATION PION  pT Distribution',
           'pT  [GeV]', 'Events / 12 MeV')

# Photon energy from π⁰ → γγ
if len(AM['gamma_E']) > 0:
    g_bins = np.linspace(0, 0.5, 60)
    g_counts, _ = np.histogram(AM['gamma_E'], bins=g_bins)
    g_bc = 0.5 * (g_bins[:-1] + g_bins[1:])
    ax_gamma_E.bar(g_bc, g_counts, width=(g_bins[1]-g_bins[0])*0.9,
                    color=ANTI_GOLD, alpha=0.85, edgecolor='none')
    ax_gamma_E.axvline(0.0675, color=ANTI_PINK, ls='--', lw=1.2,
                        label='π⁰/2 = 67.5 MeV (rest frame)')
    ax_gamma_E.legend(fontsize=5.5)
styled_ax(ax_gamma_E, '⑮  π⁰ → γγ  Photon Energy  (from annihilation)',
           'E_γ  [GeV]', 'Events / 8 MeV')

# Annihilation vertex radial distribution
if len(AM['vertex_r']) > 0:
    v_bins = np.linspace(0, 2.0, 50)
    v_counts, _ = np.histogram(AM['vertex_r'], bins=v_bins)
    v_bc = 0.5 * (v_bins[:-1] + v_bins[1:])
    ax_vertex.bar(v_bc, v_counts, width=(v_bins[1]-v_bins[0])*0.9,
                   color=ANTI_GREEN, alpha=0.85, edgecolor='none')
    # Exponential fit overlay
    lam = 1.0  # mean depth ~1 mm
    fit_curve = v_counts.max() * np.exp(-v_bc / lam)
    ax_vertex.plot(v_bc, fit_curve, color='white', lw=1.3, ls='--', label='Exp. fit')
    ax_vertex.legend(fontsize=5.5)
styled_ax(ax_vertex, '⑯  ANNIHILATION VERTEX  Radial Distribution',
           'r  [mm]', 'Events / 40 μm')

# Info box: key antimatter numbers
fig2.text(0.01, 0.01,
           f"  KEY NUMBERS:  p̄/s ≈ {AM['pbar_rate_per_s']:.1e}  ·  "
           f"H̄ 3-body rate ≈ {AM['ah_3body_rate']:.1e} s⁻¹  ·  "
           f"Trap depth ≈ {AM['trap_depth_eV']*1e3:.3f} meV  ·  "
           f"Stochastic τ ≈ {AM['stochastic_tau']:.0f} s  ·  "
           f"Threshold KE = 5.63 GeV  ·  AD C = 182.4 m  ·  ELENA C = 30.4 m",
           fontsize=6.5, color=ANTI_GOLD)

print("  [6/7] Saving figures...")
out_path2 = "CERN_Antimatter_Simulation.png"
try:
    os.makedirs("/mnt/user-data/outputs", exist_ok=True)
    out_path2_alt = "/mnt/user-data/outputs/CERN_Antimatter_Simulation.png"
    fig2.savefig(out_path2_alt, dpi=150, bbox_inches='tight', facecolor=DARK_BG, edgecolor='none')
    print(f"         Saved → {out_path2_alt}")
except Exception:
    pass
fig2.savefig(out_path2, dpi=150, bbox_inches='tight', facecolor=DARK_BG, edgecolor='none')
plt.close(fig2)
print(f"         Saved → {out_path2}")

print("  [5/7] Saving LHC figure...")
out_path = "CERN_LHC_Collider_Simulation.png"
try:
    out_path_alt = "/mnt/user-data/outputs/CERN_LHC_Collider_Simulation.png"
    fig.savefig(out_path_alt, dpi=150, bbox_inches='tight', facecolor=DARK_BG, edgecolor='none')
    print(f"  [6/7] Saved → {out_path_alt}")
except Exception:
    pass
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG, edgecolor='none')
plt.close(fig)
print(f"  [6/7] Saved → {out_path}")
print("  [7/7] Done.")

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  CERN LHC PARTICLE COLLIDER SIMULATION")
print("  Initializing physics engines...")
print("="*70)
