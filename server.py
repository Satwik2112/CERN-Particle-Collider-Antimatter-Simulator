"""
FastAPI backend for CERN LHC / Antimatter simulation.
Delegates all physics to physics_core.py, antimatter.py, detector_sim.py.
"""
import os, sys, math
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Import simulation modules ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from physics_core import (
    LHCBeam, Emittance, RFCavity, FODOCell, CollimationSystem,
    QuenchProtection, synchrotron_radiation_loss,
    beam_beam_tune_shift, track_betatron, LHC_CIRCUMFERENCE,
    m_proton, m_electron, e_charge, c, LHC_DESIGN_ENERGY,
    LHC_BUNCHES, LHC_BUNCH_POP, LHC_NORM_EMITTANCE, LHC_RF_FREQUENCY,
    gamma_factor, beta_factor
)
from detector_sim import (
    simulate_injection_chain, energy_ramp_lhc, pdf_valence_quark,
    PPEventGenerator, ATLASLikeDetector
)
from antimatter import (
    pbar_production_cross_section, pbar_yield_per_proton,
    pbar_threshold_energy, m_pbar, simulate_deceleration_chain,
    StochasticCooling, AntihydrogenFormation, IoffeTrap,
    ElectronCooling, PositronSource,
    AD_CIRCUMFERENCE
)

# ── Helpers ─────────────────────────────────────────────────────────────────
def sanitize(obj):
    """Recursively replace NaN / Inf with 0.0 so JSON encoding never fails."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    return obj


def simulate_annihilation(n_particles: int, seed: int):
    """Simple annihilation vertex scatter for visualisation."""
    rng = np.random.default_rng(seed)
    r = rng.exponential(0.1, n_particles)
    theta = rng.uniform(0, 2 * np.pi, n_particles)
    x = (r * np.cos(theta)).tolist()
    y = (r * np.sin(theta)).tolist()
    return {"x": x, "y": y}

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schemas ─────────────────────────────────────────────────────────
class LHCParams(BaseModel):
    energy: float
    bunchPop: float
    emittVal: float
    betaStar: float
    crossingAngle: float
    rfVolt: float
    primarySigma: float
    quenchR: float
    mcSeed: int

class AntimatterParams(BaseModel):
    pbarEnergy: float
    pbarMat: str
    pbarThickness: float
    coolBw: float
    hbarTemp: float
    ioffeField: float
    annSeed: int
    ecoolCurrent: float
    posEnergy: float
    posTargetT: float

# ═════════════════════════════════════════════════════════════════════════════
#  /api/lhc  —  LHC accelerator + collision simulation
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/api/lhc")
def lhc_simulation(params: LHCParams):
    # ── Core beam ───────────────────────────────────────────────────────────
    beam = LHCBeam(
        energy_eV=params.energy,
        n_bunches=LHC_BUNCHES,
        bunch_population=params.bunchPop,
        norm_emittance=params.emittVal,
        beta_star=params.betaStar,
        crossing_angle=params.crossingAngle,
    )

    # beam.summary() returns a dict like {'Energy [TeV]': 7.0, ...}
    # The frontend expects [{name, unit, val}, ...]
    raw_summary = beam.summary()
    summary_rows = []
    for key, val in raw_summary.items():
        # Try to split "Energy [TeV]" into name="Energy" unit="TeV"
        if "[" in key and key.endswith("]"):
            parts = key.rsplit("[", 1)
            name = parts[0].strip()
            unit = parts[1].rstrip("]").strip()
        else:
            name = key
            unit = ""
        summary_rows.append({"name": name, "unit": unit, "val": val})

    stored_e = (LHC_BUNCHES * params.bunchPop * params.energy * e_charge) / 1e6

    # ── 1. Injector Chain ──────────────────────────────────────────────────
    inj_data = simulate_injection_chain()

    # ── 2. FODO Cell ───────────────────────────────────────────────────────
    fodo = FODOCell(L_cell=106.9, phase_advance=np.pi / 3)
    s_fodo, beta_fodo, alpha_fodo = fodo.twiss_along_cell(150)

    # ── 3. Phase Ellipse ───────────────────────────────────────────────────
    x_ell, xp_ell = beam.emittance.phase_ellipse(fodo.beta_max(), 0.0, params.energy)
    x_ell_inj, xp_ell_inj = beam.emittance.phase_ellipse(fodo.beta_max(), 0.0, 0.45e12)

    # ── 4. Betatron Oscillations ───────────────────────────────────────────
    turns, x_bet = track_betatron(beta_fodo, 150, x0=1e-3, xp0=0.0, tune_frac=0.31)

    # ── 5. Synchrotron Radiation ───────────────────────────────────────────
    e_scan = np.linspace(1e12, 14e12, 80)
    rho_p = LHC_CIRCUMFERENCE / (2 * np.pi)
    u_scan = [float(synchrotron_radiation_loss(e, rho_p, m_proton)) for e in e_scan]

    # ── 6. RF Phase Space ──────────────────────────────────────────────────
    rf = RFCavity(frequency=LHC_RF_FREQUENCY, voltage=params.rfVolt)
    phi1, delta1 = rf.bunch_evolution(0.1, 0.0005, 1000, -3.47e-4, params.energy)
    phi2, delta2 = rf.bunch_evolution(0.3, 0.001, 1000, -3.47e-4, params.energy)

    # ── 7. Luminosity vs Energy ────────────────────────────────────────────
    e_scan_lum = np.linspace(1e12, 14e12, 50)
    l_scan_e = []
    for e in e_scan_lum:
        try:
            b = LHCBeam(energy_eV=e, n_bunches=LHC_BUNCHES,
                        bunch_population=params.bunchPop,
                        norm_emittance=params.emittVal,
                        beta_star=params.betaStar,
                        crossing_angle=params.crossingAngle)
            l_scan_e.append(float(b.peak_luminosity() / 1e34))
        except Exception:
            l_scan_e.append(0.0)

    # ── 8. Luminosity vs β* ────────────────────────────────────────────────
    bs_scan = np.linspace(0.15, 5.0, 50)
    l_scan_bs = []
    for bs in bs_scan:
        try:
            b = LHCBeam(energy_eV=params.energy, n_bunches=LHC_BUNCHES,
                        bunch_population=params.bunchPop,
                        norm_emittance=params.emittVal,
                        beta_star=bs,
                        crossing_angle=params.crossingAngle)
            l_scan_bs.append(float(b.peak_luminosity() / 1e34))
        except Exception:
            l_scan_bs.append(0.0)

    # ── 9. Ramp ────────────────────────────────────────────────────────────
    t_ramp, e_ramp = energy_ramp_lhc(0.45e12, 7e12, 1200, 100)

    # ── 10. Quench ─────────────────────────────────────────────────────────
    qp = QuenchProtection(I_nominal=11850.0, L_magnet=0.12, R_quench=params.quenchR)
    t_quench = np.linspace(0, 0.5, 100)
    i_quench = qp.current_decay(t_quench)

    # ── 11. Beam-Beam Tune Shift ───────────────────────────────────────────
    n_scan = np.linspace(0.1e11, 3.0e11, 50)
    xi_scan = [float(beam_beam_tune_shift(n, params.betaStar,
                                          beam.sigma_IP, beam.sigma_IP,
                                          beam.gamma))
               for n in n_scan]

    # ── 12. PDF ────────────────────────────────────────────────────────────
    x_pdf = np.linspace(0.005, 0.995, 80)
    pdf_u = [float(pdf_valence_quark(x, 10000.0, 'u')) for x in x_pdf]
    pdf_d = [float(pdf_valence_quark(x, 10000.0, 'd')) for x in x_pdf]
    pdf_g = [float(pdf_valence_quark(x, 10000.0, 'g') / 4.0) for x in x_pdf]

    # ── 13. Monte-Carlo Events ─────────────────────────────────────────────
    # Energy is in eV, PPEventGenerator expects TeV
    beam_energy_tev = params.energy / 1e12
    gen = PPEventGenerator(beam_energy_tev, params.mcSeed)
    det = ATLASLikeDetector()
    mc_rng = np.random.default_rng(params.mcSeed)

    z_events = gen.generate_events(1200, 'Z')
    z_masses = det.reconstruct_Z_mass(z_events, mc_rng).tolist()

    h_events = gen.generate_events(900, 'Higgs')
    h_masses = []
    for evt in h_events:
        if len(evt) < 2:
            continue
        p1s = det.smear_particle(evt[0], mc_rng)
        p2s = det.smear_particle(evt[1], mc_rng)
        if det.is_accepted(p1s) and det.is_accepted(p2s):
            m2 = ((p1s.E + p2s.E)**2 - (p1s.px + p2s.px)**2
                  - (p1s.py + p2s.py)**2 - (p1s.pz + p2s.pz)**2)
            if m2 > 0:
                h_masses.append(float(np.sqrt(m2)))

    dijet_events = gen.generate_events(1800, 'dijet')
    dijet_pt = [float(max(p.pt for p in evt)) for evt in dijet_events if evt]

    # ── Return ─────────────────────────────────────────────────────────────
    return sanitize({
        "mc": {"zMasses": z_masses, "hMasses": h_masses, "dijetPt": dijet_pt},
        "header": {
            "peakLuminosity": beam.peak_luminosity(),
            "bunchPopulation": beam.bunch_population,
            "sigmaIP": beam.sigma_IP,
            "storedEnergy": stored_e,
        },
        "summary": summary_rows,
        "inj": inj_data,
        "fodo": {
            "s": s_fodo.tolist(),
            "beta": beta_fodo.tolist(),
            "alpha": alpha_fodo.tolist(),
            "betaMax": fodo.beta_max(),
        },
        "phaseEllipse": {
            "x7T": x_ell.tolist(),
            "xp7T": xp_ell.tolist(),
            "x450G": x_ell_inj.tolist(),
            "xp450G": xp_ell_inj.tolist(),
        },
        "betatron": {"turns": turns.tolist(), "x": x_bet.tolist()},
        "radiation": {"eScan": e_scan.tolist(), "uScan": u_scan},
        "rfSpace": {
            "phi1": phi1.tolist(), "delta1": delta1.tolist(),
            "phi2": phi2.tolist(), "delta2": delta2.tolist(),
        },
        "lumEnergy": {"eScan": e_scan_lum.tolist(), "lScan": l_scan_e},
        "lumBeta":   {"bsScan": bs_scan.tolist(),   "lScan": l_scan_bs},
        "ramp":      {"t": t_ramp.tolist(),         "E": e_ramp.tolist()},
        "quench":    {"t": t_quench.tolist(),       "I": i_quench.tolist()},
        "beamBeam":  {"nScan": n_scan.tolist(),     "xiScan": xi_scan},
        "pdf":       {"x": x_pdf.tolist(), "u": pdf_u, "d": pdf_d, "g": pdf_g},
    })


# ═════════════════════════════════════════════════════════════════════════════
#  /api/antimatter  —  Antimatter simulation
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/api/antimatter")
def antimatter_simulation(params: AntimatterParams):
    # ── Annihilation vertices ──────────────────────────────────────────────
    annihilation = simulate_annihilation(1000, params.annSeed)

    # ── Yield ──────────────────────────────────────────────────────────────
    yield_val = pbar_yield_per_proton(params.pbarEnergy, params.pbarMat, params.pbarThickness)

    # ── Deceleration chain ─────────────────────────────────────────────────
    decel = simulate_deceleration_chain()

    # ── Cross-section scan ─────────────────────────────────────────────────
    t_scan_pbar = np.linspace(pbar_threshold_energy(), 35.0e9, 60)
    xs_scan = [float(pbar_production_cross_section(t)) for t in t_scan_pbar]

    # ── Yield vs thickness ─────────────────────────────────────────────────
    thick_scan = np.linspace(1e-3, 50e-3, 50)
    yield_scan = [float(pbar_yield_per_proton(params.pbarEnergy, params.pbarMat, t))
                  for t in thick_scan]

    # ── Stochastic Cooling — momentum spread evolution ─────────────────────
    sc = StochasticCooling(bandwidth_Hz=params.coolBw, gain=0.5,
                           mixing_factor=2.0, N_particles=3e7)
    sc_t_scan = np.linspace(0, 50, 50)
    dp0 = 1e-3  # initial Δp/p
    sc_dp_scan = sc.momentum_spread_evolution(dp0, sc_t_scan).tolist()

    # ── Electron Cooling — emittance / momentum-spread evolution ───────────
    ec = ElectronCooling(I_electron_A=params.ecoolCurrent,
                         KE_electron_eV=50e3, L_cooler_m=1.5, T_electron_K=10000.0)
    gamma_pb = gamma_factor(300e6 + m_pbar, m_pbar)
    beta_pb = beta_factor(gamma_pb)
    f_rev = beta_pb * c / AD_CIRCUMFERENCE
    
    # ── Schottky spectrum ──────────────────────────────────────────────────
    f_sch, P_sch = sc.schottky_spectrum(f_rev, 0.005, 12)
    schottky_data = {"f": f_sch.tolist(), "P": P_sch.tolist()}

    ec_t_scan = np.linspace(0, 10, 50)
    # Use cooling_force to derive an approximate exponential decay
    v_pbar = beta_pb * c
    F_cool = ec.cooling_force_magnitude(v_pbar)
    if F_cool > 1e-30:
        tau_ec = (m_pbar * e_charge / c) / (F_cool * e_charge / c * f_rev + 1e-30)
        ec_dp_scan = [float(dp0 * np.exp(-t / max(tau_ec, 0.01))) for t in ec_t_scan]
    else:
        ec_dp_scan = [float(dp0) for _ in ec_t_scan]

    # ── Positron source — energy spectrum ──────────────────────────────────
    ps = PositronSource()
    ps_e_arr, ps_spec_arr = ps.positron_energy_spectrum(50)
    ps_e_scan = ps_e_arr.tolist()      # [MeV]
    ps_spec   = ps_spec_arr.tolist()   # normalised spectrum

    # ── Ioffe trap — field profile ─────────────────────────────────────────
    trap = IoffeTrap(B_min_T=0.0, B_max_T=params.ioffeField, trap_length_m=0.27)
    trap_z, trap_B_arr = trap.field_profile(50)
    trap_r_scan = trap_z.tolist()     # [cm]
    trap_b_scan = trap_B_arr.tolist()

    # ── Antihydrogen — formation rate vs temperature ───────────────────────
    ah = AntihydrogenFormation(N_pbar=1e4, N_positron=1e6,
                               T_plasma_K=params.hbarTemp, plasma_radius=2e-3)
    t_hbar_arr = np.linspace(0.1, 50, 50)
    _, r3_arr, r_rad_arr = ah.temperature_scan_formation_rate(t_hbar_arr)
    t_hbar   = t_hbar_arr.tolist()
    ah_rate  = (r3_arr + r_rad_arr).tolist()  # total rate

    # ── Return ─────────────────────────────────────────────────────────────
    return sanitize({
        "summary": {"yield": yield_val},
        "annihilation": annihilation,
        "decel": decel,
        "schottky": schottky_data,
        "scans": {
            "tScanPbar": t_scan_pbar.tolist(),
            "xsScan": xs_scan,
            "thickScan": thick_scan.tolist(),
            "yieldScan": yield_scan,
            "scTime": sc_t_scan.tolist(),
            "scDp": sc_dp_scan,
            "ecTime": ec_t_scan.tolist(),
            "ecDp": ec_dp_scan,
            "psE": ps_e_scan,
            "psSpec": ps_spec,
            "trapR": trap_r_scan,
            "trapB": trap_b_scan,
            "tHbar": t_hbar,
            "ahRate": ah_rate,
        },
    })
