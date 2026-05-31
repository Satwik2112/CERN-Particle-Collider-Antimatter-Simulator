"""
=============================================================================
  CERN LHC Simulation — Injector Chain, Event Generator, Detector
=============================================================================
  Covers:
    - Injector chain: Linac4 → PSB → PS → SPS → LHC (energy ramp)
    - Simplified Pythia-like Monte Carlo pp collision generator
    - Parton Distribution Functions (simple parametric)
    - Detector geometry (ATLAS-like: tracker, ECAL, HCAL, muon spectrometer)
    - Track reconstruction (helix in solenoid field)
    - Calorimeter energy deposition
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from physics_core import (gamma_factor, beta_factor, momentum_eV,
                           beam_rigidity, e_charge, c, m_proton)

# ──────────────────────────────────────────────────────────────────────────────
# 1. INJECTOR CHAIN
# ──────────────────────────────────────────────────────────────────────────────
INJECTOR_CHAIN = [
    {"name": "Linac4",  "type": "linac",      "E_out_MeV":   160.0,  "L_m": 80.0},
    {"name": "PSB",     "type": "synchrotron", "E_out_MeV":  2000.0,  "C_m": 157.0},
    {"name": "PS",      "type": "synchrotron", "E_out_MeV": 26000.0,  "C_m": 628.0},
    {"name": "SPS",     "type": "synchrotron", "E_out_MeV":450000.0,  "C_m": 6911.0},
    {"name": "LHC",     "type": "synchrotron", "E_out_MeV":7000000.0, "C_m": 26659.0},
]

def simulate_injection_chain(n_steps: int = 100) -> Dict[str, np.ndarray]:
    """
    Simulate beam energy ramp through the CERN injector chain.
    Returns dict with stage names and energy/Bρ/β profiles.
    """
    results = {}
    for stage in INJECTOR_CHAIN:
        E_MeV = stage["E_out_MeV"]
        E_eV  = E_MeV * 1e6
        g = gamma_factor(E_eV, m_proton)
        b = beta_factor(g)
        Brho = beam_rigidity(E_eV, m_proton)
        results[stage["name"]] = {
            "energy_TeV": E_eV / 1e12,
            "gamma": g,
            "beta": b,
            "Brho_Tm": Brho,
            "circumference_m": stage.get("C_m", stage.get("L_m", 0)),
            "type": stage["type"],
        }
    return results

def energy_ramp_lhc(E_inj_eV: float = 0.45e12,
                     E_top_eV: float = 7.0e12,
                     ramp_time_s: float = 1200.0,
                     n_pts: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    LHC energy ramp: E(t) follows a sinusoidal profile (~20 min).
    Returns time [s] and energy [TeV] arrays.
    """
    t = np.linspace(0, ramp_time_s, n_pts)
    # Sinusoidal ramp (realistic magnet current profile)
    phi = t / ramp_time_s * np.pi
    E = E_inj_eV + (E_top_eV - E_inj_eV) * (1 - np.cos(phi)) / 2
    return t, E / 1e12   # [s], [TeV]

# ──────────────────────────────────────────────────────────────────────────────
# 2. PARTON DISTRIBUTION FUNCTIONS (simple parametric, CTEQ-like)
# ──────────────────────────────────────────────────────────────────────────────
def pdf_valence_quark(x: float, Q2: float, quark: str = 'u') -> float:
    """
    Simplified parametric PDF xf(x,Q²) for valence quarks.
    Based on MRST/CTEQ leading-order parametrization.
    """
    # Parametric form: xf(x) = N · x^a · (1-x)^b · (1 + c·√x + d·x)
    params = {
        'u': (1.894, -0.463, 3.048, 0.0, 0.0),   # (N, a, b, c, d)
        'd': (0.857, -0.322, 4.052, 0.0, 0.0),
        'g': (1.7,   -0.18,  3.8,  3.0, 0.0),    # gluon
        's': (0.2,    0.0,   8.0,  0.0, 0.0),
    }
    N, a, b, cc, d = params.get(quark, params['u'])
    if x <= 0 or x >= 1:
        return 0.0
    # Q² evolution (very approximate LO DGLAP)
    alpha_s = 0.118 / (1 + 0.118 / (2*np.pi) * np.log(Q2 / 91.2**2 + 1e-10))
    scale = 1 + 0.1 * np.log(Q2 / 10.0 + 1e-10)
    return N * x**a * (1-x)**b * (1 + cc*np.sqrt(x) + d*x) * scale

def sample_parton_x(n_events: int = 1000,
                    s_cm: float = (14e3)**2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample x1, x2 from PDFs for pp collision at √s = 14 TeV.
    Uses rejection sampling on u-quark PDF.
    """
    rng = np.random.default_rng(42)
    x1 = np.empty(n_events)
    x2 = np.empty(n_events)
    count = 0
    max_pdf = 5.0   # rough upper bound
    while count < n_events:
        x_try = rng.uniform(1e-3, 1.0)
        f_try = rng.uniform(0, max_pdf)
        if f_try < pdf_valence_quark(x_try, s_cm * x_try):
            x1[count] = x_try
            count += 1
    count = 0
    while count < n_events:
        x_try = rng.uniform(1e-3, 1.0)
        f_try = rng.uniform(0, max_pdf)
        if f_try < pdf_valence_quark(x_try, s_cm * x_try):
            x2[count] = x_try
            count += 1
    return x1, x2

# ──────────────────────────────────────────────────────────────────────────────
# 3. MONTE CARLO EVENT GENERATOR (Pythia-like pp → X)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Particle4Vector:
    """4-momentum (E, px, py, pz) in GeV."""
    E:  float
    px: float
    py: float
    pz: float
    pid: int   = 0    # PDG particle ID
    charge: int = 0

    @property
    def pt(self) -> float:
        return np.sqrt(self.px**2 + self.py**2)

    @property
    def eta(self) -> float:
        p = np.sqrt(self.px**2 + self.py**2 + self.pz**2)
        if p == 0 or (p + self.pz) <= 0:
            return 0.0
        return 0.5 * np.log((p + self.pz) / (p - self.pz + 1e-30))

    @property
    def phi(self) -> float:
        return np.arctan2(self.py, self.px)

    @property
    def mass(self) -> float:
        m2 = self.E**2 - self.px**2 - self.py**2 - self.pz**2
        return np.sqrt(max(m2, 0.0))

# PDG IDs used
PDG = {
    'u': 2, 'd': 1, 's': 3, 'c': 4, 'b': 5,
    'g': 21, 'gamma': 22,
    'e-': 11, 'e+': -11, 'mu-': 13, 'mu+': -13,
    'nu_e': 12, 'nu_mu': 14,
    'W+': 24, 'W-': -24, 'Z': 23, 'H': 25,
    'pi+': 211, 'pi-': -211, 'pi0': 111,
    'K+': 321, 'K-': -321, 'p': 2212, 'pbar': -2212,
}

PARTICLE_MASS_GEV = {
    11: 0.000511, 13: 0.1057, 15: 1.777,
    2212: 0.9383, 211: 0.1396, 321: 0.4937,
    24: 80.4, 23: 91.2, 25: 125.1, 0: 0.0
}

def generate_isotropic_decay(M: float, m1: float, m2: float,
                              rng: np.random.Generator
                              ) -> Tuple[Particle4Vector, Particle4Vector]:
    """2-body decay in rest frame of parent particle."""
    if M < m1 + m2:
        M = m1 + m2 + 1e-6
    pcm = np.sqrt(max((M**2 - (m1+m2)**2)*(M**2 - (m1-m2)**2), 0.0)) / (2*M)
    costh = rng.uniform(-1, 1)
    phi   = rng.uniform(0, 2*np.pi)
    sinth = np.sqrt(1 - costh**2)
    px = pcm * sinth * np.cos(phi)
    py = pcm * sinth * np.sin(phi)
    pz = pcm * costh
    E1 = np.sqrt(m1**2 + pcm**2)
    E2 = np.sqrt(m2**2 + pcm**2)
    p1 = Particle4Vector(E1,  px,  py,  pz)
    p2 = Particle4Vector(E2, -px, -py, -pz)
    return p1, p2

def lorentz_boost(p: Particle4Vector, beta_z: float) -> Particle4Vector:
    """Boost particle along z by β_z."""
    g  = 1.0 / np.sqrt(1 - beta_z**2 + 1e-30)
    Ez = g * (p.E - beta_z * p.pz)
    pz = g * (p.pz - beta_z * p.E)
    return Particle4Vector(Ez, p.px, p.py, pz, p.pid, p.charge)

class PPEventGenerator:
    """
    Simplified pp → hard scatter Monte Carlo generator.
    Processes: qqbar→Z→ll, gg→H, qq→qq (QCD jet), qg→qg
    """
    def __init__(self, sqrt_s_TeV: float = 14.0, seed: int = 42):
        self.sqrt_s = sqrt_s_TeV * 1e3  # [GeV]
        self.s = self.sqrt_s**2
        self.rng = np.random.default_rng(seed)

    def generate_Z_event(self) -> List[Particle4Vector]:
        """pp → Z → μ⁺μ⁻ (Drell-Yan)."""
        M_Z = 91.2  # GeV
        Gamma_Z = 2.495
        # Sample Z mass with BW
        m_ll = self.rng.normal(M_Z, Gamma_Z/2)
        m_ll = max(m_ll, 60.0)
        # Rapidity of Z
        tau = m_ll**2 / self.s
        y_max = -0.5 * np.log(tau)
        y_Z   = self.rng.uniform(-y_max, y_max)
        # Transverse momentum of Z
        pT_Z = abs(self.rng.exponential(5.0))  # GeV
        phi_Z = self.rng.uniform(0, 2*np.pi)
        # Z 4-vector
        mT = np.sqrt(m_ll**2 + pT_Z**2)
        pz_Z = mT * np.sinh(y_Z)
        E_Z  = mT * np.cosh(y_Z)
        px_Z = pT_Z * np.cos(phi_Z)
        py_Z = pT_Z * np.sin(phi_Z)
        # Decay Z → μ+μ-
        m_mu = PARTICLE_MASS_GEV[13]
        mu1_cm, mu2_cm = generate_isotropic_decay(m_ll, m_mu, m_mu, self.rng)
        # Boost to lab
        beta_z = pz_Z / E_Z
        beta_x = px_Z / E_Z
        beta_y = py_Z / E_Z
        def boost_full(p, bx, by, bz):
            b2 = bx**2 + by**2 + bz**2
            gamma = 1.0 / np.sqrt(1 - b2 + 1e-30)
            bdotp = bx*p.px + by*p.py + bz*p.pz
            fac = (gamma - 1) / b2 if b2 > 1e-10 else 0.0
            E_new  = gamma * (p.E - bdotp)
            px_new = p.px + fac*bdotp*bx - gamma*bx*p.E
            py_new = p.py + fac*bdotp*by - gamma*by*p.E
            pz_new = p.pz + fac*bdotp*bz - gamma*bz*p.E
            return Particle4Vector(E_new, px_new, py_new, pz_new, p.pid, p.charge)
        mu1 = boost_full(mu1_cm, beta_x, beta_y, beta_z)
        mu2 = boost_full(mu2_cm, beta_x, beta_y, beta_z)
        mu1.pid, mu1.charge = PDG['mu-'], -1
        mu2.pid, mu2.charge = PDG['mu+'], +1
        return [mu1, mu2]

    def generate_QCD_dijet(self) -> List[Particle4Vector]:
        """pp → dijet (simplified 2→2 QCD scattering)."""
        # Sample pT of hard scatter
        pT_hat = abs(self.rng.exponential(40.0)) + 20.0  # GeV
        y_hat  = self.rng.uniform(-3.0, 3.0)
        phi1   = self.rng.uniform(0, 2*np.pi)
        phi2   = phi1 + np.pi   # back-to-back

        def make_jet(pT, eta, phi, pid=PDG['g']):
            px = pT * np.cos(phi)
            py = pT * np.sin(phi)
            pz = pT * np.sinh(eta)
            E  = pT * np.cosh(eta)
            return Particle4Vector(E, px, py, pz, pid, 0)

        j1 = make_jet(pT_hat,  y_hat, phi1)
        j2 = make_jet(pT_hat, -y_hat, phi2)
        return [j1, j2]

    def generate_Higgs_event(self) -> List[Particle4Vector]:
        """gg → H → γγ (Higgs via gluon fusion, diphoton decay)."""
        M_H = 125.1  # GeV
        pT_H = abs(self.rng.exponential(15.0))
        y_H  = self.rng.normal(0, 1.5)
        phi_H = self.rng.uniform(0, 2*np.pi)
        mT = np.sqrt(M_H**2 + pT_H**2)
        px_H = pT_H * np.cos(phi_H)
        py_H = pT_H * np.sin(phi_H)
        pz_H = mT * np.sinh(y_H)
        E_H  = mT * np.cosh(y_H)
        # H → γγ decay
        g1_cm, g2_cm = generate_isotropic_decay(M_H, 0, 0, self.rng)
        bx, by, bz = px_H/E_H, py_H/E_H, pz_H/E_H
        b2 = bx**2 + by**2 + bz**2
        gamma_boost = 1.0 / np.sqrt(1 - b2 + 1e-30)
        def boost_full(p):
            bdp = bx*p.px + by*p.py + bz*p.pz
            fac = (gamma_boost - 1) / b2 if b2 > 1e-10 else 0.0
            return Particle4Vector(
                gamma_boost*(p.E - bdp),
                p.px + fac*bdp*bx - gamma_boost*bx*p.E,
                p.py + fac*bdp*by - gamma_boost*by*p.E,
                p.pz + fac*bdp*bz - gamma_boost*bz*p.E,
                PDG['gamma'], 0)
        return [boost_full(g1_cm), boost_full(g2_cm)]

    def generate_events(self, n_events: int = 1000,
                        process: str = 'Z') -> List[List[Particle4Vector]]:
        """Generate n_events of given process."""
        gen_map = {
            'Z':     self.generate_Z_event,
            'dijet': self.generate_QCD_dijet,
            'Higgs': self.generate_Higgs_event,
        }
        fn = gen_map.get(process, self.generate_Z_event)
        return [fn() for _ in range(n_events)]

# ──────────────────────────────────────────────────────────────────────────────
# 4. DETECTOR SIMULATION (ATLAS-like)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ATLASLikeDetector:
    """
    Simplified ATLAS geometry.
    Inner tracker: solenoid B=2T, |η|<2.5, r=50-1050 mm
    ECAL:          |η|<3.2, depth=25 X₀
    HCAL:          |η|<3.0, depth=10 λ
    Muon:          |η|<2.7
    """
    B_solenoid: float = 2.0    # [T] inner solenoid
    r_tracker_min: float = 0.05  # [m]
    r_tracker_max: float = 1.05
    eta_max_tracker: float = 2.5
    eta_max_muon: float    = 2.7

    def track_radius(self, pT_GeV: float, charge: int) -> float:
        """
        Helix radius in solenoid:  r = p_T / (qB)  [m]
        p_T in GeV/c, B in T → r [m]
        """
        if charge == 0:
            return np.inf
        p_SI = pT_GeV * 1e9 * e_charge / c   # [kg·m/s]
        return p_SI / (abs(charge) * e_charge * self.B_solenoid)

    def track_sagitta(self, pT_GeV: float, L: float = 1.0) -> float:
        """
        Sagitta s = L²/(8R) for track curvature measurement.
        L = track length in tracker [m].
        """
        R = self.track_radius(pT_GeV, 1)
        return L**2 / (8 * R)

    def pT_resolution(self, pT_GeV: float) -> float:
        """Approximate σ(pT)/pT ≈ a·pT ⊕ b (stochastic ⊕ constant)."""
        a = 3.4e-4  # [1/GeV] momentum resolution term
        b = 0.01    # constant term
        return np.sqrt((a * pT_GeV)**2 + b**2)

    def ecal_energy_resolution(self, E_GeV: float) -> float:
        """σ(E)/E = a/√E ⊕ b (ECAL: a~10%, b~0.7%)."""
        a = 0.10
        b = 0.007
        c_noise = 0.003
        return np.sqrt((a / np.sqrt(max(E_GeV, 0.001)))**2 +
                       b**2 + (c_noise / E_GeV)**2)

    def hcal_energy_resolution(self, E_GeV: float) -> float:
        """σ(E)/E = a/√E ⊕ b (HCAL: a~50%, b~3%)."""
        a = 0.50
        b = 0.03
        return np.sqrt((a / np.sqrt(max(E_GeV, 0.001)))**2 + b**2)

    def smear_particle(self, p: Particle4Vector,
                        rng: np.random.Generator) -> Particle4Vector:
        """Apply detector smearing to a particle's 4-momentum."""
        pid = abs(p.pid)
        if pid in (11, 13):   # electron or muon
            rel_res = self.pT_resolution(p.pt)
            pT_smeared = p.pt * (1 + rng.normal(0, rel_res))
            scale = pT_smeared / (p.pt + 1e-30)
            return Particle4Vector(p.E * scale, p.px * scale, p.py * scale,
                                   p.pz * scale, p.pid, p.charge)
        elif pid == 22:   # photon → ECAL
            rel_res = self.ecal_energy_resolution(p.E)
            scale = 1 + rng.normal(0, rel_res)
            return Particle4Vector(p.E*scale, p.px*scale, p.py*scale, p.pz*scale,
                                   p.pid, p.charge)
        else:   # hadron → HCAL
            rel_res = self.hcal_energy_resolution(p.E)
            scale = 1 + rng.normal(0, rel_res)
            return Particle4Vector(p.E*scale, p.px*scale, p.py*scale, p.pz*scale,
                                   p.pid, p.charge)

    def is_accepted(self, p: Particle4Vector) -> bool:
        """Basic detector acceptance cut."""
        return (abs(p.eta) < self.eta_max_tracker and p.pt > 0.5)

    def reconstruct_Z_mass(self, events: List[List[Particle4Vector]],
                            rng: np.random.Generator) -> np.ndarray:
        """Reconstruct di-muon invariant mass from Z events."""
        masses = []
        for evt in events:
            if len(evt) < 2:
                continue
            p1_sm = self.smear_particle(evt[0], rng)
            p2_sm = self.smear_particle(evt[1], rng)
            if not (self.is_accepted(p1_sm) and self.is_accepted(p2_sm)):
                continue
            # Invariant mass
            m2 = ((p1_sm.E + p2_sm.E)**2 -
                   (p1_sm.px + p2_sm.px)**2 -
                   (p1_sm.py + p2_sm.py)**2 -
                   (p1_sm.pz + p2_sm.pz)**2)
            if m2 > 0:
                masses.append(np.sqrt(m2))
        return np.array(masses)
