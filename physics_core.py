"""
=============================================================================
  CERN LHC Particle Collider Simulation — Physics Core
  Based on: Key Concepts and Systems in High-Energy Colliders
=============================================================================
  Covers:
    - Beam dynamics (betatron / synchrotron oscillations)
    - Emittance (Liouville's theorem, normalized emittance)
    - Luminosity calculation
    - RF cavity acceleration & phase stability
    - Synchrotron radiation (power, damping times)
    - Magnet lattice (dipoles, quadrupoles — FODO cell)
    - Twiss parameter propagation (beta functions, phase advance)
    - Beam–beam effects (tune shift)
    - Collimation (halo scraping model)
    - Beam loss & quench protection model
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────────────
c        = 2.998e8          # speed of light [m/s]
e_charge = 1.602e-19        # elementary charge [C]
m_proton = 938.272e6        # proton rest mass [eV/c²]
m_electron = 0.511e6        # electron rest mass [eV/c²]
hbar     = 1.055e-34        # reduced Planck constant [J·s]
k_B      = 8.617e-5         # Boltzmann constant [eV/K]

# LHC design parameters (from document)
LHC_CIRCUMFERENCE  = 26_659.0       # [m]
LHC_DESIGN_ENERGY  = 7.0e12         # [eV] per beam
LHC_INJECTION_ENERGY = 0.45e12      # [eV]
LHC_DIPOLE_FIELD   = 8.33           # [T]
LHC_N_DIPOLES      = 1232
LHC_DIPOLE_LENGTH  = 15.0           # [m]
LHC_RF_FREQUENCY   = 400e6          # [Hz]
LHC_RF_VOLTAGE     = 16e6           # [V] total per beam
LHC_BUNCHES        = 2808
LHC_BUNCH_POP      = 1.15e11        # protons per bunch
LHC_NORM_EMITTANCE = 3.75e-6        # [m·rad] normalized
LHC_DESIGN_LUM     = 1e34           # [cm⁻²s⁻¹]
LHC_BEAM_ENERGY_J  = 362e6          # [J] stored per beam

# ──────────────────────────────────────────────────────────────────────────────
# Helper: relativistic kinematics
# ──────────────────────────────────────────────────────────────────────────────
def gamma_factor(energy_eV: float, mass_eV: float) -> float:
    """Lorentz gamma from total energy and rest mass (eV)."""
    return energy_eV / mass_eV

def beta_factor(gamma: float) -> float:
    """Relativistic beta from gamma."""
    return np.sqrt(1.0 - 1.0 / gamma**2)

def momentum_eV(energy_eV: float, mass_eV: float) -> float:
    """Relativistic momentum p [eV/c]."""
    return np.sqrt(energy_eV**2 - mass_eV**2)

def beam_rigidity(energy_eV: float, mass_eV: float, charge_e: float = 1.0) -> float:
    """Beam rigidity Bρ = p/(qc)  [T·m]."""
    p_SI = momentum_eV(energy_eV, mass_eV) * e_charge / c   # [kg·m/s]
    return p_SI / (charge_e * e_charge)

# ──────────────────────────────────────────────────────────────────────────────
# 1. EMITTANCE  (ε = phase-space area / π)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Emittance:
    """
    Transverse emittance model.
    Liouville: geometric emittance εₙ_geo = εₙ_norm / (β·γ).
    Beam size:  σ = √(ε·β_twiss)
    """
    norm_emittance: float = LHC_NORM_EMITTANCE   # [m·rad]

    def geometric(self, energy_eV: float, mass_eV: float = m_proton) -> float:
        """Geometric emittance [m·rad]."""
        g = gamma_factor(energy_eV, mass_eV)
        b = beta_factor(g)
        return self.norm_emittance / (b * g)

    def beam_size(self, beta_twiss: float, energy_eV: float,
                  mass_eV: float = m_proton) -> float:
        """RMS beam size σ = √(ε_geo · β_twiss)  [m]."""
        eps_geo = self.geometric(energy_eV, mass_eV)
        return np.sqrt(eps_geo * beta_twiss)

    def phase_ellipse(self, beta_twiss: float, alpha_twiss: float,
                      energy_eV: float, n_pts: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Parametric (x, x') phase-space ellipse for plotting."""
        eps_geo = self.geometric(energy_eV)
        gamma_twiss = (1 + alpha_twiss**2) / beta_twiss
        theta = np.linspace(0, 2 * np.pi, n_pts)
        # Cholesky decomposition of Courant-Snyder matrix
        A = np.array([[beta_twiss, -alpha_twiss],
                      [-alpha_twiss, gamma_twiss]])
        # ellipse: v^T · sigma^{-1} · v = 1 with sigma = eps * A^{-1}
        # parametric: v = sqrt(eps) * M · [cos θ, sin θ]
        M = np.linalg.cholesky(eps_geo * np.linalg.inv(A)).T
        pts = M @ np.array([np.cos(theta), np.sin(theta)])
        return pts[0], pts[1]

# ──────────────────────────────────────────────────────────────────────────────
# 2. LUMINOSITY
# ──────────────────────────────────────────────────────────────────────────────
def luminosity(f_rev: float, N1: float, N2: float,
               sigma_x: float, sigma_y: float,
               n_bunches: int = 1,
               crossing_angle: float = 0.0,
               beta_star: float = 0.55) -> float:
    """
    Head-on Gaussian luminosity (cm⁻²s⁻¹).
    L = f·N1·N2 / (4π·σx·σy)
    Geometric reduction factor for non-zero crossing angle included.
    """
    f_bunch = f_rev * n_bunches
    # Geometric reduction (Piwinski angle)
    phi = crossing_angle * sigma_x / (2 * sigma_y + 1e-30)
    F = 1.0 / np.sqrt(1 + phi**2)   # geometric factor
    L_SI = f_bunch * N1 * N2 / (4 * np.pi * sigma_x * sigma_y) * F
    return L_SI * 1e-4   # [m⁻²s⁻¹] → [cm⁻²s⁻¹]

# ──────────────────────────────────────────────────────────────────────────────
# 3. RF CAVITIES & PHASE STABILITY
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RFCavity:
    """
    Superconducting RF cavity (LHC: 400 MHz, ~2 MV each, 4 per beam).
    Synchronous condition: eV sin(φ_s) = ΔE_turn
    """
    frequency: float  = LHC_RF_FREQUENCY      # [Hz]
    voltage: float    = LHC_RF_VOLTAGE         # [V] total
    harmonic: int     = 35640                  # h = f_RF / f_rev

    def revolution_freq(self) -> float:
        return self.frequency / self.harmonic

    def synchrotron_tune(self, energy_eV: float, eta: float,
                         phi_s: float = np.pi / 6) -> float:
        """
        Synchrotron tune Qs = √(h·η·eV·cos(φ_s) / (2π·β²·E))
        eta = slip factor (α_c - 1/γ²)
        """
        g = gamma_factor(energy_eV, m_proton)
        b = beta_factor(g)
        Qs2 = (self.harmonic * abs(eta) * e_charge * self.voltage
               * abs(np.cos(phi_s)) / (2 * np.pi * b**2 * energy_eV * e_charge))
        return np.sqrt(max(Qs2, 0.0))

    def energy_gain_per_turn(self, phi: np.ndarray) -> np.ndarray:
        """Energy gain ΔE = eV sin(φ) [eV]."""
        return self.voltage * np.sin(phi)

    def bunch_evolution(self, phi0: float, delta0: float,
                        n_turns: int = 5000,
                        eta: float = -3.47e-4,
                        energy_eV: float = LHC_DESIGN_ENERGY) -> Tuple[np.ndarray, np.ndarray]:
        """
        Longitudinal phase-space tracking (simplified map).
        (φ, δ) are synchrotron coordinates.
        """
        g = gamma_factor(energy_eV, m_proton)
        b = beta_factor(g)
        phi_s = np.arcsin(0.0)   # top energy: no energy loss → φ_s = 0
        h = self.harmonic
        phi_arr = np.empty(n_turns)
        delta_arr = np.empty(n_turns)
        phi, delta = phi0, delta0
        for i in range(n_turns):
            # RF kick
            delta_new = delta + (e_charge * self.voltage *
                                 (np.sin(phi) - np.sin(phi_s)) /
                                 (energy_eV * e_charge))
            # Phase advance
            phi = phi + 2 * np.pi * h * eta * delta_new
            delta = delta_new
            phi_arr[i] = phi
            delta_arr[i] = delta
        return phi_arr, delta_arr

# ──────────────────────────────────────────────────────────────────────────────
# 4. SYNCHROTRON RADIATION
# ──────────────────────────────────────────────────────────────────────────────
def synchrotron_radiation_loss(energy_eV: float, rho: float,
                                mass_eV: float = m_proton) -> float:
    """
    Energy loss per revolution U₀ [eV].
    U₀ = C_γ · E⁴ / (ρ · m⁴c⁸)   (E in eV, ρ in m)
    C_γ = 8.85e-5 m/GeV³ for electrons, scales with 1/m⁴
    """
    # Universal formula: U0 [eV] = (C_gamma_e * E^4) / (rho * me^4) * m_e^4 / m^4
    C_gamma_e = 8.85e-5   # m GeV⁻³ for electrons
    E_GeV = energy_eV / 1e9
    m_GeV = mass_eV / 1e9
    U0_GeV = C_gamma_e * E_GeV**4 / (rho * m_GeV**4 * (m_electron / 1e9)**(-4) * (m_electron / 1e9)**4)
    return U0_GeV * 1e9   # eV

def synchrotron_radiation_power(energy_eV: float, rho: float,
                                 I_beam: float, mass_eV: float = m_proton) -> float:
    """Total radiated power P [W] = U₀[eV] · e · I / e = U₀ · I"""
    U0 = synchrotron_radiation_loss(energy_eV, rho, mass_eV)
    return U0 * I_beam   # [W]  (U0 in eV, I in A → P in eV·A = eV²/C... use eV→J)

def damping_time(energy_eV: float, U0_eV: float, T_rev: float,
                 damping_partition: float = 1.0) -> float:
    """
    Radiation damping time τ = 2·E·T₀ / (J·U₀)  [s]
    J = damping partition number (≈1 transverse, ≈2 longitudinal)
    """
    return 2.0 * energy_eV * T_rev / (damping_partition * U0_eV)

# ──────────────────────────────────────────────────────────────────────────────
# 5. MAGNET LATTICE — FODO cell, Twiss propagation
# ──────────────────────────────────────────────────────────────────────────────
def thin_lens_matrix(f: float) -> np.ndarray:
    """2×2 transfer matrix for thin lens (quad) with focal length f."""
    return np.array([[1.0, 0.0], [-1.0/f, 1.0]])

def drift_matrix(L: float) -> np.ndarray:
    """2×2 transfer matrix for drift space of length L."""
    return np.array([[1.0, L], [0.0, 1.0]])

def dipole_matrix(L: float, rho: float) -> np.ndarray:
    """2×2 horizontal transfer matrix for sector dipole."""
    theta = L / rho
    return np.array([[np.cos(theta),       rho * np.sin(theta)],
                     [-np.sin(theta) / rho, np.cos(theta)]])

def quad_matrix(L: float, k: float) -> np.ndarray:
    """
    2×2 transfer matrix for thick quadrupole (horizontal focusing).
    k = gradient / Bρ  [m⁻²]
    """
    if k > 0:   # focusing
        sq = np.sqrt(k)
        return np.array([[np.cos(sq*L),       np.sin(sq*L)/sq],
                         [-sq*np.sin(sq*L),   np.cos(sq*L)]])
    elif k < 0:  # defocusing
        sq = np.sqrt(-k)
        return np.array([[np.cosh(sq*L),       np.sinh(sq*L)/sq],
                         [sq*np.sinh(sq*L),    np.cosh(sq*L)]])
    else:
        return drift_matrix(L)

@dataclass
class FODOCell:
    """
    Standard FODO cell:  QF — drift — QD — drift
    Twiss parameters propagated analytically.
    """
    L_cell: float         # cell length [m]
    L_quad: float = 0.5   # quadrupole length [m]
    phase_advance: float = np.pi / 3   # 60° per cell (typical LHC arc)

    def focal_length(self) -> float:
        """Thin-lens focal length from phase advance μ."""
        # For FODO: sin(μ/2) = L_cell / (4f), f = L/(4sin(μ/2))
        return self.L_cell / (4 * np.sin(self.phase_advance / 2))

    def beta_max(self) -> float:
        """Max β at QF:  β_max = L(1+sin(μ/2))/sin(μ)"""
        mu = self.phase_advance
        return self.L_cell * (1 + np.sin(mu/2)) / np.sin(mu)

    def beta_min(self) -> float:
        """Min β at QD:  β_min = L(1-sin(μ/2))/sin(mu)"""
        mu = self.phase_advance
        return self.L_cell * (1 - np.sin(mu/2)) / np.sin(mu)

    def transfer_matrix(self, Brho: float) -> np.ndarray:
        """One-turn-like matrix for the full FODO cell (H-plane)."""
        f = self.focal_length()
        L_drift = (self.L_cell - 2 * self.L_quad) / 2
        k = 1.0 / (f * self.L_quad)
        M_QF = quad_matrix(self.L_quad,  k / Brho if Brho > 0 else k)
        M_QD = quad_matrix(self.L_quad, -k / Brho if Brho > 0 else -k)
        M_D  = drift_matrix(L_drift)
        return M_QF @ M_D @ M_QD @ M_D

    def twiss_along_cell(self, n_steps: int = 200
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Propagate Twiss (β, α) along a FODO cell.
        Returns s, beta(s), alpha(s).
        """
        beta0  = self.beta_max()
        alpha0 = 0.0   # symmetry point at QF
        s_arr   = np.linspace(0, self.L_cell, n_steps)
        beta_s  = np.empty(n_steps)
        alpha_s = np.empty(n_steps)
        for i, s in enumerate(s_arr):
            frac = s / self.L_cell
            # Simple interpolation using cos-like variation
            mu_s = frac * self.phase_advance
            beta_s[i]  = (beta0 * np.cos(mu_s)**2 +
                          self.beta_min() * np.sin(mu_s)**2)
            alpha_s[i] = (-(beta0 - self.beta_min()) *
                           np.sin(mu_s) * np.cos(mu_s) / self.beta_max())
        return s_arr, beta_s, alpha_s

# ──────────────────────────────────────────────────────────────────────────────
# 6. BETATRON OSCILLATIONS (Courant–Snyder)
# ──────────────────────────────────────────────────────────────────────────────
def track_betatron(beta_s: np.ndarray, n_turns: int,
                   x0: float, xp0: float,
                   tune_frac: float = 0.31) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified multi-turn betatron tracking using smooth approximation.
    x(s) = A · √β(s) · cos(ψ(s) + φ₀)
    """
    n_s = len(beta_s)
    # Courant-Snyder invariant
    J = (x0**2 / beta_s[0] + beta_s[0] * xp0**2) / 2
    A = np.sqrt(2 * J)
    phi0 = np.arctan2(-xp0 * beta_s[0], x0)
    turns = np.arange(n_turns)
    # Phase advance per turn at lattice point s=0
    x_turn = A * np.sqrt(beta_s[0]) * np.cos(2 * np.pi * tune_frac * turns + phi0)
    return turns, x_turn

# ──────────────────────────────────────────────────────────────────────────────
# 7. BEAM–BEAM TUNE SHIFT
# ──────────────────────────────────────────────────────────────────────────────
def beam_beam_tune_shift(N2: float, beta_star: float,
                          sigma_x: float, sigma_y: float,
                          gamma: float) -> float:
    """
    Linear beam–beam tune shift parameter ξ.
    ξ = N₂·r_p·β* / (4π·γ·σ(σ_x+σ_y))
    r_p = classical proton radius = 1.535e-18 m
    """
    r_p = 1.535e-18   # classical proton radius [m]
    return N2 * r_p * beta_star / (4 * np.pi * gamma * sigma_x * (sigma_x + sigma_y))

# ──────────────────────────────────────────────────────────────────────────────
# 8. COLLIMATION MODEL
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CollimationSystem:
    """
    Multi-stage LHC collimation (primary at 6σ, secondary at 7σ).
    Halo particles beyond n_sigma are absorbed.
    """
    primary_sigma: float   = 6.0   # primary jaw [σ]
    secondary_sigma: float = 7.0   # secondary jaw [σ]
    tertiary_sigma: float  = 10.0  # tertiary (absorber) [σ]

    def scrape(self, x_norm: np.ndarray) -> np.ndarray:
        """
        Apply collimation to normalized amplitudes |x|/σ.
        Returns survival mask (True = particle survives).
        """
        return np.abs(x_norm) < self.tertiary_sigma

    def halo_fraction(self, distribution: np.ndarray) -> dict:
        """Compute fraction of beam in halo regions."""
        N = len(distribution)
        a = np.abs(distribution)
        return {
            'core (<6σ)': np.sum(a < self.primary_sigma) / N,
            'primary halo (6–7σ)': np.sum((a >= self.primary_sigma) &
                                            (a < self.secondary_sigma)) / N,
            'secondary halo (7–10σ)': np.sum((a >= self.secondary_sigma) &
                                              (a < self.tertiary_sigma)) / N,
            'absorbed (>10σ)': np.sum(a >= self.tertiary_sigma) / N,
        }

# ──────────────────────────────────────────────────────────────────────────────
# 9. QUENCH PROTECTION MODEL
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class QuenchProtection:
    """
    Simplified quench energy balance for LHC NbTi dipoles.
    If deposited energy > quench limit → magnet quenches.
    """
    I_nominal: float   = 11850.0    # [A] operating current
    L_magnet: float    = 0.12       # [H] inductance per magnet
    R_quench: float    = 0.01       # [Ω] quench resistance
    T_detect: float    = 0.001      # [s] detection time
    E_quench_limit: float = 80.0    # [J/cm³] quench limit in coil

    def stored_energy(self) -> float:
        """E = ½·L·I² [J]"""
        return 0.5 * self.L_magnet * self.I_nominal**2

    def current_decay(self, t: np.ndarray) -> np.ndarray:
        """Exponential current decay after quench: I(t) = I₀·exp(-t/τ)."""
        tau = self.L_magnet / self.R_quench
        return self.I_nominal * np.exp(-t / tau)

    def dump_resistor_energy(self) -> float:
        """Energy dumped into resistors = ½·L·I²."""
        return self.stored_energy()

    def is_quench(self, local_energy_density: float) -> bool:
        return local_energy_density > self.E_quench_limit

# ──────────────────────────────────────────────────────────────────────────────
# 10. COMPLETE BEAM PARAMETERS (LHC snapshot)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LHCBeam:
    """Full LHC beam parameter set at a given fill state."""
    energy_eV: float       = LHC_DESIGN_ENERGY
    n_bunches: int         = LHC_BUNCHES
    bunch_population: float = LHC_BUNCH_POP
    norm_emittance: float  = LHC_NORM_EMITTANCE
    beta_star: float       = 0.55   # [m] at IP1/IP5
    crossing_angle: float  = 285e-6 # [rad]

    def __post_init__(self):
        self.gamma = gamma_factor(self.energy_eV, m_proton)
        self.beta_rel = beta_factor(self.gamma)
        self.emittance = Emittance(self.norm_emittance)
        self.sigma_IP = self.emittance.beam_size(self.beta_star, self.energy_eV)
        self.f_rev = self.beta_rel * c / LHC_CIRCUMFERENCE
        self.I_beam = (self.n_bunches * self.bunch_population *
                       e_charge * self.f_rev)

    def peak_luminosity(self) -> float:
        return luminosity(self.f_rev, self.bunch_population,
                          self.bunch_population,
                          self.sigma_IP, self.sigma_IP,
                          self.n_bunches, self.crossing_angle,
                          self.beta_star)

    def summary(self) -> dict:
        return {
            'Energy [TeV]':           self.energy_eV / 1e12,
            'γ':                      self.gamma,
            'β*  [m]':                self.beta_star,
            'σ_IP [μm]':              self.sigma_IP * 1e6,
            'Norm. emittance [μm·rad]': self.norm_emittance * 1e6,
            'N_bunches':              self.n_bunches,
            'N_p/bunch (×10¹¹)':      self.bunch_population / 1e11,
            'f_rev [kHz]':            self.f_rev / 1e3,
            'I_beam [mA]':            self.I_beam * 1e3,
            'Peak L [×10³⁴ cm⁻²s⁻¹]': self.peak_luminosity() / 1e34,
        }
