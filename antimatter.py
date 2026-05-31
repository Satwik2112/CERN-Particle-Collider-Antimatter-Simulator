"""
=============================================================================
  CERN Antimatter Simulation Module
=============================================================================
  Covers (all grounded in the document):
    1.  Antiproton production cross-section  (p + p → p + p + p + p̄)
    2.  Target yield model  (26 GeV PS beam on Iridium target)
    3.  Antiproton capture & phase-space acceptance
    4.  Antiproton Decelerator (AD)  5.3 MeV → 100 keV deceleration
    5.  ELENA ring                  100 keV → ~5.3 keV
    6.  Stochastic cooling          (bandwidth, mixing, Schottky noise)
    7.  Electron cooling            (LEIR / AD)
    8.  Phase-space evolution under cooling (emittance shrinkage)
    9.  Antihydrogen formation      p̄ + e⁺ → H̄  (3-body recombination)
   10.  Antihydrogen trapping       (magnetic bottle / Ioffe trap)
   11.  Positron source             (²²Na β⁺ decay)
   12.  Antimatter annihilation     (p̄ + p → pions)
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from physics_core import (gamma_factor, beta_factor, momentum_eV,
                           beam_rigidity, e_charge, c, m_proton)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
m_pbar      = m_proton          # antiproton mass = proton mass [eV]
m_positron  = 0.511e6           # positron mass [eV]
m_H         = 938.783e6         # hydrogen atom mass [eV]  (≈ m_p + m_e)
r_p         = 1.535e-18         # classical proton radius [m]
sigma_pp_inel = 40e-31          # inelastic pp cross-section ~40 mb [m²] at 26 GeV
k_B         = 8.617e-5          # Boltzmann constant [eV/K]

# AD / ELENA design parameters (CERN)
AD_INJECTION_KE  = 3.57e6       # [eV]  3.57 MeV kinetic energy into AD
AD_EXTRACTION_KE = 5.3e6        # [eV]  nominal extraction 5.3 MeV → but AD goes down
AD_FINAL_KE      = 5.3e6        # [eV]  kinetic energy at AD extraction (to ELENA)
ELENA_FINAL_KE   = 5.3e3        # [eV]  ELENA final kinetic energy ~5.3 keV
AD_CIRCUMFERENCE = 182.4        # [m]
ELENA_CIRCUMFERENCE = 30.4      # [m]

# ──────────────────────────────────────────────────────────────────────────────
# 1. ANTIPROTON PRODUCTION  p + p → p + p + p + p̄
# ──────────────────────────────────────────────────────────────────────────────
def pbar_threshold_energy() -> float:
    """
    Threshold kinetic energy for p̄ production in fixed-target pp collision.
    T_thresh = 6·m_p·c² = 6 × 938.3 MeV ≈ 5.63 GeV
    """
    return 6.0 * m_proton   # [eV]

def pbar_production_cross_section(T_beam_eV: float) -> float:
    """
    Parametric p̄ production cross-section σ(p̄) [m²].
    Based on Buss et al. parametrization near threshold.
    σ ≈ σ₀ · (√s - √s_thresh)^n / s^m
    """
    T_thresh = pbar_threshold_energy()
    if T_beam_eV < T_thresh:
        return 0.0
    # Lab-frame √s
    E_beam = T_beam_eV + m_proton
    s = 2 * m_proton * (E_beam + m_proton)      # [eV²]
    s_thresh = (4 * m_proton)**2                  # [eV²]
    sqrt_s       = np.sqrt(s)
    sqrt_s_thresh = np.sqrt(s_thresh)
    # Empirical parametrization (Tan & Ng 1983, simplified)
    sigma_0 = 0.12e-31   # [m²]  normalisation ~0.12 μb
    n, m_exp = 1.67, 0.5
    ratio = (sqrt_s - sqrt_s_thresh) / (1e9)     # normalise to GeV
    sigma = sigma_0 * ratio**n / (s / s_thresh)**m_exp
    return max(sigma, 0.0)

def pbar_yield_per_proton(T_beam_eV: float,
                           target_material: str = 'Ir',
                           target_thickness_cm: float = 10.0) -> float:
    """
    Number of antiprotons produced per incident proton.
    Y = N_target · σ_prod · (1 - exp(-x/λ_int)) / (A/ρ·N_A)
    Simplified: Y ≈ n_target · σ_prod · thickness
    """
    # Target nuclear interaction length λ_I [g/cm²] and density
    target_props = {
        'Ir': {'rho': 22.56, 'A': 192.2, 'lambda_I': 285.0},  # g/cm², g/mol, g/cm²
        'Cu': {'rho':  8.96, 'A':  63.5, 'lambda_I': 134.9},
        'W':  {'rho': 19.30, 'A': 183.8, 'lambda_I': 185.0},
    }
    props = target_props.get(target_material, target_props['Ir'])
    N_A = 6.022e23   # Avogadro
    # Number density [m⁻³]
    n_target = (props['rho'] * 1e6 * N_A) / props['A']   # [m⁻³]  (rho in g/cm³ → g/m³)
    sigma = pbar_production_cross_section(T_beam_eV)
    thickness_m = target_thickness_cm * 1e-2
    # Geometric factor: p̄ production in forward cone (acceptance ~few %)
    acceptance = 0.04   # ~4% of produced p̄ are captured
    yield_pp = n_target * sigma * thickness_m * acceptance
    return yield_pp

def pbar_per_second(proton_intensity: float = 1.5e13,
                     rep_rate_Hz: float = 0.4,
                     T_beam_eV: float = 26e9) -> float:
    """
    Antiproton production rate [p̄/s].
    proton_intensity: protons per PS pulse
    rep_rate_Hz: PS repetition rate
    """
    Y = pbar_yield_per_proton(T_beam_eV)
    return Y * proton_intensity * rep_rate_Hz

# ──────────────────────────────────────────────────────────────────────────────
# 2. ANTIPROTON DECELERATION CHAIN  (AD + ELENA)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DecelerationStage:
    """One stage in the AD/ELENA deceleration chain."""
    name: str
    KE_in_eV:  float    # kinetic energy at injection [eV]
    KE_out_eV: float    # kinetic energy at extraction [eV]
    circumference: float # [m]
    cooling_method: str  # 'stochastic', 'electron', 'none'

    def gamma_in(self)  -> float:
        return gamma_factor(self.KE_in_eV  + m_pbar, m_pbar)
    def gamma_out(self) -> float:
        return gamma_factor(self.KE_out_eV + m_pbar, m_pbar)
    def beta_in(self)   -> float:
        return beta_factor(self.gamma_in())
    def beta_out(self)  -> float:
        return beta_factor(self.gamma_out())
    def Brho_in(self)   -> float:
        return beam_rigidity(self.KE_in_eV  + m_pbar, m_pbar)
    def Brho_out(self)  -> float:
        return beam_rigidity(self.KE_out_eV + m_pbar, m_pbar)
    def f_rev_in(self)  -> float:
        return self.beta_in()  * c / self.circumference
    def f_rev_out(self) -> float:
        return self.beta_out() * c / self.circumference
    def decel_factor(self) -> float:
        return self.KE_in_eV / self.KE_out_eV

# Full AD → ELENA chain with intermediate cooling plateaus
AD_ELENA_CHAIN = [
    DecelerationStage("AD  injection",       3.57e9,   3.57e9,  AD_CIRCUMFERENCE,   'stochastic'),
    DecelerationStage("AD  cool @ 3.57 GeV", 3.57e9,   3.57e9,  AD_CIRCUMFERENCE,   'stochastic'),
    DecelerationStage("AD  decel → 2 GeV",   3.57e9,   2.0e9,   AD_CIRCUMFERENCE,   'stochastic'),
    DecelerationStage("AD  cool @ 2 GeV",    2.0e9,    2.0e9,   AD_CIRCUMFERENCE,   'stochastic'),
    DecelerationStage("AD  decel → 300 MeV", 2.0e9,    300e6,   AD_CIRCUMFERENCE,   'electron'),
    DecelerationStage("AD  cool @ 300 MeV",  300e6,    300e6,   AD_CIRCUMFERENCE,   'electron'),
    DecelerationStage("AD  decel → 100 MeV", 300e6,    100e6,   AD_CIRCUMFERENCE,   'electron'),
    DecelerationStage("AD  extract",         100e6,    5.3e6,   AD_CIRCUMFERENCE,   'none'),
    DecelerationStage("ELENA inject",        5.3e6,    5.3e6,   ELENA_CIRCUMFERENCE, 'electron'),
    DecelerationStage("ELENA cool",          5.3e6,    5.3e6,   ELENA_CIRCUMFERENCE, 'electron'),
    DecelerationStage("ELENA decel",         5.3e6,    5.3e3,   ELENA_CIRCUMFERENCE, 'electron'),
    DecelerationStage("ELENA extract",       5.3e3,    5.3e3,   ELENA_CIRCUMFERENCE, 'none'),
]

def simulate_deceleration_chain() -> List[dict]:
    """Return physics quantities at each stage of the AD/ELENA chain."""
    results = []
    for stage in AD_ELENA_CHAIN:
        g_out = stage.gamma_out()
        b_out = stage.beta_out()
        results.append({
            'stage':        stage.name,
            'KE_MeV':       stage.KE_out_eV / 1e6,
            'gamma':        g_out,
            'beta':         b_out,
            'Brho_Tm':      stage.Brho_out(),
            'f_rev_kHz':    stage.f_rev_out() / 1e3,
            'cooling':      stage.cooling_method,
        })
    return results

# ──────────────────────────────────────────────────────────────────────────────
# 3. STOCHASTIC COOLING
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class StochasticCooling:
    """
    Stochastic cooling model (van der Meer, 1972).
    The pickup detects beam fluctuations; a kicker corrects them
    after a delay of half a revolution.
    Cooling rate: 1/τ = W/N · (2g - g²·M)
    where W = bandwidth, N = particles, g = gain, M = mixing factor
    """
    bandwidth_Hz:   float = 500e6    # [Hz]  frequency bandwidth of system
    gain:           float = 0.5      # optimal gain g ≈ 1/M for M≫1
    mixing_factor:  float = 2.0      # M: ratio of bandwidth to revolution harmonic spacing
    N_particles:    float = 3e7      # number of stored antiprotons

    def cooling_rate(self) -> float:
        """1/τ_cool [s⁻¹]"""
        W = self.bandwidth_Hz
        N = self.N_particles
        g = self.gain
        M = self.mixing_factor
        return W / N * (2*g - g**2 * M)

    def cooling_time(self) -> float:
        """τ_cool [s]"""
        rate = self.cooling_rate()
        return 1.0 / rate if rate > 0 else np.inf

    def emittance_evolution(self, eps0: float,
                             t_arr: np.ndarray) -> np.ndarray:
        """
        Emittance vs time under stochastic cooling.
        ε(t) = ε₀ · exp(-t/τ_cool)
        Includes noise floor (diffusion limit).
        """
        tau = self.cooling_time()
        eps_floor = eps0 * 0.05   # ~5% irreducible floor (Schottky noise)
        return eps_floor + (eps0 - eps_floor) * np.exp(-t_arr / tau)

    def momentum_spread_evolution(self, dp0: float,
                                   t_arr: np.ndarray) -> np.ndarray:
        """Longitudinal momentum spread Δp/p under stochastic cooling."""
        tau_L = self.cooling_time() * 1.5   # longitudinal is slower
        dp_floor = dp0 * 0.03
        return dp_floor + (dp0 - dp_floor) * np.exp(-t_arr / tau_L)

    def schottky_spectrum(self, f_rev: float,
                           dp_p: float,
                           n_harmonics: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Longitudinal Schottky noise spectrum.
        Power at harmonic n: peak at n·f_rev, width n·η·(Δp/p)·f_rev
        Used to measure momentum spread non-destructively.
        eta = slip factor ≈ 1/γ² for AD energies
        """
        eta = 1.0 / gamma_factor(300e6 + m_pbar, m_pbar)**2
        f_all, P_all = [], []
        for n in range(1, n_harmonics + 1):
            f_center = n * f_rev
            width    = n * abs(eta) * dp_p * f_rev
            f_range  = np.linspace(f_center - 3*width, f_center + 3*width, 50)
            power    = np.exp(-0.5 * ((f_range - f_center) / (width + 1e-10))**2)
            f_all.extend(f_range)
            P_all.extend(power / n)
        return np.array(f_all), np.array(P_all)

# ──────────────────────────────────────────────────────────────────────────────
# 4. ELECTRON COOLING
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ElectronCooling:
    """
    Electron cooling model (Budker, 1966).
    A cold electron beam co-propagates with p̄ beam;
    Coulomb collisions transfer heat from p̄ to electrons.
    Cooling force: F_cool ∝ -v_rel (for |v_rel| < v_thermal_e)
    """
    I_electron_A:   float = 0.5      # [A] electron beam current
    KE_electron_eV: float = 300.0    # [eV] electron kinetic energy (matched to p̄)
    L_cooler_m:     float = 1.5      # [m] cooler section length
    T_electron_K:   float = 1e4      # [K] transverse electron temperature

    def electron_velocity(self) -> float:
        """Electron velocity matching condition (co-moving)."""
        g = gamma_factor(self.KE_electron_eV + 0.511e6, 0.511e6)
        return beta_factor(g) * c

    def v_thermal_electron(self) -> float:
        """Transverse thermal velocity of electron beam."""
        return np.sqrt(k_B * self.T_electron_K * e_charge / (9.109e-31 * c**2)) * c

    def cooling_force_magnitude(self, v_pbar: float) -> float:
        """
        Simplified cooling force [eV/m] on p̄.
        F ∝ n_e · r_e² · ln(Λ) · v_rel / v_e³  for v_rel < v_e
        """
        n_e = self.I_electron_A / (e_charge * self.electron_velocity() * np.pi * (0.01)**2)
        r_e = 2.818e-15   # classical electron radius [m]
        ln_Lambda = 10.0  # Coulomb logarithm
        v_th = self.v_thermal_electron()
        v_rel = abs(v_pbar - self.electron_velocity())
        if v_rel < v_th:
            # Linear regime: F ∝ v_rel
            F = (4 * np.pi * n_e * r_e**2 * (9.109e-31 * c**2) *
                 ln_Lambda * v_rel / v_th**3) * c**2 / e_charge
        else:
            # Log regime: F ∝ 1/v_rel²
            F = (4 * np.pi * n_e * r_e**2 * (9.109e-31 * c**2) *
                 ln_Lambda / v_rel**2) * c**2 / e_charge
        return F * self.L_cooler_m

    def cooling_time_transverse(self, KE_pbar_eV: float,
                                  emittance_m: float) -> float:
        """
        Transverse cooling time τ_⊥ [s].
        τ ≈ (p²/m) / (F_cool · n_turns_per_second)
        """
        g = gamma_factor(KE_pbar_eV + m_pbar, m_pbar)
        b = beta_factor(g)
        p_pbar = momentum_eV(KE_pbar_eV + m_pbar, m_pbar) * e_charge / c
        v_pbar = b * c
        F = self.cooling_force_magnitude(v_pbar)
        if F < 1e-30:
            return np.inf
        tau = p_pbar / (F * e_charge / c) * (b * c / AD_CIRCUMFERENCE)
        return abs(tau)

    def emittance_evolution_ec(self, eps0: float,
                                KE_pbar_eV: float,
                                t_arr: np.ndarray) -> np.ndarray:
        """Emittance under electron cooling: faster than stochastic at low energy."""
        tau = max(self.cooling_time_transverse(KE_pbar_eV, eps0), 0.1)
        eps_floor = eps0 * 0.01   # electron cooling can reach ~1% floor
        return eps_floor + (eps0 - eps_floor) * np.exp(-t_arr / tau)

# ──────────────────────────────────────────────────────────────────────────────
# 5. POSITRON SOURCE  (²²Na β⁺ decay)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PositronSource:
    """
    ²²Na radioactive source for positrons (used in antihydrogen experiments).
    ²²Na → ²²Ne + e⁺ + νₑ    (T½ = 2.6 years, Q = 1.82 MeV)
    """
    activity_Bq:    float = 3.7e7    # [Bq] = 1 μCi
    efficiency:     float = 0.20     # positron capture efficiency
    T_half_years:   float = 2.6      # half-life of ²²Na

    def positron_rate(self) -> float:
        """Positrons per second delivered to trap."""
        return self.activity_Bq * self.efficiency

    def activity_decay(self, t_years: np.ndarray) -> np.ndarray:
        """Activity vs time: A(t) = A₀ · 2^(-t/T½)"""
        return self.activity_Bq * 2**(-t_years / self.T_half_years)

    def positron_energy_spectrum(self, n_pts: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """
        β⁺ energy spectrum from ²²Na decay.
        Fermi-Kurie distribution: N(E) ∝ p·E·(Q-E)²·F(Z,E)
        Q = 1.82 MeV endpoint energy
        """
        Q = 1.82e6   # [eV]
        E = np.linspace(1e3, Q - 1e3, n_pts)
        p = np.sqrt(E**2 + 2 * E * 0.511e6) / c   # momentum [eV/c·s/m... dimensionless]
        # Fermi function F(Z,E) ≈ 1 + α·Z/β  (simplified for Z=10, Ne)
        beta_e = np.sqrt(1 - (0.511e6 / (E + 0.511e6))**2)
        F_fermi = 1 + 0.0073 / (beta_e + 1e-10)
        N = p * (E + 0.511e6) * (Q - E)**2 * F_fermi
        N = np.clip(N, 0, None)
        return E / 1e6, N / N.max()   # [MeV], normalised

# ──────────────────────────────────────────────────────────────────────────────
# 6. ANTIHYDROGEN FORMATION  p̄ + e⁺ → H̄  (3-body recombination)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class AntihydrogenFormation:
    """
    Antihydrogen (H̄) formation via 3-body recombination.
    p̄ + e⁺ + e⁺ → H̄(n) + e⁺   (dominant mechanism at low T)
    Rate: R₃ ∝ n_e² / T^(9/2)   (Stevefelt et al.)
    Also: radiative recombination  p̄ + e⁺ → H̄ + γ   (slower)
    """
    N_pbar:        float = 1e4     # antiprotons in trap
    N_positron:    float = 1e6     # positrons in trap
    T_plasma_K:    float = 15.0    # [K] plasma temperature (very cold)
    plasma_radius: float = 2e-3    # [m] plasma radius

    def positron_density(self) -> float:
        """n_e⁺ [m⁻³]"""
        V = np.pi * self.plasma_radius**2 * 0.02   # L ~ 2 cm
        return self.N_positron / V

    def three_body_rate_coeff(self) -> float:
        """
        3-body recombination rate coefficient [m⁶/s].
        k₃ = C · (e²/kT)^(9/2) · (kT/m_e)^(1/2)
        Stevefelt (1975): k₃ ≈ 5.5×10⁻²⁰ · T^(-9/2)  [m⁶·s⁻¹] (T in K)
        """
        return 5.5e-20 * self.T_plasma_K**(-4.5)

    def radiative_rate_coeff(self) -> float:
        """
        Radiative recombination p̄ + e⁺ → H̄(n) + γ  [m³/s].
        k_rad ≈ 5×10⁻²⁰ / sqrt(T)  [m³/s]  (sum over n)
        """
        return 5e-20 / np.sqrt(self.T_plasma_K)

    def formation_rate_three_body(self) -> float:
        """H̄ production rate [s⁻¹] via 3-body."""
        n_e = self.positron_density()
        k3  = self.three_body_rate_coeff()
        return k3 * n_e**2 * self.N_pbar

    def formation_rate_radiative(self) -> float:
        """H̄ production rate [s⁻¹] via radiative recombination."""
        n_e    = self.positron_density()
        k_rad  = self.radiative_rate_coeff()
        return k_rad * n_e * self.N_pbar

    def n_level_distribution(self, n_max: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Principal quantum number n distribution of freshly formed H̄.
        3-body: dn/dn ∝ n^4 · exp(-E_n/kT)  → high-n Rydberg states
        """
        n_arr  = np.arange(2, n_max + 1, dtype=float)
        E_n    = -13.6 / n_arr**2   # [eV] binding energy
        kT     = k_B * self.T_plasma_K
        # 3-body forms high-n Rydberg states: rate ∝ n^4·exp(-|E_n|/kT) ... clipped
        weight = n_arr**4 * np.exp(-np.abs(E_n) / (kT + 1e-10))
        weight /= weight.sum()
        return n_arr, weight

    def temperature_scan_formation_rate(self,
                                         T_range_K: np.ndarray
                                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Formation rates vs temperature for both mechanisms."""
        r3 , r_rad = [], []
        for T in T_range_K:
            old_T = self.T_plasma_K
            self.T_plasma_K = T
            r3.append(self.formation_rate_three_body())
            r_rad.append(self.formation_rate_radiative())
            self.T_plasma_K = old_T
        return T_range_K, np.array(r3), np.array(r_rad)

# ──────────────────────────────────────────────────────────────────────────────
# 7. ANTIHYDROGEN TRAPPING  (Ioffe–Pritchard magnetic trap)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class IoffeTrap:
    """
    Ioffe–Pritchard magnetic trap for neutral H̄.
    Traps weak-field-seeking hyperfine state.
    Trap depth: ΔE = μ_B · ΔB  (μ_B = Bohr magneton)
    """
    B_min_T: float  = 0.0     # [T] minimum field at trap centre
    B_max_T: float  = 1.0     # [T] maximum field at trap edge
    trap_length_m: float = 0.27  # [m] ALPHA-style trap

    mu_B = 9.274e-24  # Bohr magneton [J/T]

    def trap_depth_K(self) -> float:
        """Trap depth in temperature units [K]."""
        delta_B = self.B_max_T - self.B_min_T
        delta_E = self.mu_B * delta_B   # [J]
        return delta_E / (k_B * e_charge)  # K  (k_B in eV/K, need J/K)

    def trap_depth_eV(self) -> float:
        delta_B = self.B_max_T - self.B_min_T
        return self.mu_B * delta_B / e_charge   # [eV]

    def trapping_fraction(self, T_Hbar_K: float) -> float:
        """
        Fraction of H̄ with kinetic energy < trap depth.
        Assumes Maxwell-Boltzmann: f = 1 - exp(-Δ/kT) for Δ ≫ kT
        """
        depth_K = self.trap_depth_K()
        ratio   = depth_K / (T_Hbar_K + 1e-30)
        # Fraction in 3D MB below trap depth (numerical integral approximation)
        x = ratio
        frac = (2 / np.sqrt(np.pi)) * (np.sqrt(x) * np.exp(-x) +
                0.5 * np.sqrt(np.pi) * (1 - 2*x/3))
        return np.clip(frac, 0, 1)

    def field_profile(self, n_pts: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Magnetic field along z-axis of Ioffe trap."""
        z = np.linspace(-self.trap_length_m/2, self.trap_length_m/2, n_pts)
        # Ioffe-Pritchard: B(z) = sqrt(B_min² + (B'/z)²)  simplified
        B_grad = (self.B_max_T - self.B_min_T) / (self.trap_length_m/2)**2
        B_z = np.sqrt(self.B_min_T**2 + (B_grad * z**2)**2 + 1e-10)
        B_z = np.clip(B_z, self.B_min_T, self.B_max_T)
        return z * 100, B_z   # [cm], [T]

# ──────────────────────────────────────────────────────────────────────────────
# 8. ANNIHILATION  p̄ + p → π mesons
# ──────────────────────────────────────────────────────────────────────────────
def pbar_annihilation_products(n_events: int = 5000,
                                rng: np.random.Generator = None
                                ) -> Dict[str, np.ndarray]:
    """
    Simulate p̄ + p annihilation at rest → charged + neutral pions.
    Average: ~3 π⁺, ~3 π⁻, ~2 π⁰  (from bubble chamber data)
    π⁰ → γγ immediately
    """
    if rng is None:
        rng = np.random.default_rng(1234)

    m_pi_ch  = 139.6e-3   # GeV
    m_pi_0   = 135.0e-3   # GeV
    sqrt_s   = 2 * m_proton / 1e9  # GeV (annihilation at rest)
    results  = {'pi_plus': [], 'pi_minus': [], 'pi0': [],
                'gamma': [], 'vertex_r_mm': []}

    for _ in range(n_events):
        # Multiplicity fluctuations (Poisson around mean)
        n_ch = max(2, rng.poisson(6))   # total charged
        n_0  = max(0, rng.poisson(2))   # neutrals
        n_plus  = n_ch // 2
        n_minus = n_ch - n_plus

        # Isotropic momenta constrained by energy conservation (simplified)
        def random_pion_mom(m_GeV, n_pions):
            moms = []
            if n_pions == 0:
                return moms
            # Each pion gets ~equal share of kinetic energy
            E_avail = (sqrt_s - n_pions * m_GeV) / n_pions
            E_avail = max(E_avail, 0.001)
            for _ in range(n_pions):
                p_mag = np.sqrt(max(E_avail**2 + 2*E_avail*m_GeV, 0.0))
                costh = rng.uniform(-1, 1)
                phi   = rng.uniform(0, 2*np.pi)
                sinth = np.sqrt(1 - costh**2)
                moms.append(np.array([p_mag*sinth*np.cos(phi),
                                       p_mag*sinth*np.sin(phi),
                                       p_mag*costh]))
            return moms

        results['pi_plus'].extend(random_pion_mom(m_pi_ch, n_plus))
        results['pi_minus'].extend(random_pion_mom(m_pi_ch, n_minus))
        pi0_moms = random_pion_mom(m_pi_0, n_0)
        results['pi0'].extend(pi0_moms)

        # π⁰ → γγ  (immediately, c·τ ~ 25 nm)
        for p0 in pi0_moms:
            p_mag = np.linalg.norm(p0)
            E_pi0 = np.sqrt(p_mag**2 + m_pi_0**2)
            # Decay isotropically in rest frame, boost to lab
            costh_cm = rng.uniform(-1, 1)
            phi_cm   = rng.uniform(0, 2*np.pi)
            E_gamma_cm = m_pi_0 / 2
            pg_cm = E_gamma_cm * np.array([
                np.sqrt(1-costh_cm**2)*np.cos(phi_cm),
                np.sqrt(1-costh_cm**2)*np.sin(phi_cm), costh_cm])
            beta_z = p0[2] / E_pi0 if E_pi0 > 0 else 0
            g_boost = E_pi0 / (m_pi_0 + 1e-10)
            E_g_lab = g_boost * (E_gamma_cm + beta_z * pg_cm[2])
            results['gamma'].append(E_g_lab)

        # Annihilation vertex (displaced from beam if in-flight)
        r_vertex = rng.exponential(0.1)   # mm, interaction depth
        results['vertex_r_mm'].append(r_vertex)

    for key in results:
        results[key] = np.array(results[key]) if results[key] else np.array([])
    return results

# ──────────────────────────────────────────────────────────────────────────────
# 9. FULL ANTIMATTER SIMULATION RUNNER
# ──────────────────────────────────────────────────────────────────────────────
def run_antimatter_simulation(seed: int = 777) -> dict:
    """Run all antimatter sub-simulations, return data dict for plotting."""
    rng = np.random.default_rng(seed)

    # --- Production yield vs beam energy
    T_arr = np.linspace(pbar_threshold_energy(), 30e9, 300)
    sigma_arr = np.array([pbar_production_cross_section(T) for T in T_arr])
    yield_arr = np.array([pbar_yield_per_proton(T) for T in T_arr])

    # --- Deceleration chain
    decel_chain = simulate_deceleration_chain()

    # --- Stochastic cooling evolution (at 3.57 GeV injection)
    sc = StochasticCooling(N_particles=3e7, bandwidth_Hz=500e6)
    t_cool = np.linspace(0, 200, 500)   # seconds
    eps_sc = sc.emittance_evolution(500e-6, t_cool)   # start 500 μm·rad
    dp_sc  = sc.momentum_spread_evolution(0.02, t_cool)

    # --- Electron cooling (at 100 MeV in AD)
    ec = ElectronCooling(KE_electron_eV=50e3, L_cooler_m=1.5)
    t_ec = np.linspace(0, 30, 300)
    eps_ec = ec.emittance_evolution_ec(200e-6, 100e6, t_ec)

    # --- Schottky spectrum
    f_rev_300MeV = beta_factor(gamma_factor(300e6+m_pbar, m_pbar)) * c / AD_CIRCUMFERENCE
    f_sch, P_sch = sc.schottky_spectrum(f_rev_300MeV, 0.005, n_harmonics=15)

    # --- Positron source
    ps  = PositronSource()
    E_pos, N_pos = ps.positron_energy_spectrum()
    t_years = np.linspace(0, 15, 200)
    A_decay = ps.activity_decay(t_years)

    # --- Antihydrogen formation
    ah = AntihydrogenFormation(T_plasma_K=15.0)
    T_scan = np.logspace(0.3, 3, 200)   # 2 K to 1000 K
    T_s, r3_s, rrad_s = ah.temperature_scan_formation_rate(T_scan)
    n_arr, n_weight = ah.n_level_distribution()

    # --- Ioffe trap
    trap = IoffeTrap(B_min_T=0.0, B_max_T=1.0)
    z_trap, B_trap = trap.field_profile()
    T_hbar_range = np.linspace(0.1, 5.0, 200)
    trap_frac = np.array([trap.trapping_fraction(T) for T in T_hbar_range])

    # --- Annihilation products
    anni = pbar_annihilation_products(3000, rng)
    pip_pt = np.array([np.sqrt(p[0]**2+p[1]**2) for p in anni['pi_plus']])
    pim_pt = np.array([np.sqrt(p[0]**2+p[1]**2) for p in anni['pi_minus']])
    gamma_E = anni['gamma']
    vertex_r = anni['vertex_r_mm']

    return {
        'T_arr': T_arr, 'sigma_arr': sigma_arr, 'yield_arr': yield_arr,
        'decel_chain': decel_chain,
        't_cool': t_cool, 'eps_sc': eps_sc, 'dp_sc': dp_sc,
        't_ec': t_ec, 'eps_ec': eps_ec,
        'f_sch': f_sch, 'P_sch': P_sch,
        'E_pos': E_pos, 'N_pos': N_pos,
        't_years': t_years, 'A_decay': A_decay,
        'T_scan': T_scan, 'r3_scan': r3_s, 'rrad_scan': rrad_s,
        'n_arr': n_arr, 'n_weight': n_weight,
        'z_trap': z_trap, 'B_trap': B_trap,
        'T_hbar': T_hbar_range, 'trap_frac': trap_frac,
        'pip_pt': pip_pt, 'pim_pt': pim_pt,
        'gamma_E': gamma_E, 'vertex_r': vertex_r,
        'pbar_rate_per_s': pbar_per_second(),
        'ah_3body_rate': ah.formation_rate_three_body(),
        'ah_rad_rate':   ah.formation_rate_radiative(),
        'trap_depth_eV': trap.trap_depth_eV(),
        'stochastic_tau': sc.cooling_time(),
    }
