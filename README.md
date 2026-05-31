# CERN Particle Collider & Antimatter Simulator

An interactive, high-fidelity scientific simulator modeling the advanced physics processes of the **Large Hadron Collider (LHC)**, the **Antimatter Decelerator (AD)**, and the **ELENA (Extra Low ENergy Antiproton)** facilities at CERN.

The simulator bridges mathematical models of accelerator optics, relativistic beam dynamics, quantum chromodynamics (QCD) parton collisions, and subatomic particle detector tracking with modern, real-time interactive visualizations.

---

## 🔬 Core Physics Simulations

### 1. Large Hadron Collider (LHC) & Beam Dynamics
*   **FODO Lattice & Betatron Tracking**: Models transverse beam focusing and tracking through alternating quadrupole configurations (FODO cells), computing phase space trajectories, tune shifts, and beam emittance growth.
*   **RF Acceleration & Cavity Cavitation**: Tracks longitudinal acceleration, phase-slip factors, and synchrotron radiation losses.
*   **Beam-Beam Interactions**: Computes luminosity ($L$), crossing-angle reduction factors, and beam-beam tune shifts.
*   **Collimation & Machine Protection**: Simulates primary/secondary multi-stage collimator intercepts and quench protection thresholds for superconducting magnets.
*   **Monte Carlo PP Event Generator**: Performs proton-proton collision simulations using parton distribution function (PDF) valence quark sampling and ATLAS-like tracker/calorimeter detector energy deposit mapping.

### 2. Antimatter Decelerator (AD) & Antihydrogen Synthesis
*   **Antiproton Production & Yield**: Simulates high-energy proton collisions on heavy targets (e.g., Iridium/Tungsten) calculating cross-sections and antiproton conversion efficiency.
*   **Deceleration & Beam Cooling**: Features physical models of **Stochastic Cooling** (bandwidth-limited thermal noise mitigation) and **Electron Cooling** (momentum transfer with a cold co-moving electron beam) to reduce beam emittance.
*   **Antihydrogen (Hbar) Recombination**: Models positron target cooling and three-body recombination ($p^- + e^+ + e^+ \rightarrow \bar{H} + e^+$) as a function of plasma temperatures.
*   **Ioffe-Pritchard Magnetic Trap**: Computes superconducting magnetic bottle potential wells, trapping efficiency, and quantum transition spectroscopy.
*   **Annihilation Vertex Reconstruction**: Maps the annihilation coordinates of unstable antimatter interactions.

---

## 🛠 Architecture & Tech Stack

The application employs a decoupled client-server architecture:

```
                  ┌──────────────────────────────────────────────┐
                  │                 Web Browser                  │
                  │  (index.html, styles.css, app.js, Chart.js)  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                HTTP POST / JSON
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │               FastAPI Server                 │
                  │             (server.py, Port 8001)           │
                  └──────────────────────┬───────────────────────┘
                                         │
                     ┌───────────────────┼───────────────────┐
                     ▼                   ▼                   ▼
          ┌─────────────────────┐┌──────────────┐┌──────────────────────┐
          │  physics_core.py    ││ antimatter.py││   detector_sim.py    │
          │  LHC Beam Optics,   ││ Deceleration,││ PP Event Generator,  │
          │  FODO, RF Cavities  ││ Trap Physics ││ Injection, Detector │
          └─────────────────────┘└──────────────┘└──────────────────────┘
```

*   **Backend (Python)**:
    *   **FastAPI**: Asynchronous web API layer.
    *   **NumPy & SciPy**: High-performance scientific computation, matrix physics tracking, and Monte Carlo sampling.
    *   **Pydantic**: Robust data validation schemas for physics parameters.
*   **Frontend (HTML5 / Vanilla CSS3 / JavaScript)**:
    *   **Chart.js**: Dynamic, real-time graph rendering of beam profiles, energy ramps, detector calorimeters, and cooling efficiency curves.
    *   **Vanilla CSS**: Premium dark-themed, glassmorphic scientific dashboard design optimized for clear data hierarchy.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### 1. Install Backend Dependencies

Navigate to the project root directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI Backend Server

Start the FastAPI application using the `uvicorn` ASGI server:

```bash
python3 -m uvicorn server:app --port 8001 --reload
```

The API will be available at `http://127.0.0.1:8001`. You can view the automatically generated interactive API documentation (Swagger UI) at `http://127.0.0.1:8001/docs`.

### 3. Run the Frontend Web Interface

To serve the web dashboard, spin up a lightweight local server in the project directory:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 📈 Dashboard Features & Controls

*   **LHC Parameter Configuration**:
    *   *Beam Energy (TeV)*: Controls relativistic gamma factor and synchrotron radiation.
    *   *Bunch Population*: Directly scales crossing-angle tune shifts and instantaneous luminosity.
    *   *Emittance & Beta Star*: Modifies the transverse beam size at the interaction points.
    *   *RF Cavity Voltage (MV)*: Influences longitudinal bucket stability and acceleration rates.
*   **Antimatter Decelerator Controls**:
    *   *Target Thickness & Material*: Adjusts antiproton yield and energy loss.
    *   *Cooling Power (Stochastic & Electron)*: Dynamically visualizes emittance reduction.
    *   *Ioffe Trap Field Strength (Tesla)*: Directly controls confinement time and synthesis count.
*   **Real-time Analytics**:
    *   Live plots for LHC Injector Energy Ramp, FODO Transverse Phase Space tracking, PP Collision Event Calorimeter, Beam Cooling trajectory, and Antihydrogen Trapping efficiency.
    *   Instantaneous calculation of physical metrics (e.g., Luminosity, Synchrotron Loss, Quench margins, and Hbar Synthesis rates).