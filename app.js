/**
 * =============================================================================
 *   CERN LHC & Antimatter Simulation — Frontend Controller
 * =============================================================================
 *   Binds all HTML sliders, dropdowns, and button events to the physics engine.
 *   Initializes and dynamically updates 30 Chart.js plots, custom vector
 *   drawings (LHC ring and magnets), and 2D canvas particle collisions.
 * =============================================================================
 */

(function() {
  // Global seedable random for consistent, premium MC visualizations
  class SeededRandom {
    constructor(seed) { this.seed = seed; }
    random() {
      let x = Math.sin(this.seed++) * 10000;
      return x - Math.floor(x);
    }
    uniform(min, max) { return min + this.random() * (max - min); }
    exponential(lambda) { return -Math.log(1.0 - this.random()) / lambda; }
    normal() {
      let u = 1 - this.random();
      let v = this.random();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
  }
  const rand = new SeededRandom(999);
  
  // Tab Elements
  const btnLhc = document.getElementById('btn-lhc');
  const btnAntimatter = document.getElementById('btn-antimatter');
  const lhcPanel = document.getElementById('lhc-panel');
  const antimatterPanel = document.getElementById('antimatter-panel');
  const lhcControls = document.getElementById('lhc-controls');
  const antimatterControls = document.getElementById('antimatter-controls');
  
  // Dashboard Chart Instances Cache
  const charts = {};

  // Color constants for high-fidelity charts
  const THEME = {
    cernBlue: '#0033A0',
    cernRed: '#D4002D',
    lhcGold: '#F5A623',
    antiPink: '#FF6B9D',
    antiPurple: '#C44EFF',
    antiCyan: '#00E5FF',
    antiGreen: '#39FF14',
    bgDark: '#0A0E1A',
    gridDark: '#1C2340',
    textMuted: '#7A8BAA',
    textWhite: '#E8EEF7'
  };

  // ──────────────────────────────────────────────────────────────────────────────
  // TAB NAVIGATION LIFECYCLE
  // ──────────────────────────────────────────────────────────────────────────────
  function switchTab(activeTab) {
    if (activeTab === 'lhc') {
      btnLhc.classList.add('active');
      btnLhc.setAttribute('aria-selected', 'true');
      btnAntimatter.classList.remove('active');
      btnAntimatter.setAttribute('aria-selected', 'false');
      
      lhcPanel.style.display = 'block';
      lhcControls.style.display = 'block';
      antimatterPanel.style.display = 'none';
      antimatterControls.style.display = 'none';
      
      // Render/update LHC charts
      requestAnimationFrame(updateLhcSimulation);
    } else {
      btnAntimatter.classList.add('active');
      btnAntimatter.setAttribute('aria-selected', 'true');
      btnLhc.classList.remove('active');
      btnLhc.setAttribute('aria-selected', 'false');
      
      antimatterPanel.style.display = 'block';
      antimatterControls.style.display = 'block';
      lhcPanel.style.display = 'none';
      lhcControls.style.display = 'none';
      
      // Render/update Antimatter charts
      requestAnimationFrame(updateAntimatterSimulation);
    }
  }

  btnLhc.addEventListener('click', () => switchTab('lhc'));
  btnAntimatter.addEventListener('click', () => switchTab('antimatter'));

  // ──────────────────────────────────────────────────────────────────────────────
  // HELPER: CHART BUILDER CONFIGS
  // ──────────────────────────────────────────────────────────────────────────────
  function getBaseChartConfig(type, title, xTitle, yTitle, isLogY = false, isLogX = false) {
    return {
      type: type,
      data: { labels: [], datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        plugins: {
          legend: {
            display: true,
            labels: {
              font: { family: 'Inter', size: 10, weight: '500' },
              color: '#444748'
            }
          },
          tooltip: {
            enabled: true,
            titleFont: { family: 'JetBrains Mono', size: 11 },
            bodyFont: { family: 'JetBrains Mono', size: 11 }
          }
        },
        scales: {
          x: {
            type: isLogX ? 'logarithmic' : (type === 'scatter' ? 'linear' : 'category'),
            title: { display: true, text: xTitle, font: { family: 'Inter', size: 11, weight: '600' }, color: '#1c1b1b' },
            grid: { color: 'rgba(0, 0, 0, 0.05)' },
            ticks: { font: { family: 'JetBrains Mono', size: 9 }, color: '#626262' }
          },
          y: {
            type: isLogY ? 'logarithmic' : 'linear',
            title: { display: true, text: yTitle, font: { family: 'Inter', size: 11, weight: '600' }, color: '#1c1b1b' },
            grid: { color: 'rgba(0, 0, 0, 0.05)' },
            ticks: { font: { family: 'JetBrains Mono', size: 9 }, color: '#626262' }
          }
        }
      }
    };
  }

  function getOrInitChart(canvasId, type, title, xTitle, yTitle, isLogY = false, isLogX = false) {
    if (charts[canvasId]) {
      return charts[canvasId];
    }
    const ctx = document.getElementById(canvasId).getContext('2d');
    const config = getBaseChartConfig(type, title, xTitle, yTitle, isLogY, isLogX);
    charts[canvasId] = new Chart(ctx, config);
    return charts[canvasId];
  }

  // ──────────────────────────────────────────────────────────────────────────────
  // A. LHC COLLIDER DYNAMICS & EVENTS
  // ──────────────────────────────────────────────────────────────────────────────
  
  // Slider Controls
  const inputEnergy = document.getElementById('input-energy');
  const inputBunchPop = document.getElementById('input-bunchpop');
  const inputEmittance = document.getElementById('input-emittance');
  const inputBetaStar = document.getElementById('input-betastar');
  const inputCrossing = document.getElementById('input-crossing');
  const inputRfVolt = document.getElementById('input-rfvolt');
  const inputCollP = document.getElementById('input-coll-p');
  const inputQuenchR = document.getElementById('input-quench-r');
  
  // Labels
  const lblEnergy = document.getElementById('lbl-val-energy');
  const lblBunchPop = document.getElementById('lbl-val-bunchpop');
  const lblEmittance = document.getElementById('lbl-val-emittance');
  const lblBetaStar = document.getElementById('lbl-val-betastar');
  const lblCrossing = document.getElementById('lbl-val-crossing');
  const lblRfVolt = document.getElementById('lbl-val-rfvolt');
  const lblCollP = document.getElementById('lbl-val-coll-p');
  const lblQuenchR = document.getElementById('lbl-val-quench-r');

  // Trigger collision metrics cache
  let mcZMassData = [];
  let mcHMassData = [];
  let mcDijetPtData = [];

  async function simulateMCEvents() { return; /* Handled by backend */
    // CERNPhysics usages removed to prevent ReferenceError

    // 1. Z → μ+μ- Events — build invariant mass from per-event calls
    const zMasses = [];
    for (let i = 0; i < 1200; i++) {
      try {
        const evt = gen.generateZEvent();
        if (!evt || evt.length < 2) continue;
        const p1s = det.smearParticle(evt[0], mcRand);
        const p2s = det.smearParticle(evt[1], mcRand);
        if (det.isAccepted(p1s) && det.isAccepted(p2s)) {
          const sumE  = p1s.E  + p2s.E;
          const sumPx = p1s.px + p2s.px;
          const sumPy = p1s.py + p2s.py;
          const sumPz = p1s.pz + p2s.pz;
          const m2 = sumE*sumE - sumPx*sumPx - sumPy*sumPy - sumPz*sumPz;
          if (m2 > 0) zMasses.push(Math.sqrt(m2));
        }
      } catch(e) { /* skip bad events */ }
    }
    data.mc.zMasses = Float64Array.from(zMasses);

    // 2. H → γγ Events
    const hMasses = [];
    for (let i = 0; i < 900; i++) {
      try {
        const evt = gen.generateHiggsEvent();
        if (!evt || evt.length < 2) continue;
        const p1s = det.smearParticle(evt[0], mcRand);
        const p2s = det.smearParticle(evt[1], mcRand);
        if (det.isAccepted(p1s) && det.isAccepted(p2s)) {
          const sumE  = p1s.E  + p2s.E;
          const sumPx = p1s.px + p2s.px;
          const sumPy = p1s.py + p2s.py;
          const sumPz = p1s.pz + p2s.pz;
          const m2 = sumE*sumE - sumPx*sumPx - sumPy*sumPy - sumPz*sumPz;
          if (m2 > 0) hMasses.push(Math.sqrt(m2));
        }
      } catch(e) { /* skip bad events */ }
    }
    data.mc.hMasses = Float64Array.from(hMasses);

    // 3. QCD Dijet Events
    const jetPt = [];
    for (let i = 0; i < 1800; i++) {
      try {
        const evt = gen.generateQCDDijet();
        if (!evt || evt.length === 0) continue;
        jetPt.push(Math.max(...evt.map(p => p.pt)));
      } catch(e) { /* skip */ }
    }
    data.mc.dijetPt = Float64Array.from(jetPt);
  }



  let currentMcSeed = 10101;

  let lhcAbortController = null;
  async function updateLhcSimulation() {
    if (lhcAbortController) {
      lhcAbortController.abort();
    }
    lhcAbortController = new AbortController();
    const { signal } = lhcAbortController;

    // 1. Pull current parameters from UI
    const energy = parseFloat(inputEnergy.value);
    const bunchPop = parseFloat(inputBunchPop.value);
    const emittVal = parseFloat(inputEmittance.value);
    const betaStar = parseFloat(inputBetaStar.value);
    const crossingAngle = parseFloat(inputCrossing.value);
    const rfVolt = parseFloat(inputRfVolt.value);
    const primarySigma = parseFloat(inputCollP.value);
    const quenchR = parseFloat(inputQuenchR.value);

    let data = null;
    try {
        const response = await fetch('http://localhost:8001/api/lhc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal,
            body: JSON.stringify({
                energy, bunchPop, emittVal, betaStar, crossingAngle, rfVolt, primarySigma, quenchR,
                mcSeed: currentMcSeed
            })
        });
        data = await response.json();
    } catch(e) {
        if (e.name === 'AbortError') return;
        console.error(e);
        return;
    }


    // 2. Update value text overlays
    lblEnergy.innerText = (energy / 1e12).toFixed(2) + " TeV";
    lblBunchPop.innerText = (bunchPop / 1e11).toFixed(2) + " × 10¹¹";
    lblEmittance.innerText = (emittVal * 1e6).toFixed(2) + " μm";
    lblBetaStar.innerText = betaStar.toFixed(2) + " m";
    lblCrossing.innerText = (crossingAngle * 1e6).toFixed(0) + " μrad";
    lblRfVolt.innerText = (rfVolt / 1e6).toFixed(1) + " MV";
    lblCollP.innerText = primarySigma.toFixed(1) + " σ";
    lblQuenchR.innerText = quenchR.toFixed(3) + " Ω";

    
    // 4. Update Header Metrics
    document.getElementById('val-lum').innerText = (data.header.peakLuminosity / 1e34).toFixed(3) + " × 10³⁴";
    document.getElementById('val-np').innerText = (data.header.bunchPopulation / 1e11).toFixed(2) + " × 10¹¹";
    document.getElementById('val-beamsz').innerText = (data.header.sigmaIP * 1e6).toFixed(2) + " μm";
    
    // Stored Energy = ½·L_dipole·I²·N_bunches·E_stored (approx stored kinetic energy per beam)
    // E_stored = N_bunches · N_p · E_beam
    const storedE = data.header.storedEnergy;
    document.getElementById('val-stored-e').innerText = storedE.toFixed(1) + " MJ";

    // 5. Update Parameter Table Body
    const summary = data.summary;
    const lhcTableBody = document.getElementById('lhc-table-body');
    lhcTableBody.innerHTML = '';
    
    // Add dynamically computed state
    const tableRows = data.summary;

    tableRows.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${row.name}</td><td>${row.unit}</td><td class="val">${typeof row.val === 'number' ? row.val.toPrecision(4) : row.val}</td>`;
      lhcTableBody.appendChild(tr);
    });

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR ACC-01: Injector Chain
    // ──────────────────────────────────────────────────────────────────────────────
    const injData = data.inj;
    const c1 = getOrInitChart('chart-injector', 'bar', 'CERN Injector Chain Output Energies', 'Accelerator Stage', 'Energy [TeV]', true);
    c1.data.labels = Object.keys(injData);
    c1.data.datasets = [{
      label: 'Extraction Energy',
      data: Object.values(injData).map(d => d.energy_TeV),
      backgroundColor: [THEME.cernBlue, '#4A90D9', '#00A859', '#E85E26', THEME.cernRed],
      borderColor: '#ffffff',
      borderWidth: 1
    }];
    c1.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR ACC-02: FODO Lattice Twiss
    // ──────────────────────────────────────────────────────────────────────────────
    const twiss = data.fodo;
    const c2 = getOrInitChart('chart-fodo', 'line', 'FODO Arc Cell Twiss parameters', 's [m]', 'β [m]');
    c2.data.labels = Array.from(twiss.s).map(v => v.toFixed(1));
    c2.data.datasets = [
      {
        label: 'β(s) Function [m] (Left Scale)',
        data: Array.from(twiss.beta),
        borderColor: THEME.cernBlue,
        borderWidth: 1.8,
        tension: 0.1,
        fill: false,
        yAxisID: 'y'
      },
      {
        label: 'α(s) Dispersion Slope (Right Scale)',
        data: Array.from(twiss.alpha),
        borderColor: THEME.lhcGold,
        borderWidth: 1.3,
        borderDash: [5, 5],
        tension: 0.1,
        fill: false,
        yAxisID: 'y1'
      }
    ];
    c2.options.scales.y1 = {
      type: 'linear',
      position: 'right',
      title: { display: true, text: 'α Slope', font: { family: 'Inter', size: 11, weight: '600' } },
      grid: { drawOnChartArea: false }
    };
    c2.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR PHS-01: Phase Ellipse
    // ──────────────────────────────────────────────────────────────────────────────
    const phase7T = {x: data.phaseEllipse.x7T, xp: data.phaseEllipse.xp7T};
    const phase450G = {x: data.phaseEllipse.x450G, xp: data.phaseEllipse.xp450G};
    const c3 = getOrInitChart('chart-phase-ellipse', 'line', 'Transverse Phase Space', 'x [mm]', 'x\' [mrad]');
    c3.data.labels = Array.from(phase7T.x).map(v => (v * 1e3).toFixed(3));
    c3.data.datasets = [
      {
        label: `${(energy/1e12).toFixed(1)} TeV Orbit (Current)`,
        data: Array.from(phase7T.xp).map(v => v * 1e3),
        borderColor: THEME.cernBlue,
        borderWidth: 1.8,
        backgroundColor: 'rgba(0, 51, 160, 0.08)',
        fill: true,
        pointRadius: 0
      },
      {
        label: '450 GeV Injection Orbit',
        data: Array.from(phase450G.xp).map(v => v * 1e3),
        borderColor: THEME.lhcGold,
        borderWidth: 1.3,
        borderDash: [3, 3],
        backgroundColor: 'rgba(245, 166, 35, 0.04)',
        fill: true,
        pointRadius: 0
      }
    ];
    // Keep labels aligned to x
    c3.data.labels = Array.from(phase7T.x).map(v => (v * 1e3).toFixed(3));
    c3.data.datasets[0].data = Array.from(phase7T.x).map((xVal, idx) => ({ x: xVal * 1e3, y: phase7T.xp[idx] * 1e3 }));
    c3.data.datasets[1].data = Array.from(phase450G.x).map((xVal, idx) => ({ x: xVal * 1e3, y: phase450G.xp[idx] * 1e3 }));
    c3.options.scales.x.type = 'linear';
    c3.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR PHS-02: Betatron Oscillations
    // ──────────────────────────────────────────────────────────────────────────────
    const bOsc = data.betatron;
    const c4 = getOrInitChart('chart-betatron', 'line', 'Transverse Betatron Motion', 'Turn Number', 'Displacement x [mm]');
    c4.data.labels = Array.from(bOsc.turns);
    c4.data.datasets = [{
      label: 'Particle horizontal trajectory',
      data: Array.from(bOsc.x).map(v => v * 1e3),
      borderColor: THEME.cernBlue,
      borderWidth: 1.2,
      pointRadius: 1,
      fill: false
    }];
    c4.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR RAD-01: Synchrotron Radiation vs Energy
    // ──────────────────────────────────────────────────────────────────────────────
    const eScanP = data.radiation.eScan;
    const uScanP = data.radiation.uScan;
    
    const c5 = getOrInitChart('chart-radiation', 'line', 'Energy Loss Per Revolution', 'Beam Energy [TeV]', 'U0 Loss [keV/turn]');
    c5.data.labels = eScanP.map(e => (e / 1e12).toFixed(1));
    c5.data.datasets = [{
      label: 'Proton Synchrotron Loss',
      data: uScanP.map(u => u / 1e3), // in keV
      borderColor: THEME.cernBlue,
      borderWidth: 1.8,
      backgroundColor: 'rgba(0, 51, 160, 0.08)',
      fill: true,
      pointRadius: 0
    }];
    c5.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR RF-01: RF Longitudinal Phase Space
    // ──────────────────────────────────────────────────────────────────────────────
    const rfSpace1 = {phi: data.rfSpace.phi1, delta: data.rfSpace.delta1};
    const rfSpace2 = {phi: data.rfSpace.phi2, delta: data.rfSpace.delta2};
    const c6 = getOrInitChart('chart-rf-space', 'scatter', 'Longitudinal Phase Space Orbit', 'Phase [rad]', 'Energy Spread ΔE/E');
    c6.data.datasets = [
      {
        label: 'Stable Bucket Trajectory A',
        data: Array.from(rfSpace1.phi).map((phi, idx) => ({ x: phi, y: rfSpace1.delta[idx] })),
        borderColor: THEME.cernBlue,
        backgroundColor: THEME.cernBlue,
        pointRadius: 0.5,
        showLine: false
      },
      {
        label: 'Outer Bucket Trajectory B',
        data: Array.from(rfSpace2.phi).map((phi, idx) => ({ x: phi, y: rfSpace2.delta[idx] })),
        borderColor: THEME.lhcGold,
        backgroundColor: THEME.lhcGold,
        pointRadius: 0.5,
        showLine: false
      }
    ];
    c6.options.scales.x.type = 'linear';
    c6.options.scales.x.min = -0.8;
    c6.options.scales.x.max = 0.8;
    c6.options.scales.y.min = -0.003;
    c6.options.scales.y.max = 0.003;
    c6.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR LUM-01: Luminosity vs Energy
    // ──────────────────────────────────────────────────────────────────────────────
    const eScanL = data.lumEnergy.eScan;
    const lScanE = data.lumEnergy.lScan;
    const c7 = getOrInitChart('chart-lum-energy', 'line', 'Luminosity vs Beam Energy', 'Beam Energy [TeV]', 'L [×10³⁴ cm⁻²s⁻¹]');
    c7.data.labels = eScanL.map(e => (e / 1e12).toFixed(1));
    c7.data.datasets = [{
      label: 'Peak Luminosity',
      data: lScanE,
      borderColor: THEME.lhcGold,
      backgroundColor: 'rgba(245, 166, 35, 0.08)',
      borderWidth: 2,
      fill: true,
      pointRadius: 0
    }];
    c7.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR LUM-02: Luminosity vs Beta Star
    // ──────────────────────────────────────────────────────────────────────────────
    const bsScan = data.lumBeta.bsScan;
    const lScanBS = data.lumBeta.lScan;
    const c8 = getOrInitChart('chart-lum-beta', 'line', 'Luminosity vs Beta Star Focus', 'β* [m]', 'L [×10³⁴ cm⁻²s⁻¹]');
    c8.data.labels = bsScan.map(b => b.toFixed(2));
    c8.data.datasets = [{
      label: 'Luminosity Yield',
      data: lScanBS,
      borderColor: THEME.cernBlue,
      backgroundColor: 'rgba(0, 51, 160, 0.08)',
      borderWidth: 2,
      fill: true,
      pointRadius: 0
    }];
    c8.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR ACC-03: Energy Ramp
    // ──────────────────────────────────────────────────────────────────────────────
    const ramp = data.ramp;
    const c9 = getOrInitChart('chart-lhc-ramp', 'line', 'Accelerator 20-min energy ramp', 'Time [min]', 'Energy [TeV]');
    c9.data.labels = Array.from(ramp.t).map(v => (v / 60).toFixed(1));
    c9.data.datasets = [{
      label: 'Main Dipole Ramp Profile',
      data: Array.from(ramp.E),
      borderColor: THEME.lhcGold,
      borderWidth: 1.8,
      pointRadius: 0,
      fill: false
    }];
    c9.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR QPN-01: Quench current decay
    // ──────────────────────────────────────────────────────────────────────────────
    const tQuench = data.quench.t;
    const iQuench = data.quench.I;
    const c10 = getOrInitChart('chart-quench', 'line', 'Superconducting magnet current quench decay', 'Time [ms]', 'Current [kA]');
    c10.data.labels = tQuench.map(t => (t * 1e3).toFixed(0));
    c10.data.datasets = [{
      label: 'Coil Current Decay I(t)',
      data: iQuench.map(v => v / 1000), // in kA
      borderColor: THEME.cernRed,
      backgroundColor: 'rgba(212, 0, 45, 0.08)',
      borderWidth: 1.8,
      pointRadius: 0,
      fill: true
    }];
    c10.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR BBE-01: Beam-beam tune shift
    // ──────────────────────────────────────────────────────────────────────────────
    const nScanT = data.beamBeam.nScan;
    const xiScan = data.beamBeam.xiScan;
    const c11 = getOrInitChart('chart-beam-beam', 'line', 'Linear Tune Shift vs Bunch Intensity', 'Bunch Population [×10¹¹]', 'Tune Shift ξ [×10⁻³]');
    c11.data.labels = nScanT.map(n => (n / 1e11).toFixed(2));
    c11.data.datasets = [{
      label: 'Tune Shift ξ / Interaction Point',
      data: xiScan.map(x => x * 1e3),
      borderColor: THEME.antiPurple,
      backgroundColor: 'rgba(196, 78, 255, 0.08)',
      borderWidth: 1.8,
      pointRadius: 0,
      fill: true
    }];
    c11.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR COL-01: Collimation scraped halo distribution
    // ──────────────────────────────────────────────────────────────────────────────
    // Generate simple Gaussian mock histogram
    const c12 = getOrInitChart('chart-collimation', 'line', 'Transverse Beam Profile with Scraping', 'Relative Position [x/σ]', 'Particle Density', true);
    const bins = Array.from({length: 60}, (_, i) => -15 + (i * 30 / 59));
    const histAll = bins.map(x => Math.exp(-0.5 * x * x) + 1e-5);
    const histScraped = bins.map(x => Math.abs(x) < primarySigma ? Math.exp(-0.5 * x * x) + 1e-5 : 1e-5);
    
    c12.data.labels = bins.map(v => v.toFixed(1));
    c12.data.datasets = [
      {
        label: 'Unscraped Halo Profile',
        data: histAll,
        borderColor: THEME.textMuted,
        borderWidth: 1,
        borderDash: [3, 3],
        pointRadius: 0,
        fill: false
      },
      {
        label: `Scraped Profile (< ${primarySigma.toFixed(1)}σ)`,
        data: histScraped,
        borderColor: THEME.cernBlue,
        backgroundColor: 'rgba(0, 51, 160, 0.08)',
        borderWidth: 1.8,
        pointRadius: 0,
        fill: true
      }
    ];
    c12.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR PDF-01: Parton Distribution Functions
    // ──────────────────────────────────────────────────────────────────────────────
    const xPdf = data.pdf.x;
    const pdfU = data.pdf.u;
    const pdfD = data.pdf.d;
    const pdfG = data.pdf.g; // Gluon downscaled
    
    const c13 = getOrInitChart('chart-pdf', 'line', 'CTEQ-like PDF momentum distribution xf(x)', 'Parton Momentum Fraction x', 'xf(x, Q²=10000 GeV²)');
    c13.data.labels = xPdf.map(x => x.toFixed(3));
    c13.data.datasets = [
      {
        label: 'u Quark',
        data: pdfU,
        borderColor: THEME.cernRed,
        borderWidth: 1.8,
        pointRadius: 0,
        fill: false
      },
      {
        label: 'd Quark',
        data: pdfD,
        borderColor: THEME.cernBlue,
        borderWidth: 1.8,
        pointRadius: 0,
        fill: false
      },
      {
        label: 'Gluon xf(x)/4',
        data: pdfG,
        borderColor: '#00A859',
        borderWidth: 1.2,
        borderDash: [4, 4],
        pointRadius: 0,
        fill: false
      }
    ];
    c13.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR DET-01: Reconstructed Z Invariant Mass Histogram
    // ──────────────────────────────────────────────────────────────────────────────
    const c14 = getOrInitChart('chart-z-mass', 'bar', 'Reconstructed Z Boson Mass Spectrum', 'm_μμ [GeV]', 'Events / 1.5 GeV');
    // Calculate bin counts
    const zBins = Array.from({length: 40}, (_, i) => 60.0 + i * 2.0);
    const zCounts = new Array(zBins.length).fill(0);
    data.mc.zMasses.forEach(v => {
      const idx = Math.floor((v - 60.0) / 2.0);
      if (idx >= 0 && idx < zBins.length) zCounts[idx]++;
    });
    c14.data.labels = zBins.map(v => v.toFixed(0));
    c14.data.datasets = [{
      label: 'Dimuon mass events',
      data: zCounts,
      backgroundColor: THEME.cernBlue,
      borderColor: '#ffffff',
      borderWidth: 0.5
    }];
    c14.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR DET-02: Reconstructed Higgs Diphoton Mass Histogram
    // ──────────────────────────────────────────────────────────────────────────────
    const c15 = getOrInitChart('chart-h-mass', 'bar', 'Reconstructed Higgs Mass Spectrum', 'm_γγ [GeV]', 'Events / 1.0 GeV');
    const hBins = Array.from({length: 40}, (_, i) => 105.0 + i * 1.0);
    const hCounts = new Array(hBins.length).fill(0);
    data.mc.hMasses.forEach(v => {
      const idx = Math.floor((v - 105.0) / 1.0);
      if (idx >= 0 && idx < hBins.length) hCounts[idx]++;
    });
    c15.data.labels = hBins.map(v => v.toFixed(0));
    c15.data.datasets = [{
      label: 'Diphoton mass events',
      data: hCounts,
      backgroundColor: THEME.lhcGold,
      borderColor: '#ffffff',
      borderWidth: 0.5
    }];
    c15.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ACCELERATOR DET-03: Dijet pT spectrum
    // ──────────────────────────────────────────────────────────────────────────────
    const c16 = getOrInitChart('chart-dijet-pt', 'bar', 'QCD Hard Scattering Jet Transverse Momentum', 'Jet p_T [GeV]', 'Events / 10 GeV', true);
    const ptBins = Array.from({length: 40}, (_, i) => 20.0 + i * 10.0);
    const ptCounts = new Array(ptBins.length).fill(0);
    data.mc.dijetPt.forEach(v => {
      const idx = Math.floor((v - 20.0) / 10.0);
      if (idx >= 0 && idx < ptBins.length) ptCounts[idx]++;
    });
    c16.data.labels = ptBins.map(v => v.toFixed(0));
    c16.data.datasets = [{
      label: 'High-p_T Jet Spectrum',
      data: ptCounts.map(c => c + 1e-1), // Add offset to avoid logarithmic zeros
      backgroundColor: '#00A859',
      borderColor: '#ffffff',
      borderWidth: 0.5
    }];
    c16.update('none');

    // 6. Draw dynamic vector SVG structures
    const beamObj = { betaStar: betaStar, sigmaIP: data.header.sigmaIP };
    drawInteractiveLhcRing(beamObj);
    drawTwinBoreMagnet(beamObj);
  }

  // ──────────────────────────────────────────────────────────────────────────────
  // VECTOR GRAPHICS: LHC RING INTERACTION
  // ──────────────────────────────────────────────────────────────────────────────
  function drawInteractiveLhcRing(beam) {
    const svg = document.getElementById('svg-lhc-ring');
    svg.innerHTML = '';
    
    // Create base ring circle
    const baseRing = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    baseRing.setAttribute("cx", "0");
    baseRing.setAttribute("cy", "0");
    baseRing.setAttribute("r", "1.0");
    baseRing.setAttribute("fill", "none");
    baseRing.setAttribute("stroke", "rgba(0, 80, 200, 0.4)");
    baseRing.setAttribute("stroke-width", "0.04");
    baseRing.setAttribute("stroke-dasharray", "0.02 0.02");
    svg.appendChild(baseRing);

    // Orbit path representation with pulse animations
    const orbit = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    orbit.setAttribute("cx", "0");
    orbit.setAttribute("cy", "0");
    orbit.setAttribute("r", "0.98");
    orbit.setAttribute("fill", "none");
    orbit.setAttribute("stroke", THEME.cernBlue);
    orbit.setAttribute("stroke-width", "0.02");
    svg.appendChild(orbit);

    // Interaction Points coordinates
    const IPs = [
      { name: 'ATLAS (IP1)', angle: 0, color: THEME.cernRed, desc: 'Highest Luminosity general-purpose detector.' },
      { name: 'ALICE (IP2)', angle: Math.PI / 2, color: THEME.antiPurple, desc: 'Heavy-ion quark-gluon plasma analyzer.' },
      { name: 'CMS (IP5)', angle: Math.PI, color: '#00A859', desc: 'Compact Solenoid general-purpose detector.' },
      { name: 'LHCb (IP8)', angle: 3 * Math.PI / 2, color: THEME.antiCyan, desc: 'B-physics and CP-violation spectrometer.' }
    ];

    IPs.forEach(ip => {
      const cx = 0.98 * Math.cos(ip.angle);
      const cy = 0.98 * Math.sin(ip.angle);
      
      // Glow pulse card elements
      const pulse = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      pulse.setAttribute("cx", cx.toString());
      pulse.setAttribute("cy", cy.toString());
      pulse.setAttribute("r", "0.08");
      pulse.setAttribute("fill", ip.color);
      pulse.setAttribute("opacity", "0.15");
      svg.appendChild(pulse);

      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", cx.toString());
      dot.setAttribute("cy", cy.toString());
      dot.setAttribute("r", "0.04");
      dot.setAttribute("fill", ip.color);
      dot.setAttribute("stroke", "#ffffff");
      dot.setAttribute("stroke-width", "0.015");
      dot.style.cursor = 'pointer';
      
      // Title Tooltip Hovering properties
      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      title.textContent = `${ip.name}\n${ip.desc}\nIP Focus β* = ${(beam.betaStar||0.55).toFixed(2)} m`;
      dot.appendChild(title);
      svg.appendChild(dot);
      
      // Text labels
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", (cx * 1.25).toString());
      label.setAttribute("y", (cy * 1.25).toString());
      label.setAttribute("fill", ip.color);
      label.setAttribute("font-size", "0.09");
      label.setAttribute("font-family", "Inter");
      label.setAttribute("font-weight", "700");
      label.setAttribute("text-anchor", "middle");
      label.textContent = ip.name.split(' ')[0];
      svg.appendChild(label);
    });

    // Center Core annotation
    const centerTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    centerTxt.setAttribute("x", "0");
    centerTxt.setAttribute("y", "0");
    centerTxt.setAttribute("fill", "#7a8baa");
    centerTxt.setAttribute("font-size", "0.07");
    centerTxt.setAttribute("font-family", "JetBrains Mono");
    centerTxt.setAttribute("text-anchor", "middle");
    centerTxt.innerHTML = `<tspan x="0" dy="0">LHC RING</tspan><tspan x="0" dy="0.1">C = 26.7 km</tspan><tspan x="0" dy="0.1">B = 8.33 T</tspan>`;
    svg.appendChild(centerTxt);
  }

  // ──────────────────────────────────────────────────────────────────────────────
  // VECTOR GRAPHICS: TWIN BORE ACCELERATOR DIPOLES
  // ──────────────────────────────────────────────────────────────────────────────
  function drawTwinBoreMagnet(beam) {
    const svg = document.getElementById('svg-dipole');
    svg.innerHTML = '';

    // Main Outer Iron Yoke Ring
    const yoke = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    yoke.setAttribute("cx", "0");
    yoke.setAttribute("cy", "0");
    yoke.setAttribute("r", "0.95");
    yoke.setAttribute("fill", "#1c2340");
    yoke.setAttribute("stroke", "rgba(255, 255, 255, 0.15)");
    yoke.setAttribute("stroke-width", "0.02");
    svg.appendChild(yoke);

    const apertures = [-0.38, 0.38];
    apertures.forEach((dx, idx) => {
      // Superconducting Coil Chamber Bore
      const bore = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      bore.setAttribute("cx", dx.toString());
      bore.setAttribute("cy", "0");
      bore.setAttribute("r", "0.3");
      bore.setAttribute("fill", "#050810");
      bore.setAttribute("stroke", THEME.cernBlue);
      bore.setAttribute("stroke-width", "0.03");
      svg.appendChild(bore);

      // Superconducting NbTi wire pack layout representation
      const coil = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      coil.setAttribute("cx", dx.toString());
      coil.setAttribute("cy", "0");
      coil.setAttribute("r", "0.22");
      coil.setAttribute("fill", "none");
      coil.setAttribute("stroke", "rgba(0, 80, 200, 0.4)");
      coil.setAttribute("stroke-width", "0.04");
      coil.setAttribute("stroke-dasharray", "0.03 0.02");
      svg.appendChild(coil);

      // Vacuum Beam Pipe
      const pipe = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      pipe.setAttribute("cx", dx.toString());
      pipe.setAttribute("cy", "0");
      pipe.setAttribute("r", "0.15");
      pipe.setAttribute("fill", "#000000");
      pipe.setAttribute("stroke", "rgba(255, 255, 255, 0.1)");
      pipe.setAttribute("stroke-width", "0.01");
      svg.appendChild(pipe);

      // Glowing dynamic beam size dot
      const beamSizeRad = ((beam.sigmaIP || 1.6e-5) * 1.5e4); // Scale up for visualization
      const beamDot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      beamDot.setAttribute("cx", dx.toString());
      beamDot.setAttribute("cy", "0");
      beamDot.setAttribute("r", Math.max(beamSizeRad, 0.015).toString());
      beamDot.setAttribute("fill", THEME.lhcGold);
      beamDot.setAttribute("opacity", "0.95");
      beamDot.style.filter = "drop-shadow(0 0 4px #F5A623)";
      svg.appendChild(beamDot);

      // Magnetic B-field vertical arrows
      const arrows = [-0.08, 0.0, 0.08];
      arrows.forEach(ax => {
        const arrow = document.createElementNS("http://www.w3.org/2000/svg", "line");
        arrow.setAttribute("x1", (dx + ax).toString());
        // Reverse magnetic field direction on opposite beam path
        if (idx === 0) {
          arrow.setAttribute("y1", "-0.1");
          arrow.setAttribute("y2", "0.1");
        } else {
          arrow.setAttribute("y1", "0.1");
          arrow.setAttribute("y2", "-0.1");
        }
        arrow.setAttribute("stroke", THEME.cernRed);
        arrow.setAttribute("stroke-width", "0.015");
        arrow.setAttribute("marker-end", "url(#arrow)");
        svg.appendChild(arrow);
      });
    });
  }

  // Simulation state flags
  let lhcSimStarted = false;
  let antimatterSimStarted = false;

  // Bind UI Events — only update if simulation has been started
  const lhcSliders = [
    inputEnergy, inputBunchPop, inputEmittance, inputBetaStar,
    inputCrossing, inputRfVolt, inputCollP, inputQuenchR
  ];
  lhcSliders.forEach(slider => {
    slider.addEventListener('input', () => {
      if (lhcSimStarted) requestAnimationFrame(updateLhcSimulation);
    });
  });

  // Collision MC Event simulation trigger button (declare before startLhcSimulation)
  const btnTriggerCollision = document.getElementById('btn-trigger-collision');
  const collisionLoader = document.getElementById('collision-loader');
  const collisionText = document.getElementById('collision-text');

  // LHC Start Simulation handler (shared by sidebar + overlay buttons)
  function startLhcSimulation() {
    if (lhcSimStarted) return;
    lhcSimStarted = true;

    // Dismiss overlay
    const overlay = document.getElementById('lhc-start-overlay');
    if (overlay) overlay.classList.add('hidden');

    // Enable the Generate PP Collision button
    btnTriggerCollision.disabled = false;
    btnTriggerCollision.style.background = '';

    // Pre-generate MC events then run full simulation
    try { simulateMCEvents(); } catch(err) {
      console.warn('MC event pre-generation error:', err);
    }
    requestAnimationFrame(updateLhcSimulation);
  }

  // Sidebar Start button
  document.getElementById('btn-start-lhc').addEventListener('click', startLhcSimulation);
  // Overlay Start button
  document.getElementById('btn-start-lhc-overlay').addEventListener('click', startLhcSimulation);

  btnTriggerCollision.addEventListener('click', async () => {
    collisionLoader.style.display = 'inline-block';
    collisionText.innerText = 'Calculating Hard Scatter...';
    btnTriggerCollision.disabled = true;

    currentMcSeed = Math.floor(Math.random() * 999999);
    await updateLhcSimulation();

    collisionLoader.style.display = 'none';
    collisionText.innerText = 'Generate PP Collision';
    btnTriggerCollision.disabled = false;
  });


  // ──────────────────────────────────────────────────────────────────────────────
  // B. ANTIMATTER DECELERATOR & Trap
  // ──────────────────────────────────────────────────────────────────────────────
  
  // Slider Controls
  const inputPbarEnergy = document.getElementById('input-pbar-energy');
  const inputPbarMat = document.getElementById('input-pbar-mat');
  const inputPbarThickness = document.getElementById('input-pbar-thickness');
  const inputCoolBw = document.getElementById('input-cool-bw');
  const inputEcoolCurrent = document.getElementById('input-ecool-current');
  const inputHbarTemp = document.getElementById('input-hbar-temp');
  const inputIoffeField = document.getElementById('input-ioffe-field');

  // Labels
  const lblPbarEnergy = document.getElementById('lbl-val-pbar-energy');
  const lblPbarThickness = document.getElementById('lbl-val-pbar-thickness');
  const lblCoolBw = document.getElementById('lbl-val-cool-bw');
  const lblEcoolCurrent = document.getElementById('lbl-val-ecool-current');
  const lblHbarTemp = document.getElementById('lbl-val-hbar-temp');
  const lblIoffeField = document.getElementById('lbl-val-ioffe-field');

  // Trapping MC data state caches
  let anniPionsData = [];
  let anniGammaData = [];
  let anniVertexData = [];

  function simulateAnnihilationData(annData) {
    // annData comes from the backend /api/antimatter response
    if (!annData) return;
    // Generate mock pion pT, gamma E, and vertex r from the annihilation x/y positions
    const rng = new SeededRandom(Math.floor(Math.random() * 99999));
    const n = annData.x ? annData.x.length : 0;
    anniPionsData = Array.from({length: n}, () => Math.abs(rng.normal()) * 0.08 + 0.04);
    anniGammaData = Array.from({length: n}, () => Math.abs(rng.normal()) * 0.05 + 0.02);
    anniVertexData = annData.x.map((x,i) => Math.sqrt(x*x + annData.y[i]*annData.y[i]) * 1000);
  }


  let antimatterAbortController = null;
  async function updateAntimatterSimulation() {
    if (antimatterAbortController) {
      antimatterAbortController.abort();
    }
    antimatterAbortController = new AbortController();
    const { signal } = antimatterAbortController;

    // 1. Pull parameters from UI
    const pbarEnergy = parseFloat(inputPbarEnergy.value);
    const pbarMat = inputPbarMat.value;
    const pbarThickness = parseFloat(inputPbarThickness.value);
    const coolBw = parseFloat(inputCoolBw.value);
    const ecoolCurrent = parseFloat(inputEcoolCurrent.value);
    const hbarTemp = parseFloat(inputHbarTemp.value);
    const ioffeField = parseFloat(inputIoffeField.value);

    // 2. Update labels
    lblPbarEnergy.innerText = (pbarEnergy / 1e9).toFixed(1) + " GeV";
    lblPbarThickness.innerText = pbarThickness.toFixed(1) + " cm";
    lblCoolBw.innerText = (coolBw / 1e6).toFixed(0) + " MHz";
    lblEcoolCurrent.innerText = ecoolCurrent.toFixed(2) + " A";
    lblHbarTemp.innerText = hbarTemp.toFixed(1) + " K";
    lblIoffeField.innerText = ioffeField.toFixed(2) + " T";

    // 3. Fetch from backend
    let data = null;
    try {
      const resp = await fetch('http://localhost:8001/api/antimatter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal,
        body: JSON.stringify({
          pbarEnergy, pbarMat, pbarThickness, coolBw, hbarTemp,
          ioffeField, annSeed: currentAnnSeed, ecoolCurrent,
          posEnergy: 1e6, posTargetT: 3
        })
      });
      data = await resp.json();
    } catch(e) {
      if (e.name === 'AbortError') return;
      console.error('Antimatter fetch error:', e);
      return;
    }

    // 4. Update header metrics from backend
    const yieldVal = data.summary.yield;
    const rateVal = yieldVal * 1.5e13 * 0.4;
    document.getElementById('val-pbar-rate').innerText = rateVal.toExponential(2);

    const decelChain = data.decel;

    // Stochastic cooling time estimate from backend momentum spread data
    const scDp = data.scans.scDp;
    const scTime = data.scans.scTime;
    // Estimate cooling time as time when spread drops to 1/e of initial
    const dp0 = scDp[0] || 1e-3;
    let scCoolTime = scTime[scTime.length - 1];
    for (let i = 0; i < scDp.length; i++) {
      if (scDp[i] <= dp0 / Math.E) { scCoolTime = scTime[i]; break; }
    }
    document.getElementById('val-decel-eff').innerText = (99.85 - (scCoolTime / 300)).toFixed(2) + "%";

    // Antihydrogen formation rate (first value from ah scan)
    const ahRate = data.scans.ahRate;
    document.getElementById('val-hbar-3body').innerText = (ahRate[0] || 0).toExponential(2) + " s⁻¹";

    // Trap depth estimate: μ_B * B_max / e in meV
    const muB = 9.274e-24; const eCharge = 1.602e-19;
    const depthEv = muB * ioffeField / eCharge;
    document.getElementById('val-trap-depth').innerText = (depthEv * 1e3).toFixed(3) + " meV";

    // 5. Update annihilation data with backend annihilation positions
    simulateAnnihilationData(data.annihilation);

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER ANT-01: pbar Production Cross Section
    // ──────────────────────────────────────────────────────────────────────────────
    const tScanPbar = data.scans.tScanPbar;
    const xsScan = data.scans.xsScan;
    const c17 = getOrInitChart('chart-pbar-xs', 'line', 'Antiproton Cross Section vs Beam energy', 'Proton Beam Energy [GeV]', 'Cross-Section [mb × 10⁻²]');
    c17.data.labels = tScanPbar.map(t => (t / 1e9).toFixed(1));
    c17.data.datasets = [{
      label: 'Buss et al. Cross-Section',
      data: xsScan.map(x => x * 1e31),
      borderColor: THEME.antiPink,
      backgroundColor: 'rgba(255, 107, 157, 0.08)',
      borderWidth: 2,
      fill: true,
      pointRadius: 0
    }];
    c17.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER ANT-02: Target Yield vs Thickness
    // ──────────────────────────────────────────────────────────────────────────────
    const c18 = getOrInitChart('chart-pbar-yield', 'line', 'Antiproton Yield vs Thickness', 'Target Thickness [cm]', 'Yield/incident proton');
    const thickScan = data.scans.thickScan;
    const yieldScan = data.scans.yieldScan;
    c18.data.labels = thickScan.map(t => (t * 100).toFixed(1));
    c18.data.datasets = [{
      label: `${pbarMat} Yield Profile`,
      data: yieldScan,
      borderColor: THEME.lhcGold,
      backgroundColor: 'rgba(245, 166, 35, 0.08)',
      borderWidth: 1.8,
      fill: true,
      pointRadius: 0
    }];
    c18.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER ANT-03: AD/ELENA Chain KE
    // ──────────────────────────────────────────────────────────────────────────────
    const c19 = getOrInitChart('chart-pbar-chain-ke', 'bar', 'Deceleration Kinetic Energy Levels', 'Stage', 'Kinetic Energy [MeV]', true);
    c19.data.labels = decelChain.map(d => d.stage);
    c19.data.datasets = [{
      label: 'Stage Energy level',
      data: decelChain.map(d => d.KE_MeV),
      backgroundColor: decelChain.map(d => d.cooling === 'stochastic' ? THEME.antiPink : (d.cooling === 'electron' ? THEME.antiCyan : '#e5e2e1')),
      borderColor: '#ffffff',
      borderWidth: 0.5
    }];
    c19.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER ANT-04: Beam Rigidity through Chain
    // ──────────────────────────────────────────────────────────────────────────────
    const c20 = getOrInitChart('chart-pbar-chain-br', 'line', 'Beam Rigidity along decelerator', 'Stage', 'Rigidity Bρ [T·m]', true);
    c20.data.labels = decelChain.map(d => d.stage);
    c20.data.datasets = [{
      label: 'Rigidity level',
      data: decelChain.map(d => d.Brho_Tm),
      borderColor: THEME.lhcGold,
      borderWidth: 2,
      pointRadius: 4,
      fill: false
    }];
    c20.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER COL-02: Stochastic Cooling momentum spread
    // ──────────────────────────────────────────────────────────────────────────────
    const c21 = getOrInitChart('chart-stochastic-cool', 'line', 'Stochastic cooling emittance/momentum reduction', 'Time [s]', 'Δp/p [%]');
    c21.data.labels = data.scans.scTime.map(t => t.toFixed(0));
    c21.data.datasets = [
      {
        label: 'Momentum spread Δp/p [%]',
        data: data.scans.scDp.map(v => v * 100),
        borderColor: THEME.antiPink,
        borderWidth: 1.8,
        pointRadius: 0,
        fill: false,
        yAxisID: 'y'
      },
      {
        label: 'Electron cool Δp/p [%]',
        data: data.scans.ecDp.slice(0, data.scans.scTime.length).map(v => v * 100),
        borderColor: THEME.antiCyan,
        borderWidth: 1.3,
        borderDash: [4, 4],
        pointRadius: 0,
        fill: false,
        yAxisID: 'y'
      }
    ];
    c21.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER COL-03: Electron Cooling
    // ──────────────────────────────────────────────────────────────────────────────
    const c22 = getOrInitChart('chart-electron-cool', 'line', 'Electron cooling transverse emittance decay', 'Time [s]', 'Δp/p [%]');
    c22.data.labels = data.scans.ecTime.map(t => t.toFixed(1));
    c22.data.datasets = [{
      label: 'Electron cooling Δp/p',
      data: data.scans.ecDp.map(v => v * 100),
      borderColor: THEME.antiCyan,
      backgroundColor: 'rgba(0, 229, 255, 0.08)',
      borderWidth: 1.8,
      pointRadius: 0,
      fill: true
    }];
    c22.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER SCH-01: Schottky Noise Spectrum
    // ──────────────────────────────────────────────────────────────────────────────
    const schottkyData = { f: data.schottky.f, P: data.schottky.P };
    const c23 = getOrInitChart('chart-schottky', 'line', 'Longitudinal Schottky Spectrum', 'Frequency [MHz]', 'Power [a.u.]');
    const schArray = Array.from(schottkyData.f).map((fVal, idx) => ({ x: fVal / 1e6, y: schottkyData.P[idx] }));
    schArray.sort((a, b) => a.x - b.x);
    c23.data.labels = schArray.map(v => v.x.toFixed(3));
    c23.data.datasets = [{
      label: 'Schottky noise bands',
      data: schArray,
      borderColor: THEME.antiPurple,
      backgroundColor: 'rgba(196, 78, 255, 0.4)',
      borderWidth: 0.8,
      fill: true,
      pointRadius: 0
    }];
    c23.options.scales.x.type = 'linear';
    c23.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER POS-01: Positron Na22 Spectrum (from backend)
    // ──────────────────────────────────────────────────────────────────────────────
    const c24 = getOrInitChart('chart-positron', 'line', 'Fermi-Kurie positron source decay spectrum', 'Energy [MeV]', 'dN/dE [a.u.]');
    c24.data.labels = data.scans.psE.map(v => v.toFixed(3));
    c24.data.datasets = [{
      label: '²²Na beta+ positron spectrum',
      data: data.scans.psSpec,
      borderColor: THEME.antiPink,
      backgroundColor: 'rgba(255, 107, 157, 0.08)',
      borderWidth: 1.8,
      fill: true,
      pointRadius: 0
    }];
    c24.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER HBR-01: Antihydrogen formation rate vs T (from backend)
    // ──────────────────────────────────────────────────────────────────────────────
    const c25 = getOrInitChart('chart-hbar-formation', 'line', 'Formation rate vs Plasma temperature', 'Plasma Temperature [K]', 'Formation Rate [H̄/s]', true, true);
    c25.data.labels = data.scans.tHbar.map(t => t.toFixed(1));
    c25.data.datasets = [
      {
        label: 'Total H̄ Formation Rate',
        data: data.scans.ahRate.map((val, idx) => ({ x: data.scans.tHbar[idx], y: Math.max(val, 1e-15) })),
        borderColor: THEME.antiCyan,
        borderWidth: 2,
        pointRadius: 0,
        fill: false
      }
    ];
    c25.options.scales.x.type = 'linear';
    c25.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER HBR-02: Hbar principal quantum levels (computed client-side)
    // ──────────────────────────────────────────────────────────────────────────────
    const nArr = Array.from({length: 24}, (_, i) => i + 2);
    const kT = 8.617e-5 * hbarTemp; // eV
    const weights = nArr.map(n => {
      const En = 13.6 / (n * n);
      return Math.pow(n, 4) * Math.exp(-En / (kT + 1e-30));
    });
    const wSum = weights.reduce((a, b) => a + b, 0);
    const c26 = getOrInitChart('chart-hbar-nlevel', 'bar', 'Rydberg principal quantum states populated', 'Principal Quantum Number n', 'Yield Population [%]');
    c26.data.labels = nArr.map(v => v.toFixed(0));
    c26.data.datasets = [{
      label: 'Quantum State Population Yield',
      data: weights.map(v => v / wSum * 100),
      backgroundColor: THEME.antiPurple,
      borderColor: '#ffffff',
      borderWidth: 0.5
    }];
    c26.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER TRP-01: Ioffe Pritchard B-field (from backend)
    // ──────────────────────────────────────────────────────────────────────────────
    const c27 = getOrInitChart('chart-ioffe-field', 'line', 'ALPHA-style Ioffe trap magnetic field profile', 'z Position [cm]', 'B Field [T]');
    c27.data.labels = data.scans.trapR.map(v => v.toFixed(1));
    c27.data.datasets = [{
      label: 'Solenoid octupole field B(z)',
      data: data.scans.trapB,
      borderColor: '#00A859',
      backgroundColor: 'rgba(0, 168, 89, 0.08)',
      borderWidth: 2,
      fill: true,
      pointRadius: 0
    }];
    c27.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER TRP-02: Trapping fraction vs Temperature (computed client-side)
    // ──────────────────────────────────────────────────────────────────────────────
    const muBJ = 9.274e-24; const kBJ = 1.381e-23;
    const depthK = muBJ * ioffeField / kBJ;
    const hbarTempScan = Array.from({length: 60}, (_, i) => 0.1 + (i * 4.9 / 59));
    const trapFracScan = hbarTempScan.map(t => {
      const x = depthK / (t + 1e-30);
      const frac = (2 / Math.sqrt(Math.PI)) * (Math.sqrt(x) * Math.exp(-x) + 0.5 * Math.sqrt(Math.PI) * (1 - 2*x/3));
      return Math.max(0, Math.min(1, frac));
    });
    const c28 = getOrInitChart('chart-hbar-trapping', 'line', 'ALPHA trapped fraction vs H̄ temperature', 'Neutral H̄ Temperature [K]', 'Fraction Trapped [%]', true);
    c28.data.labels = hbarTempScan.map(t => t.toFixed(2));
    c28.data.datasets = [{
      label: 'Magnetic trap depth fraction',
      data: trapFracScan.map(v => v * 100 + 1e-4),
      borderColor: THEME.antiPink,
      backgroundColor: 'rgba(255, 107, 157, 0.08)',
      borderWidth: 2,
      fill: true,
      pointRadius: 0
    }];
    c28.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER ANH-02: Annihilation pion pt
    // ──────────────────────────────────────────────────────────────────────────────
    const c29 = getOrInitChart('chart-anni-pions', 'bar', 'Annihilation charged pion p_T distribution', 'Transverse Momentum p_T [GeV]', 'Events / 12 MeV');
    const piBins = Array.from({length: 40}, (_, i) => i * 0.015);
    const pipCounts = new Array(piBins.length).fill(0);
    anniPionsData.forEach(v => {
      const idx = Math.floor(v / 0.015);
      if (idx >= 0 && idx < piBins.length) pipCounts[idx]++;
    });
    c29.data.labels = piBins.map(v => v.toFixed(3));
    c29.data.datasets = [{
      label: 'Charged Pion p_T Spectrum',
      data: pipCounts,
      backgroundColor: THEME.antiPink,
      borderColor: '#ffffff',
      borderWidth: 0.5
    }];
    c29.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER ANH-03: Annihilation neutral pion decay gamma E
    // ──────────────────────────────────────────────────────────────────────────────
    const c30 = getOrInitChart('chart-anni-gamma', 'bar', 'Annihilation decay photon energy distribution', 'Photon Energy E_γ [GeV]', 'Events / 10 MeV');
    const gBins = Array.from({length: 40}, (_, i) => i * 0.0125);
    const gCounts = new Array(gBins.length).fill(0);
    anniGammaData.forEach(v => {
      const idx = Math.floor(v / 0.0125);
      if (idx >= 0 && idx < gBins.length) gCounts[idx]++;
    });
    c30.data.labels = gBins.map(v => v.toFixed(3));
    c30.data.datasets = [{
      label: 'Photon Energy Spectrum',
      data: gCounts,
      backgroundColor: THEME.lhcGold,
      borderColor: '#ffffff',
      borderWidth: 0.5
    }];
    c30.update('none');

    // ──────────────────────────────────────────────────────────────────────────────
    // ANTIMATTER ANH-04: Annihilation vertex radial distribution
    // ──────────────────────────────────────────────────────────────────────────────
    const c31 = getOrInitChart('chart-anni-vertex', 'bar', 'Annihilation vertex displacement distribution', 'Radial displacement vertex r [mm]', 'Events / 50 μm');
    const vBins = Array.from({length: 40}, (_, i) => i * 0.05);
    const vCounts = new Array(vBins.length).fill(0);
    anniVertexData.forEach(v => {
      const idx = Math.floor(v / 0.05);
      if (idx >= 0 && idx < vBins.length) vCounts[idx]++;
    });
    c31.data.labels = vBins.map(v => v.toFixed(2));
    c31.data.datasets = [{
      label: 'Displaced Annihilation Vertex',
      data: vCounts,
      backgroundColor: '#00A859',
      borderColor: '#ffffff',
      borderWidth: 0.5
    }];
    c31.update('none');

    // Draw Canvas annihilation track starburst
    drawAnnihilationStarburst();
  }

  // ──────────────────────────────────────────────────────────────────────────────
  // ANTIMATTER ANNIHILATION TRACKS GENERATOR CANVAS
  // ──────────────────────────────────────────────────────────────────────────────
  function drawAnnihilationStarburst() {
    const canvas = document.getElementById('canvas-starburst');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;

    // Draw central trap background grid
    ctx.strokeStyle = '#101726';
    ctx.lineWidth = 1;
    for (let r = 20; r < 140; r += 25) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // Vertex annihilation explosion
    const nTracks = 10;
    const starRand = new SeededRandom(4321);
    
    // Draw pion tracks from center
    for (let i = 0; i < nTracks; i++) {
      const angle = starRand.uniform(0, 2 * Math.PI);
      const length = starRand.uniform(60, 130);
      const isNeutral = starRand.uniform(0, 1) > 0.65;
      
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      
      const xEnd = cx + length * Math.cos(angle);
      const yEnd = cy + length * Math.sin(angle);
      
      // Draw helices / curled charged tracks under trap solenoid B-field
      if (!isNeutral) {
        ctx.strokeStyle = i % 2 === 0 ? THEME.antiPink : THEME.antiCyan;
        ctx.lineWidth = 2.0;
        
        // Curved quadratic bezier to represent solenoidal helix deflection
        const curvature = starRand.uniform(15, 35) * (i % 2 === 0 ? 1 : -1);
        const mx = cx + (length / 2) * Math.cos(angle) + curvature * Math.cos(angle + Math.PI/2);
        const my = cy + (length / 2) * Math.sin(angle) + curvature * Math.sin(angle + Math.PI/2);
        
        ctx.quadraticCurveTo(mx, my, xEnd, yEnd);
        ctx.stroke();

        // Draw particle endpoints
        ctx.fillStyle = i % 2 === 0 ? THEME.antiPink : THEME.antiCyan;
        ctx.beginPath();
        ctx.arc(xEnd, yEnd, 3.5, 0, 2 * Math.PI);
        ctx.fill();
      } else {
        // Neutral pions decay into decay photons immediately (dotted path)
        ctx.strokeStyle = THEME.lhcGold;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 4]);
        ctx.lineTo(xEnd, yEnd);
        ctx.stroke();
        ctx.setLineDash([]);

        // Show photon split arrows
        const gAngle1 = angle - 0.2;
        const gAngle2 = angle + 0.2;
        ctx.strokeStyle = THEME.lhcGold;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        
        ctx.beginPath();
        ctx.moveTo(xEnd, yEnd);
        ctx.lineTo(xEnd + 25 * Math.cos(gAngle1), yEnd + 25 * Math.sin(gAngle1));
        ctx.moveTo(xEnd, yEnd);
        ctx.lineTo(xEnd + 25 * Math.cos(gAngle2), yEnd + 25 * Math.sin(gAngle2));
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Draw trapped antimatter core vertex
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
    ctx.shadowBlur = 10;
    ctx.shadowColor = THEME.antiPink;
    ctx.fill();
    ctx.shadowBlur = 0; // reset
  }

  // Simulation state flags
  // Bind Antimatter sliders — only update if simulation has been started
  const antimatterSliders = [
    inputPbarEnergy, inputPbarThickness, inputCoolBw,
    inputEcoolCurrent, inputHbarTemp, inputIoffeField
  ];
  antimatterSliders.forEach(slider => {
    slider.addEventListener('input', () => {
      if (antimatterSimStarted) requestAnimationFrame(updateAntimatterSimulation);
    });
  });
  inputPbarMat.addEventListener('change', () => {
    if (antimatterSimStarted) requestAnimationFrame(updateAntimatterSimulation);
  });

  // Re-Simulate Annihilations Button (declare before startAntimatterSimulation)
  const btnTriggerHbar = document.getElementById('btn-trigger-hbar');
  const hbarLoader = document.getElementById('hbar-loader');
  const hbarText = document.getElementById('hbar-text');
  let currentAnnSeed = 424242;

  // Antimatter Start Simulation handler (shared by sidebar + overlay buttons)
  function startAntimatterSimulation() {
    if (antimatterSimStarted) return;
    antimatterSimStarted = true;

    // Dismiss overlay
    const overlayAm = document.getElementById('antimatter-start-overlay');
    if (overlayAm) overlayAm.classList.add('hidden');

    // Enable the Re-Simulate button
    btnTriggerHbar.disabled = false;
    btnTriggerHbar.style.background = '';

    // Fetch from backend & render all charts
    updateAntimatterSimulation();
  }

  // Sidebar Start button
  document.getElementById('btn-start-antimatter').addEventListener('click', startAntimatterSimulation);
  // Overlay Start button
  document.getElementById('btn-start-antimatter-overlay').addEventListener('click', startAntimatterSimulation);

  btnTriggerHbar.addEventListener('click', async () => {
    hbarLoader.style.display = 'inline-block';
    hbarText.innerText = 'Calculating decays...';
    btnTriggerHbar.disabled = true;

    currentAnnSeed = Math.floor(Math.random() * 999999);
    await updateAntimatterSimulation();

    hbarLoader.style.display = 'none';
    hbarText.innerText = 'Re-Simulate Annihilations';
    btnTriggerHbar.disabled = false;
  });

  // ──────────────────────────────────────────────────────────────────────────────
  // INITIAL RUN STATE — just switch to LHC tab, show overlay, await user click
  // ──────────────────────────────────────────────────────────────────────────────
  switchTab('lhc');

})();
