import re

def refactor_app_js(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove simulateMCEvents call
    content = content.replace("  function simulateMCEvents() {", "  async function simulateMCEvents() { return; /* Handled by backend */")
    
    # 1. Refactor updateLhcSimulation
    # Replace synchronous declaration with async fetch
    lhc_start = """  async function updateLhcSimulation() {
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
            body: JSON.stringify({
                energy, bunchPop, emittVal, betaStar, crossingAngle, rfVolt, primarySigma, quenchR
            })
        });
        data = await response.json();
    } catch(e) { console.error(e); return; }
"""
    # Replace the beginning of updateLhcSimulation
    content = re.sub(r'  function updateLhcSimulation\(\) \{[\s\S]*?const quenchR = parseFloat\(inputQuenchR\.value\);', lhc_start, content)

    # Remove `const beam = new CERNPhysics.LHCBeam(...)`
    content = re.sub(r'    // 3\. Construct core Beam State\n    const beam = new CERNPhysics\.LHCBeam[^\n]+\n', '', content)

    # Header Metrics
    content = content.replace("beam.peakLuminosity()", "data.header.peakLuminosity")
    content = content.replace("beam.bunchPopulation", "data.header.bunchPopulation")
    content = content.replace("beam.sigmaIP", "data.header.sigmaIP")
    content = content.replace("const storedE = (CERNPhysics.LHC_BUNCHES * bunchPop * energy * CERNPhysics.e_charge) / 1e6; // in MJ", "const storedE = data.header.storedEnergy;")

    # Table rows
    content = content.replace("const summary = beam.summary();", "const summary = data.summary;")
    content = re.sub(r"const tableRows = \[\s*\{ name: 'Beam Operating Energy'[\s\S]*?\];", "const tableRows = data.summary;", content)
    content = content.replace("<td>${row.name}</td><td>${row.unit}</td><td class=\"val\">${row.val}</td>", "<td>${row.name}</td><td>${row.unit}</td><td class=\"val\">${typeof row.val === 'number' ? row.val.toPrecision(4) : row.val}</td>")
    content = content.replace("summary.forEach", "tableRows.forEach")

    # c1: Injector chain
    content = content.replace("const injData = CERNPhysics.simulateInjectionChain();", "const injData = data.inj;")

    # c2: FODO Lattice
    content = re.sub(r'const fodo = new CERNPhysics\.FODOCell[^;]+;\n\s*const twiss = fodo\.twissAlongCell[^;]+;', 'const twiss = data.fodo;', content)

    # c3: Phase ellipse
    content = re.sub(r'const phase7T = beam\.emittance\.phaseEllipse[^;]+;\n\s*const phase450G = beam\.emittance\.phaseEllipse[^;]+;', 'const phase7T = {x: data.phaseEllipse.x7T, xp: data.phaseEllipse.xp7T};\n    const phase450G = {x: data.phaseEllipse.x450G, xp: data.phaseEllipse.xp450G};', content)

    # c4: Betatron
    content = re.sub(r'const bOsc = CERNPhysics\.trackBetatron[^;]+;', 'const bOsc = data.betatron;', content)

    # c5: Radiation
    content = re.sub(r'const eScanP = Array\.from[^;]+;\n\s*const rhoP = CERNPhysics\.LHC_CIRCUMFERENCE[^;]+;\n\s*const uScanP = eScanP\.map[^;]+;', 'const eScanP = data.radiation.eScan;\n    const uScanP = data.radiation.uScan;', content)

    # c6: RF Phase space
    content = re.sub(r'const rf = new CERNPhysics\.RFCavity[^;]+;\n\s*const rfSpace1 = rf\.bunchEvolution[^;]+;\n\s*const rfSpace2 = rf\.bunchEvolution[^;]+;', 'const rfSpace1 = {phi: data.rfSpace.phi1, delta: data.rfSpace.delta1};\n    const rfSpace2 = {phi: data.rfSpace.phi2, delta: data.rfSpace.delta2};', content)

    # c7: Lum E
    content = re.sub(r'const eScanL = Array\.from[^;]+;\n\s*const lScanE = eScanL\.map\([^\}]+\}\);', 'const eScanL = data.lumEnergy.eScan;\n    const lScanE = data.lumEnergy.lScan;', content)

    # c8: Lum Beta
    content = re.sub(r'const bsScan = Array\.from[^;]+;\n\s*const lScanBS = bsScan\.map\([^\}]+\}\);', 'const bsScan = data.lumBeta.bsScan;\n    const lScanBS = data.lumBeta.lScan;', content)

    # c9: Ramp
    content = re.sub(r'const ramp = CERNPhysics\.energyRampLHC[^;]+;', 'const ramp = data.ramp;', content)

    # c10: Quench
    content = re.sub(r'const qp = new CERNPhysics\.QuenchProtection[^;]+;\n\s*const tQuench = Array\.from[^;]+;\n\s*const iQuench = tQuench\.map[^;]+;', 'const tQuench = data.quench.t;\n    const iQuench = data.quench.I;', content)

    # c11: Beam Beam
    content = re.sub(r'const nScanT = Array\.from[^;]+;\n\s*const xiScan = nScanT\.map[^;]+;', 'const nScanT = data.beamBeam.nScan;\n    const xiScan = data.beamBeam.xiScan;', content)

    # c13: PDF
    content = re.sub(r'const xPdf = Array\.from[^;]+;\n\s*const pdfU = xPdf\.map[^;]+;\n\s*const pdfD = xPdf\.map[^;]+;\n\s*const pdfG = xPdf\.map[^;]+;', 'const xPdf = data.pdf.x;\n    const pdfU = data.pdf.u;\n    const pdfD = data.pdf.d;\n    const pdfG = data.pdf.g;', content)

    # MC variables
    content = content.replace("mcZMassData", "data.mc.zMasses")
    content = content.replace("mcHMassData", "data.mc.hMasses")
    content = content.replace("mcDijetPtData", "data.mc.dijetPt")

    # 2. Refactor updateAntimatterSimulation
    am_start = """  async function updateAntimatterSimulation() {
    const pbarEnergy = parseFloat(inputPbarEnergy.value);
    const pbarMat = inputPbarMat.value;
    const pbarThickness = parseFloat(inputPbarThick.value);
    const coolBw = parseFloat(inputCoolBw.value);
    const hbarTemp = parseFloat(inputHbarTemp.value);
    const ioffeField = parseFloat(inputIoffeB.value);
    const annSeed = parseInt(inputAnnSeed.value);
    const ecoolCurrent = parseFloat(inputEcoolI.value);
    const posEnergy = parseFloat(inputPosEnergy.value);
    const posTargetT = parseFloat(inputPosThick.value);

    let data = null;
    try {
        const response = await fetch('http://localhost:8001/api/antimatter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pbarEnergy, pbarMat, pbarThickness, coolBw, hbarTemp, ioffeField, annSeed, ecoolCurrent, posEnergy, posTargetT
            })
        });
        data = await response.json();
    } catch(e) { console.error(e); return; }
"""
    content = re.sub(r'  function updateAntimatterSimulation\(\) \{[\s\S]*?const posTargetT = parseFloat\(inputPosThick\.value\);', am_start, content)

    # AM logic
    content = re.sub(r'const data = CERNPhysics\.simulateAnnihilation\([^;]+\);', 'const annData = data.annihilation;\n    const data_pts = annData.x.map((x,i) => ({x:x, y:annData.y[i]}));', content)
    content = content.replace("c23.data.datasets[0].data = data;", "c23.data.datasets[0].data = data_pts;")
    
    content = re.sub(r'const yieldVal = CERNPhysics\.pbarYieldPerProton[^;]+;', 'const yieldVal = data.summary.yield;', content)
    content = re.sub(r'const decelChain = CERNPhysics\.simulateDecelerationChain\(\);', 'const decelChain = data.decel;', content)

    # SC, AH, Trap, EC, PS declarations
    content = re.sub(r'const sc = new CERNPhysics\.StochasticCooling[^;]+;', '', content)
    content = re.sub(r'const ah = new CERNPhysics\.AntihydrogenFormation[^;]+;', '', content)
    content = re.sub(r'const trap = new CERNPhysics\.IoffeTrap[^;]+;', '', content)
    content = re.sub(r'const ec = new CERNPhysics\.ElectronCooling[^;]+;', '', content)
    content = re.sub(r'const fRevDecel = CERNPhysics\.betaFactor[^;]+;\n\s*const fRev = fRevDecel;', '', content)
    content = re.sub(r'const ps = new CERNPhysics\.PositronSource\(\);', '', content)

    # AM Arrays
    content = re.sub(r'const tScanPbar = Array\.from[^;]+;\n\s*const xsScan = tScanPbar\.map[^;]+;', 'const tScanPbar = data.scans.tScanPbar;\n    const xsScan = data.scans.xsScan;', content)
    content = re.sub(r'const yieldScan = thickScan\.map[^;]+;', 'const yieldScan = data.scans.yieldScan;', content)
    content = re.sub(r'const thickScan = Array\.from[^;]+;', 'const thickScan = data.scans.thickScan;', content)

    content = re.sub(r'const tScan = Array\.from[^;]+;\n\s*const dpScan = tScan\.map[^;]+;', 'const tScan = data.scans.scTime;\n    const dpScan = data.scans.scDp;', content)
    content = re.sub(r'const tScanEC = Array\.from[^;]+;\n\s*const coolRate = ec\.coolingRate[^;]+;\n\s*const dpScanEC = tScanEC\.map[^;]+;', 'const tScanEC = data.scans.ecTime;\n    const dpScanEC = data.scans.ecDp;', content)

    content = re.sub(r'const eScan = Array\.from[^;]+;\n\s*const spec = eScan\.map[^;]+;', 'const eScan = data.scans.psE;\n    const spec = data.scans.psSpec;', content)
    content = re.sub(r'const rScan = Array\.from[^;]+;\n\s*const bScan = rScan\.map[^;]+;', 'const rScan = data.scans.trapR;\n    const bScan = data.scans.trapB;', content)
    content = re.sub(r'const tempScan = Array\.from[^;]+;\n\s*const rateScan = tempScan\.map[^;]+;', 'const tempScan = data.scans.tHbar;\n    const rateScan = data.scans.ahRate;', content)

    with open(filepath, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    refactor_app_js('/Users/satwikdubey/Documents/CERN proj/app.js')
