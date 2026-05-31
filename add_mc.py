import re
with open('server.py', 'r') as f:
    content = f.read()

mc_code = """    # 12. MC Events
    from detector_sim import PPEventGenerator, ATLASLikeDetector
    gen = PPEventGenerator(14.0, 42)
    det = ATLASLikeDetector()
    rng = np.random.default_rng(10101)

    z_events = gen.generate_events(1200, 'Z')
    z_masses = det.reconstruct_Z_mass(z_events, rng).tolist()

    h_events = gen.generate_events(900, 'Higgs')
    h_masses = []
    for evt in h_events:
        if len(evt) < 2: continue
        p1s = det.smear_particle(evt[0], rng)
        p2s = det.smear_particle(evt[1], rng)
        if det.is_accepted(p1s) and det.is_accepted(p2s):
            m2 = ((p1s.E+p2s.E)**2 - (p1s.px+p2s.px)**2 - (p1s.py+p2s.py)**2 - (p1s.pz+p2s.pz)**2)
            if m2 > 0: h_masses.append(float(np.sqrt(m2)))

    dijet_events = gen.generate_events(1800, 'dijet')
    dijet_pt = [max(p.pt for p in evt) for evt in dijet_events if evt]

"""

content = content.replace("    return {\n        \"header\"", mc_code + "    return {\n        \"mc\": {\"zMasses\": z_masses, \"hMasses\": h_masses, \"dijetPt\": dijet_pt},\n        \"header\"")

with open('server.py', 'w') as f:
    f.write(content)
