import re

with open('server.py', 'r') as f:
    content = f.read()

# Remove simulate_annihilation from imports
content = content.replace("simulate_annihilation", "")

# Add simulate_annihilation definition
ann_def = """
def simulate_annihilation(n_particles: int, seed: int):
    rng = np.random.default_rng(seed)
    r = rng.exponential(0.1, n_particles)
    theta = rng.uniform(0, 2*np.pi, n_particles)
    return [{'x': float(r[i]*np.cos(theta[i])), 'y': float(r[i]*np.sin(theta[i]))} for i in range(n_particles)]

"""
content = content.replace("app = FastAPI()", ann_def + "app = FastAPI()")

with open('server.py', 'w') as f:
    f.write(content)
