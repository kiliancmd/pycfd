# pycfd

A 2D incompressible Navier–Stokes solver built around Chorin's projection method
on a staggered (MAC) grid. It runs the classic benchmarks — lid-driven cavity,
Poiseuille channel, flow past a cylinder, Taylor–Green vortex — with live
visualisation, quantitative validation against published data, and a grid
convergence study that confirms second-order spatial accuracy.

This is a teaching and research-prototype code, comparable to an early-career
research tool. It is not a production CFD suite: no 3D, no complex geometry, no
industrial turbulence modelling.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. `numba` and `tqdm` are optional — the solver falls back to
pure NumPy stencils and a silent run if they are missing.

## Quickstart

```bash
python -m pycfd.main --list-cases
```

```bash
python -m pycfd.main --case cavity --re 1000 --nx 128 --live
```

```bash
python -m pycfd.main --case cylinder --re 100 --live --display vorticity
```

```bash
python -m pycfd.main --convergence
```

Every case also runs headlessly and writes 300 DPI figures plus a validation
report:

```bash
python -m pycfd.main --case cavity --re 400 --export-vtk --checkpoint
```

In the live window, `space` pauses and resumes, `q` closes.

### As a library

```python
from pycfd.config import BCKind, BCSpec, SimulationConfig
from pycfd.physics.incompressible import Simulation

cfg = SimulationConfig(
    nx=128, ny=128, re=1000.0, t_end=30.0, steady_tol=1e-6,
    boundary_config={
        "left":   BCSpec(BCKind.NO_SLIP),
        "right":  BCSpec(BCKind.NO_SLIP),
        "bottom": BCSpec(BCKind.NO_SLIP),
        "top":    BCSpec(BCKind.MOVING_WALL, velocity=1.0),
    },
)
sim = Simulation(cfg)
result = sim.run(progress=True)
print(sim.diagnostics())
```

### Example output

Running a case writes 300 DPI figures under `results/<case>/`:

| file | contents |
|---|---|
| `cavity/cavity_Re100_128x128_fields.png` | 4-panel: speed + streamlines, pressure, vorticity, stream function |
| `cavity/cavity_Re100_128x128_centerlines.png` | centreline profiles overlaid with the Ghia et al. points |
| `channel/channel_periodic_Re10_32x64_profile.png` | computed vs analytical parabola, with the point-wise error below |
| `convergence/taylor_green_convergence.png` | log-log error vs resolution against a second-order slope |
| `cylinder/cylinder_Re100_256x128_fields.png` | the vortex street, with the cylinder outlined |
| `cylinder/cylinder_Re100_256x128_forces.png` | Cd and Cl time histories |

The Re = 100 cavity reproduces the standard result closely: the primary vortex
centre lands at ≈ (0.62, 0.74) against Ghia's (0.6172, 0.7344), both bottom
corner vortices are resolved, and the stream-function minimum is ≈ −0.105
against a published −0.1034.

---

---

## Documentation

| guide | what it covers |
|---|---|
| **[Usage](docs/usage.md)** | The full five-step workflow — geometry, meshing, boundary conditions, running, results — plus the complete CLI reference for all 46 flags and worked recipes |
| **[Numerical method](docs/numerics.md)** | Staggered MAC grid, SSP-RK3 time integration, the pressure Poisson solve, and why each choice was made |
| **[Validation](docs/validation.md)** | Measured agreement against Ghia et al., Poiseuille, and Taylor–Green, with the grid-convergence study |
| **[Performance](docs/performance.md)** | Measured Numba speed-ups, and the honest finding that the pressure solve — not the stencils — is the bottleneck |
| **[Development](docs/development.md)** | Test suite, regression baselines, CI, project layout, and known limitations |

## At a glance

| property | measured |
|---|---|
| spatial accuracy | **2.000** observed order (Taylor–Green, 16→128) |
| incompressibility | `max \|div u\|` at machine precision (~1e-14) |
| cavity vs Ghia et al. | L2 ≤ 0.015 at Re = 100 / 400 / 1000 |
| Poiseuille (periodic) | centreline exact to **1.0e-07 %** |
| test suite | 317 tests, plus 6 full-fidelity benchmark regressions |

Every number above is pinned by a test rather than by prose — see
[regression baselines](docs/development.md#regression-baselines).

## Reproducibility

Each figure, VTK dump and checkpoint carries a record of the run that produced
it: the exact command, the configuration, the pycfd version and the git commit.
It travels inside the file where the format allows (PNG `tEXt` chunks, the VTK
title line, a field in the `.npz`) and in a `<name>.provenance.json` sidecar
alongside. So a result found later can always answer *what made this*:

```bash
python -c "import json; print(json.load(open('results/run.provenance.json'))['command'])"
# python -m pycfd.main --case cylinder --re 100 --geometry shield.csv --name run
```
