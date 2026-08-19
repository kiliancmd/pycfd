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

## Usage guide

The workflow is the usual CFD sequence: bring in a shape, mesh the domain around
it, prescribe what happens at the edges, solve, then look at the answer.

#### Where each step lives

**You do not edit the package to set up a run.** A case is *data*: you build a
`SimulationConfig` (and optionally an `Obstacle`), hand them to `Simulation`, and
call `run()`. The table below says which file *defines* each piece, so you know
where to look when you need the exact signature or default.

| step | file | what it provides |
|---|---|---|
| **1. 2D model** | `geometry/obstacles.py` | `load_polygon`, `transform_polygon`, `polygon_mask`, `mask_from_image`, `mask_from_function`, `circle_mask`, `rectangle_mask`, and the `Obstacle` they all return |
| **2. mesh** | `core/mesh.py` | `StructuredMesh` — cell coordinates, spacing, and the staggered index convention (documented in its module docstring) |
| | `config.py` | the `nx`, `ny`, `lx`, `ly` fields the mesh is built from |
| **3. boundaries** | `config.py` | `SimulationConfig` (every setting), `BCSpec`, `BCKind`, and the `TimeScheme` / `AdvectionScheme` / `PressureSolver` enums |
| | `core/boundary.py` | the implementations — `NoSlip`, `MovingWall`, `Inlet`, `Outlet`, `Symmetry`, `Periodic`. You name these through `config.BCSpec`; you rarely import them directly |
| **4. run** | `physics/incompressible.py` | `Simulation` — the object you actually drive |
| | `core/solver.py` | `ProjectionSolver` — the fractional-step algorithm |
| | `core/timestepper.py` | `TimeStepper` (adaptive `dt`, the run loop), `SimulationResult`, `DivergenceError` |
| | `main.py` | the `argparse` CLI, if you would rather not write Python |
| **5. results** | `analysis/postprocess.py` | `vorticity`, `stream_function`, `kinetic_energy`, `enstrophy`, `force_coefficients`, `strouhal_number`, `sample_at`, `Probe` |
| | `analysis/export.py` | `export_vtk`, `export_csv`, `save_checkpoint`, `load_checkpoint` |
| | `visualization/static_plot.py` | `four_panel_figure`, `centerline_comparison_figure`, `profile_comparison_figure`, `convergence_figure`, `time_series_figure` |
| | `visualization/live_plot.py` | `LiveViewer` — the real-time window |

Two more files are worth knowing about. `analysis/validation.py` holds the
analytical solutions and the Ghia et al. reference data, if you want to check a
result against something known. And **`cases/` is the best starting point for a
new problem** — each file there is a complete, working setup of all five steps:

| file | setup it demonstrates |
|---|---|
| `cases/lid_driven_cavity.py` | closed box, one moving wall, steady-state detection |
| `cases/channel_flow.py` | periodic + body force, and inlet/outlet |
| `cases/cylinder_flow.py` | **external flow past a body** — copy this one for custom geometry |
| `cases/taylor_green.py` | doubly periodic, exact initial condition, refinement study |

#### Notation

The names used throughout the code and this guide:

*Grid and geometry*

| symbol | meaning |
|---|---|
| `nx`, `ny` | number of **cells** in x and y (not grid points) |
| `lx`, `ly` | domain extent, so the domain is `[0, lx] x [0, ly]` |
| `dx`, `dy` | cell size, `lx/nx` and `ly/ny` — uniform by construction |
| `mask` | boolean `(nx, ny)` array, `True` inside a solid body |
| `fraction` | solid **volume fraction** of each cell, in `[0, 1]` — a sub-cell measure of how much of the cell the body covers |
| `characteristic_length` | the body's reference length `L` (a cylinder's diameter); forms `Re` and normalises `Cd`, `Cl` |

*Flow variables*

| symbol | meaning |
|---|---|
| `u`, `v` | velocity components along x and y |
| `p` | pressure divided by density (kinematic pressure) — only its gradient is physical, so its mean is set to zero |
| `nu` (ν) | kinematic viscosity, **derived** as `u_ref * l_ref / re` |
| `re` (Re) | Reynolds number — ratio of inertial to viscous forces |
| `u_ref`, `l_ref` | the reference velocity and length that define `nu`; they are what make `re` mean what you intend |
| `body_force` | `(fx, fy)` force per unit mass, added to the momentum equation |
| `omega` (ω) | vorticity, `dv/dx - du/dy` — local rate of rotation |
| `psi` (ψ) | stream function; contours of `psi` are streamlines |
| `Cd`, `Cl` | drag and lift coefficients, `F / (0.5 * u_ref^2 * l_ref)` in 2D |
| `St` | Strouhal number, `f * l_ref / u_ref` — dimensionless shedding frequency |

*Time stepping*

| symbol | meaning |
|---|---|
| `dt` | time step; under adaptive stepping also the largest step allowed |
| `t_end` | simulated time at which to stop |
| `cfl_max` | upper bound on the **Courant number** `dt * (max\|u\|/dx + max\|v\|/dy)`, i.e. how many cells information may cross in one step |
| `steady_tol` | stop early once `max\|du/dt\|` falls below this |
| `max_steps` | hard cap on iterations, whatever the clock says |

### 1. Bringing in a 2D model

> **Where:** `geometry/obstacles.py`. CLI equivalent: `--geometry FILE`.

Geometry is represented as an **obstacle mask**: cells inside the body are
marked solid, and every staggered face touching one is held at zero velocity.
There are four ways to produce one, all returning the same `Obstacle`.

**A vertex file** — two columns of `x y`, comma- or whitespace-separated.
Comments (`#`) and blank lines are ignored, and a repeated closing vertex is
optional. This is the format most CAD and airfoil tools export:

```
# NACA 0018 outline, chord 1
0.000000, 0.000000
0.012658, 0.028578
0.025316, 0.039447
...
```

```python
from pycfd.core.mesh import StructuredMesh
from pycfd.geometry.obstacles import load_polygon, polygon_mask, transform_polygon

mesh = StructuredMesh(nx=256, ny=128, lx=16.0, ly=8.0)
outline = load_polygon("aerofoil.csv")
outline = transform_polygon(outline, scale=2.0, center=(4.0, 4.0), rotate_deg=-10.0)
body = polygon_mask(mesh, outline, name="aerofoil")
```

`transform_polygon` scales and rotates about the shape's own centroid and then
places that centroid at `center`, so one file can be reused across domains of
different size without editing it. Concave outlines are handled correctly (the
test suite covers an L-shape); winding order does not matter.

**A bitmap silhouette** — dark pixels are solid, so a black shape on a white
background works with no preparation:

```python
from pycfd.geometry.obstacles import mask_from_image

body = mask_from_image(mesh, "shape.png", threshold=0.5)   # invert=True to flip
```

The image is stretched across the **whole domain**, so match its aspect ratio to
`lx : ly` or the shape will be distorted. A 400×200 picture on a 16×8 domain is
consistent; the same picture on a 16×16 domain is stretched 2:1.

**A formula**, when the shape is easier to write than to draw:

```python
from pycfd.geometry.obstacles import mask_from_function

body = mask_from_function(
    mesh, lambda x, y: ((x - 4) / 1.0) ** 2 + ((y - 4) / 0.3) ** 2 <= 1.0,
    characteristic_length=0.6,          # used to form Re, Cd and Cl
)
```

**Built-in primitives** — `circle_mask(mesh, center, radius)` and
`rectangle_mask(mesh, lower_left, upper_right)`.

The arguments these share:

| argument | meaning |
|---|---|
| `mesh` | the `StructuredMesh` the mask is built on — it must be the **same mesh** the simulation uses |
| `center` | where to place the shape, in domain coordinates `(x, y)` |
| `scale` | multiplier applied about the shape's own centroid; `2.0` doubles it |
| `rotate_deg` | rotation about the centroid, in degrees, anticlockwise |
| `threshold` | for images: luminance in `[0, 1]` below which a pixel counts as solid |
| `invert` | for images: treat *light* pixels as solid instead |
| `subsamples` | sub-samples per cell edge used to estimate the volume fraction (default 8). Higher is more accurate at the surface and slower |
| `characteristic_length` | the reference length `L` for `Re`, `Cd` and `Cl` |
| `name` | label used in figure titles and output filenames |

All four estimate each cell's **solid volume fraction** by supersampling rather
than testing the cell centre, so the staircase follows the true surface as
closely as a cell-centred mask can. `obstacle.fraction` holds those fractions and
integrates to the true area to a fraction of a percent.

> `characteristic_length` is not cosmetic: it is the length that forms the
> Reynolds number and normalises Cd and Cl. It defaults to the shape's height
> (its extent across the flow), matching the cylinder-diameter convention. Set
> it explicitly if you want coefficients based on chord instead.

There is **no 3D import and no STL/DXF reader** — this is a 2D solver, and a
2D outline or silhouette is the whole model.

### 2. Meshing

> **Where:** `core/mesh.py` defines `StructuredMesh`; the four numbers that
> determine it are `nx`, `ny`, `lx`, `ly` on `SimulationConfig` in `config.py`.

The mesh is a uniform structured Cartesian grid, generated from the domain size
and cell counts; there is no meshing step to run and nothing to check for
skewness or aspect-ratio quality. You choose four numbers:

```python
SimulationConfig(nx=256, ny=128, lx=16.0, ly=8.0)   # dx = dy = 0.0625
```

Keep `lx/nx == ly/ny` unless you deliberately want anisotropic cells — the
operators are second-order either way, but a strongly stretched cell resolves
the two directions unequally.

Three rules of thumb decide the numbers:

| question | guidance |
|---|---|
| how fine? | **≥ 16 cells across the body.** Below that, forces get crude fast; the solver logs a warning under 8. |
| how wide? | **blockage `L/ly` ≤ 5%.** Confining walls raise Cd measurably — see the cylinder table above. |
| how long? | **≥ 4 body lengths upstream, ≥ 10 downstream** so the wake leaves cleanly. |

Check what you actually got before committing to a long run:

```python
print(mesh)          # StructuredMesh(256x128 cells, domain [0,16]x[0,8], uniform, dx=0.0625, dy=0.0625)
print(body)          # Obstacle('aerofoil', 412 solid cells, L=0.694)
print(body.characteristic_length / mesh.dy, "cells across the body")
```

`StructuredMesh` also supports geometric stretching (`stretch_x`, `stretch_y`),
but **the solver will refuse to run on a stretched mesh** — the finite-difference
operators are derived for uniform spacing, and it raises `NonUniformMeshError`
rather than quietly returning a first-order answer.

### 3. Boundary conditions and simulation settings

> **Where:** `config.py` — `SimulationConfig`, `BCSpec` and `BCKind`. The
> behaviour behind each kind lives in `core/boundary.py`.

Boundaries are a dict over the four walls. Every wall needs an entry:

```python
from pycfd.config import BCKind, BCSpec, SimulationConfig

boundary_config = {
    "left":   BCSpec(BCKind.INLET, velocity=1.0, profile="uniform"),
    "right":  BCSpec(BCKind.OUTLET),
    "bottom": BCSpec(BCKind.SYMMETRY),
    "top":    BCSpec(BCKind.SYMMETRY),
}
```

| kind | meaning | typical use |
|---|---|---|
| `NO_SLIP` | both components zero at the wall | solid walls |
| `MOVING_WALL` | wall slides in its own plane at `velocity` | the cavity lid |
| `INLET` | prescribed inflow, `profile="uniform"` or `"parabolic"` | upstream boundary |
| `OUTLET` | zero-gradient outflow, rescaled to conserve mass | downstream boundary |
| `PRESSURE_OUTLET` | outflow at fixed pressure `p = p_ref` | external aerodynamics, open domains |
| `SYMMETRY` | no through-flow, free slip | far-field, or a mirror plane |
| `PERIODIC` | wraps to the opposite wall | streamwise-periodic channels |

`velocity` means the *tangential* wall speed for `MOVING_WALL` and the *inflow*
speed for `INLET` (signed into the domain, so the same spec works on any wall).
Periodicity must be declared on **both** walls of an axis — declaring one raises
a validation error rather than silently doing something else.

#### `PRESSURE_OUTLET` — anchoring the pressure instead of the velocity

**What it is.** A Dirichlet pressure condition on the outflow boundary: the
static pressure on that face is held at `p_ref` (default `0.0`), and the
velocity leaving through it is whatever the pressure field produces.

**How it differs from `OUTLET`.** The two fix opposite things:

| | `OUTLET` | `PRESSURE_OUTLET` |
|---|---|---|
| velocity at the boundary | prescribed by extrapolation | **solved for** |
| pressure at the boundary | floats (`dp/dn = 0`) | **fixed at `p_ref`** |
| pressure level of the domain | arbitrary — pinned at a reference cell, mean removed | absolute and physical |
| Poisson operator | singular (constant null space) | non-singular |
| mass conservation | enforced by rescaling the outflow | automatic |

That last row is worth spelling out. Because every fluid cell is driven to zero
divergence, the divergence theorem makes the net boundary flux exactly zero, so
the outflow matches the inflow with no rescaling — measured imbalance is `0.0`,
not merely small.

**When to use it.** External aerodynamics and any open domain, where the far
field is at a known *pressure* rather than a known velocity. It is the default
for the cylinder case. Use plain `OUTLET` when you genuinely want to impose the
outflow profile — a duct whose downstream flow rate you are setting.

**Configuration:**

```python
boundary_config = {
    "left":   BCSpec(BCKind.INLET, velocity=1.0),
    "right":  BCSpec(BCKind.PRESSURE_OUTLET, p_ref=0.0),   # anchors p = 0
    "bottom": BCSpec(BCKind.SYMMETRY),
    "top":    BCSpec(BCKind.SYMMETRY),
}
```

`p_ref` is a datum — only pressure differences drive the flow, so `0.0` simply
makes the far field the zero of the reported field. The condition works on any
wall, not just the right one.

**Switching it from the command line.** You do not have to edit a case file to
change the outflow condition:

```bash
python -m pycfd.main --case cylinder --re 100 --outlet-type pressure_outlet --p-ref 0
python -m pycfd.main --case cylinder --re 100 --outlet-type outlet
python -m pycfd.main --case channel --mode developing --outlet-type pressure_outlet
```

| flag | effect |
|---|---|
| `--outlet-type outlet` | velocity outlet: extrapolate the outflow, let the pressure float |
| `--outlet-type pressure_outlet` | pressure outlet: hold `p = p_ref`, solve for the outflow |
| `--p-ref P` | the anchored pressure (default `0.0`); keeps the existing kind if `--outlet-type` is omitted |

Omitting both leaves each case's own choice — a pressure outlet for `cylinder`,
a velocity outlet for `channel --mode developing`. Only walls already carrying
an outflow condition are retyped, so an inlet or symmetry plane can never be
clobbered by accident, and a case with no outflow at all (`cavity`,
`taylor_green`, `channel --mode periodic`) rejects the flags rather than
ignoring them. When a pressure outlet is active the run reports `outlet_p_ref`
and `outlet_p_deviation`, so you can see the anchor took hold.

**Keep the inlet Neumann.** pycfd's `INLET` imposes `dp/dn = 0` by construction,
which is what you want alongside a pressure outlet: the inflow *velocity* is
prescribed, so the inlet pressure must be free to adjust. Prescribing the
pressure at both ends *and* the inlet velocity over-determines the flow — the
pressure drop and the flow rate are not independent. (The Poisson system itself
stays perfectly well posed with Dirichlet at both ends; that combination is the
standard way to drive a channel by a pressure difference, but then the inlet
velocity must not also be imposed.)

**Implementation note.** The condition is applied at the boundary **face**,
through the ghost relation `p_ghost = 2*p_ref - p_interior`, rather than by
overwriting the last cell's equation with `p = p_ref`. Anchoring the cell centre
instead would place the datum half a cell inside the domain, drop the scheme to
first order, and — decisively — discard that cell's continuity equation, so the
column beside the outlet would stop being divergence-free. The face formulation
keeps `max |div u|` at round-off right up to the boundary.

Pressure needs no boundary condition from you: `dp/dn = 0` is imposed at every
non-periodic wall (except a `PRESSURE_OUTLET`), which is a consistency
requirement of the projection rather than a modelling choice.

The remaining settings, all on `SimulationConfig`:

| field | default | what it means |
|---|---|---|
| `re` | 100 | **Reynolds number** — inertia over viscosity. Sets `nu = u_ref*l_ref/re`; higher means thinner boundary layers and a finer grid needed |
| `u_ref`, `l_ref` | 1.0, 1.0 | **reference velocity and length** that give `re` its meaning. **Set these** whenever the body is not unit-sized, or the Reynolds number you get is not the one you asked for |
| `dt` | 1e-3 | **time step** — the initial value, **and the ceiling** under adaptive stepping (see the note below) |
| `adaptive_dt` | True | choose `dt` each step from the stability limits instead of holding it fixed |
| `cfl_max` | 0.5 | **Courant limit** — how far information may travel per step, in cells. Lower is safer and slower; > 1 is unstable |
| `t_end` | 10.0 | **stop time** in simulated seconds |
| `max_steps` | None | hard cap on iterations regardless of `t_end`; `None` means no cap |
| `steady_tol` | None | **steady-state threshold** — stop once `max\|du/dt\|` falls below it. `None` runs the full `t_end` |
| `time_scheme` | `rk3` | **time integrator**. `euler` and `rk2` exist but are unstable for central advection |
| `advection_scheme` | `central` | **convective discretisation**. `central` is 2nd-order and non-diffusive; `upwind` adds damping when a high-Re run will not stay bounded |
| `pressure_solver` | `direct` | **linear solver** for the pressure equation: `direct` (factorised once), or `cg` / `sor` / `jacobi` |
| `body_force` | (0, 0) | **force per unit mass** `(fx, fy)` applied everywhere — this is what drives the periodic channel |
| `use_les` | False | enable the **Smagorinsky sub-grid model** for under-resolved turbulence |
| `stretch_x`, `stretch_y` | 1.0, 1.0 | geometric cell-growth ratios. The mesh supports them; **the solver does not** and will raise |
| `name` | `"simulation"` | label used in logs, figure titles and output filenames |

Invalid values are rejected at construction with a specific message, so a bad
configuration cannot reach the solver.

> **`dt` is a ceiling, not just a starting value.** With `adaptive_dt=True` the
> step is `min(convective limit, viscous limit, dt)`, so leaving `dt` at its
> `1e-3` default caps every step at `1e-3` even when the physics would allow far
> more. That is deliberate — it stops a quiescent start from taking one enormous
> first step — but it means a coarse-grid run can end up an order of magnitude
> slower than necessary. Set `dt` to the largest step you would accept (`0.02`
> is reasonable for a unit-velocity external flow) and let the CFL and viscous
> limits do the rest.

### 4. Running the simulation

> **Where:** `physics/incompressible.py` (`Simulation`), driving
> `core/solver.py` and `core/timestepper.py`. CLI: `main.py`.

From Python:

```python
from pycfd.physics.incompressible import Simulation

cfg = SimulationConfig(
    nx=256, ny=128, lx=16.0, ly=8.0,
    re=100.0, u_ref=1.0, l_ref=body.characteristic_length,
    t_end=120.0, cfl_max=0.4, boundary_config=boundary_config,
    name="aerofoil_Re100",
)
sim = Simulation(cfg, obstacle=body, u_init=1.0)     # start from uniform flow
result = sim.run(progress=True)
print(result.summary())
```

Or from the command line, with the geometry loaded for you:

```bash
python -m pycfd.main --case cylinder --re 100 --geometry aerofoil.csv --geometry-scale 2 --geometry-rotate -10
```

`--geometry` puts your body into the same uniform-inflow / outflow / symmetry
configuration the cylinder benchmark uses, which is the usual external-flow
setup. It accepts `--geometry-scale` and `--geometry-rotate` for vertex files.

The downstream boundary can be chosen at the same time with `--outlet-type` and
`--p-ref` — see [`PRESSURE_OUTLET`](#pressure_outlet--anchoring-the-pressure-instead-of-the-velocity)
above for what the two conditions differ in. For anything the flags cannot
express, build the `boundary_config` dict yourself as shown in step 3; the case
files under `cases/` are worked examples of exactly that.

#### Complete CLI reference

Everything below is settable from the command line — you should not need to edit
any file to launch a run. `python -m pycfd.main --help` prints the same list.

**Choosing what to run**

| flag | default | meaning |
|---|---|---|
| `--case {cavity,channel,cylinder,taylor_green}` | `cavity` | which benchmark to run |
| `--list-cases` | — | print the available cases and exit |
| `--convergence` | — | run the Taylor–Green grid-convergence study and exit |

**Grid and physics**

| flag | default | meaning |
|---|---|---|
| `--re RE` | case-specific | Reynolds number |
| `--nx N`, `--ny N` | case-specific | cells in x and y |
| `--dt DT` | case-specific | time step; **also the ceiling** under adaptive stepping |
| `--t-end T` | case-specific | stop time |
| `--max-steps N` | none | hard cap on iterations, whatever the clock says |
| `--cfl C` | case-specific | maximum Courant number |
| `--mode MODE` | `periodic` | channel only: `periodic` or `developing` |
| `--domain-length L` † | 16 (cylinder) | streamwise extent of the domain |
| `--domain-height H` † | 8 (cylinder) | cross-stream extent; with the body size this sets the blockage ratio |

**Outflow boundary** †

| flag | default | meaning |
|---|---|---|
| `--outlet-type {outlet,pressure_outlet}` | case's own choice | velocity outlet (pressure floats) or pressure outlet (pressure anchored) |
| `--p-ref P` | `0.0` | pressure held on a pressure outlet |

**Custom geometry** †

| flag | default | meaning |
|---|---|---|
| `--geometry FILE` | none | 2D body: vertex file (`.csv`/`.txt`/`.dat`) or bitmap (`.png`/`.jpg`) |
| `--geometry-scale S` | `1.0` | scale a vertex outline about its centroid |
| `--geometry-rotate DEG` | `0.0` | rotate a vertex outline, degrees anticlockwise |

**Numerics**

| flag | default | meaning |
|---|---|---|
| `--time-scheme {euler,rk2,rk3}` | `rk3` | time integrator; `euler`/`rk2` are unstable for central advection |
| `--advection {central,upwind}` | `central` | 2nd-order central, or blended upwind for stubborn high-Re runs |
| `--pressure-solver {direct,cg,jacobi,sor}` | `direct` | linear solver for the pressure equation |
| `--les` / `--no-les` | laminar | enable or force off the Smagorinsky sub-grid model |

**Visualisation**

| flag | default | meaning |
|---|---|---|
| `--live` | off | real-time window; `space` pauses, `q` closes |
| `--display {speed,pressure,vorticity,streamlines}` | `speed` | field shown live |
| `--plot-every N` | `50` | solver steps between rendered frames |
| `--quiver` | off | overlay velocity vectors on the live view |
| `--save-animation PATH` | none | render headlessly to a file (e.g. `wake.gif`) |
| `--frames N` | `200` | frames to render when saving an animation |
| `--no-plots` | off | skip the static figures a case would write |

**Input and output**

| flag | default | meaning |
|---|---|---|
| `--outdir DIR` | `results` | where figures and exports go |
| `--name LABEL` | `<case>_Re<re>` | label for logs, figure titles **and the files written** |
| `--export-vtk` | off | write the final field as legacy VTK (ParaView) |
| `--export-csv` | off | write the final field as CSV |
| `--checkpoint` | off | save a restartable `.npz` |
| `--resume PATH` | none | continue from a checkpoint (pass a larger `--t-end`) |
| `--progress` | off | `tqdm` progress bar |
| `-v`/`--verbose`, `-q`/`--quiet` | INFO | DEBUG logging / warnings only |

† **Case-specific.** `--domain-length`/`--domain-height`, the two outflow flags
and the three geometry flags only apply to cases that have the corresponding
feature — an external flow with an outlet and a configurable domain, which today
means `cylinder` (and the outflow flags also on `channel --mode developing`).
Using one where it cannot apply is an **error naming the cases that do support
it**, not a silent no-op.

Exit codes: `0` ran and validated, `1` ran but a validation check missed its
tolerance, `2` could not run (bad configuration, divergence, unsupported flag).

#### Worked recipes

Custom body from a vertex file, in a larger domain, with the wake anchored:

```bash
python -m pycfd.main --case cylinder --geometry shield.csv \
    --re 100 --nx 384 --ny 192 --domain-length 24 --domain-height 12 \
    --geometry-scale 1.0 --geometry-rotate 0 \
    --outlet-type pressure_outlet --p-ref 0 \
    --t-end 120 --cfl 0.4 --name shield_run1 \
    --export-vtk --checkpoint --progress
```

Watch it develop instead, then keep going from where it stopped:

```bash
python -m pycfd.main --case cylinder --geometry shield.csv --re 100 --live --display vorticity
python -m pycfd.main --resume results/shield_run1.npz --t-end 240
```

A quick low-resolution sanity pass before committing to a long run:

```bash
python -m pycfd.main --case cylinder --geometry shield.csv --re 100 \
    --nx 96 --ny 48 --t-end 5 --no-plots
```

The time step is chosen automatically from both stability limits, so you do not
normally set `dt`. The run aborts with a clear message — never silent garbage —
if the solution goes non-finite, exceeds `1e6`, or the step collapses below
`1e-8`.

Long runs are restartable:

```bash
python -m pycfd.main --case cylinder --re 100 --t-end 60 --checkpoint
python -m pycfd.main --resume results/cylinder_Re100.npz --t-end 200
```

A checkpoint stores the raw ghosted arrays plus the full configuration, so a
resumed run continues bit-for-bit.

### 5. Viewing and visualising results

> **Where:** `visualization/live_plot.py` and `visualization/static_plot.py`
> for figures; `analysis/postprocess.py` and `analysis/export.py` for numbers
> and files.

**Watch it live.** `space` pauses and resumes, `q` closes:

```bash
python -m pycfd.main --case cylinder --re 100 --live --display vorticity --quiver
```

`--display` takes `speed`, `pressure`, `vorticity` or `streamlines`, and
`--plot-every N` trades frame rate against solver throughput. Add
`--save-animation wake.gif --frames 200` to render headlessly to a file instead.

**Static figures** are written automatically by each case at 300 DPI, or on
demand:

```python
from pycfd.visualization import static_plot as sp

sp.use_headless_backend()                     # no display needed
sp.four_panel_figure(sim.fields, solid=sim.solid_mask,
                     title="Aerofoil, Re = 100", path="out/fields.png")
sp.centerline_comparison_figure(sim.fields, path="out/centerlines.png")
sp.time_series_figure(result.history, keys=("kinetic_energy", "max_div"),
                      path="out/history.png")
```

**Quantitative post-processing** — everything returns arrays, so it composes
with whatever else you use:

```python
from pycfd.analysis.postprocess import (
    vorticity, stream_function, kinetic_energy, enstrophy, sample_at)

w   = vorticity(sim.fields)             # (nx, ny) at cell centres
psi = stream_function(sim.fields)       # (nx+1, ny+1) at corners
ke  = kinetic_energy(sim.fields, mask=sim.solid_mask)
cd, cl = sim.force_coefficients()       # from the immersed-boundary reaction
print(sim.diagnostics())                # t, step, KE, max divergence, Cd, Cl
```

Point probes record a time series while the run proceeds, which is what you need
for a shedding frequency:

```python
from pycfd.analysis.postprocess import strouhal_number

probe = sim.add_probe(x=6.0, y=4.0, name="wake")
sim.run(callback_every=10)                     # probes record on each callback
series = probe.as_arrays()
print(strouhal_number(series["t"], series["v"], l_ref=1.0, u_ref=1.0))
```

`strouhal_number` returns `nan` rather than a wrong answer if the record is too
short to resolve a peak (under 16 samples), so if you get `nan`, lower
`callback_every` or run longer. It cannot tell you whether the signal is
*genuinely* periodic, though — check that the force history has settled into a
steady oscillation first, or you will read back the record-length frequency.

**Export** for external tools:

```bash
python -m pycfd.main --case cylinder --re 100 --export-vtk --export-csv --checkpoint
```

```python
sim.export_vtk("out/field.vtk")     # legacy RECTILINEAR_GRID, opens in ParaView
sim.export_csv("out/field.csv")     # x, y, u, v, speed, pressure, vorticity
sim.save_checkpoint("out/state.npz")
```

The VTK writer is hand-rolled, so no `vtk` package is needed.

### End-to-end example

All five steps, in one script:

```python
"""Flow past a custom 2D body, start to finish."""
import numpy as np
from pycfd.config import BCKind, BCSpec, SimulationConfig
from pycfd.core.mesh import StructuredMesh
from pycfd.geometry.obstacles import load_polygon, polygon_mask, transform_polygon
from pycfd.physics.incompressible import Simulation
from pycfd.visualization import static_plot as sp

sp.use_headless_backend()
LX, LY, NX, NY = 16.0, 8.0, 256, 128

# 1. model  ---------------------------------------------------------------
mesh = StructuredMesh(NX, NY, LX, LY)
outline = transform_polygon(load_polygon("aerofoil.csv"),
                            scale=2.0, center=(4.0, 4.0), rotate_deg=-10.0)
body = polygon_mask(mesh, outline, name="aerofoil")
print(body, "->", round(body.characteristic_length / mesh.dy, 1), "cells across")

# 2 + 3. mesh, boundaries and settings  ------------------------------------
cfg = SimulationConfig(
    nx=NX, ny=NY, lx=LX, ly=LY,
    re=100.0, u_ref=1.0, l_ref=body.characteristic_length,
    dt=0.02,                      # ceiling: without it the default 1e-3 caps every step
    t_end=80.0, cfl_max=0.4, name="aerofoil_Re100",
    boundary_config={
        "left":   BCSpec(BCKind.INLET, velocity=1.0),
        "right":  BCSpec(BCKind.OUTLET),
        "bottom": BCSpec(BCKind.SYMMETRY),
        "top":    BCSpec(BCKind.SYMMETRY),
    },
)

# 4. run  ------------------------------------------------------------------
sim = Simulation(cfg, obstacle=body, u_init=1.0)
probe = sim.add_probe(6.0, 4.0, "wake")
result = sim.run(progress=True, callback_every=10)   # probe samples every 10 steps
print(result.summary())

# 5. results  --------------------------------------------------------------
cd, cl = sim.force_coefficients()
s = probe.as_arrays()
print(f"Cd = {cd:.3f}   Cl = {cl:.3f}")
print(f"wake probe: {len(s['t'])} samples, v in "
      f"[{s['v'].min():.3f}, {s['v'].max():.3f}]")
print(f"max |div u| = {sim.solver.max_divergence(sim.fields):.2e}")

sp.four_panel_figure(sim.fields, solid=sim.solid_mask,
                     title="Aerofoil, Re = 100", path="out/aerofoil_fields.png")
sim.export_vtk("out/aerofoil.vtk")
sim.save_checkpoint("out/aerofoil.npz")
```

Note what this example does *not* do: it reports the probe's range rather than a
Strouhal number. `strouhal_number` is only meaningful once the wake has become
periodic, and this one has not — its Cd and Cl are still drifting at `t = 60`.
Feeding it a non-periodic record returns the record-length frequency, which
looks like an answer and is not one. The cylinder at Re = 100 is where shedding
is established and the measurement is validated (St = 0.187).

Run as written it reports `7.4 cells across` — **below the 16 recommended
above**, so its Cd and Cl are indicative rather than converged. That is the
honest state of a thin body on a uniform Cartesian grid: the aerofoil's
projected height is only 0.46 in a domain 8 units tall, so resolving it properly
means either scaling the body up, shrinking the domain, or paying for a much
finer grid (`ny = 288` would reach 16 cells here). The printed check exists
precisely so the trade-off is visible before you trust a number, and it is the
first thing to fix if forces matter.

---

## Numerical method

### Staggered (MAC) grid

Pressure sits at cell centres, `u` on x-faces, `v` on y-faces. The alternative —
a collocated grid — leaves odd and even pressure modes decoupled under a
second-order central pressure gradient, producing the classic chequerboard
oscillation and requiring Rhie–Chow interpolation to suppress it. On the MAC
grid the discrete divergence and gradient are exact negative adjoints, so the
projection removes the divergence **to machine precision**:

```
max |div u| after a step:  ~1e-14   (cavity, 128x128)
```

Every field carries one ghost layer per side, which makes boundary conditions
and periodic wrapping uniform across all three variables. The index convention
is documented in full in `core/mesh.py`.

### Time integration

Three explicit schemes are available; **SSP-RK3 is the default**, and that is a
correctness choice rather than an accuracy one:

| scheme | order | stable for central-difference advection? |
|---|---|---|
| `euler` | 1 | **no** — the stability region meets the imaginary axis only at the origin |
| `rk2` | 2 | no — `\|R(iy)\|² = 1 + y⁴/4 > 1`, though only weakly unstable |
| `rk3` | 3 | **yes** — `\|R(iy)\|² = 1 − y⁶/36 < 1` for `y < √3` |

Forward Euler is provided because the task specification names it as the
baseline, and it survives in practice only because the viscous term damps the
growth. At high Reynolds number that margin disappears. RK3 also makes the
temporal error negligible in a spatial convergence study.

The projection is applied at every Runge–Kutta stage. Because the stage
combinations are convex and the projection is affine, every intermediate state
is itself divergence-free.

### Adaptive time step

Two explicit limits act simultaneously and the smaller wins:

```
dt_conv = cfl / ( max|u|/dx + max|v|/dy )
dt_visc = 0.8 / ( 2 nu (1/dx² + 1/dy²) )
```

The viscous limit is not in the original specification but is not optional: on a
fine grid or at low Reynolds number it is by far the tighter of the two, and
omitting it makes the solver diverge in precisely the cases that should be
easiest. The convective form is the standard multi-dimensional generalisation of
`cfl·min(dx,dy)/max(|u|,|v|)` and is never less restrictive.

### Pressure Poisson

The operator is assembled once as a sparse matrix and factorised once with
`splu`; each step costs a pair of triangular solves. (`spsolve` would refactorise
on every call.) Three iterative alternatives are provided: preconditioned CG,
red/black SOR, and damped Jacobi.

The single most important property is that the Laplacian is assembled over
**exactly** the face set the projection corrects. A face is in the stencil if and
only if the projection may change it — so boundary faces and obstacle faces are
dropped from both. This is what makes the corrected velocity divergence-free to
solver tolerance rather than to truncation error.

With Neumann conditions everywhere the operator is singular by one constant
mode. The right-hand side is projected onto the compatible subspace (its fluid
mean is removed) and one reference cell is pinned. Pinning does not perturb the
answer: the rows sum to zero, so once the other `N−1` equations hold and the
right-hand side sums to zero, the pinned equation is satisfied automatically.

> Undamped Jacobi does **not** converge on this operator. For the Neumann
> Laplacian assembled by dropping boundary coefficients, the chequerboard mode
> has eigenvalue exactly −1 at interior, edge and corner cells alike, because the
> diagonal shrinks in step with the neighbour count. Damping by `ω` maps that to
> `1 − 2ω`; the default is the classical `2/3`.

### Spatial discretisation

Convection uses the conservative form `d(uu)/dx + d(uv)/dy`, with the mixed
product evaluated once per cell **corner** and shared by both momentum equations
— the Harlow–Welch arrangement that makes the scheme discretely momentum
conserving. An optional donor-cell blend adds controllable upwinding for high
Reynolds numbers.

Diffusion is the 5-point Laplacian. With the Smagorinsky model active the
viscous term switches to the full stress-divergence form, the only form correct
for a spatially varying viscosity; on a solenoidal field with constant viscosity
the two agree to 4e-16.

---

## Validation

### Grid convergence — Taylor–Green vortex

The Taylor–Green vortex is an *exact unsteady* solution of the full nonlinear
equations on a periodic domain. Periodicity matters: a walled domain gives the
fractional-step method a pressure boundary layer that contaminates any measured
order of accuracy. The Courant number is held fixed, so with third-order time
integration the temporal error is `O(h³)` and the second-order spatial error
dominates cleanly.

| N | L2 error | order | L∞ error | order |
|---:|---:|---:|---:|---:|
| 16 | 3.041e-04 | — | 5.966e-04 | — |
| 32 | 7.631e-05 | 1.995 | 1.519e-04 | 1.974 |
| 64 | 1.909e-05 | 1.999 | 3.814e-05 | 1.993 |
| 128 | 4.775e-06 | **2.000** | 9.559e-06 | **1.997** |

```bash
python -m pycfd.main --convergence
```

### Lid-driven cavity vs Ghia et al. (1982)

Centreline profiles at 128×128, run to steady state.

| Re | u profile L2 | u profile L∞ | v profile L2 | v profile L∞ | max \|div u\| |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.0067 | 0.0257 | 0.0053 | 0.0091 | 9.1e-14 |
| 400 | 0.0098 | 0.0395 | 0.0057 | 0.0166 | 3.6e-14 |
| 1000 | 0.0149 | 0.0588 | 0.0104 | 0.0293 | 5.1e-14 |

All three pass a 0.02 L2 tolerance. Error grows with Reynolds number as expected
— the wall layers thin and a uniform 128×128 grid resolves them less well.

> **One withheld reference value.** `GHIA_V[400]` at `x = 0.9063` is stored as
> `nan` and skipped by the error norms. The value originally transcribed there
> (−0.23827) could not be confirmed and is inconsistent with its own
> neighbours: every *other* point of that profile is reproduced to better than
> 0.006 and improves under refinement, while this one is off by 0.150 and does
> **not** move between 96×96 and 160×160 — the signature of a bad reference
> value, not of discretisation error. The published neighbours are −0.44993 at
> `x = 0.8594` and −0.22847 at `x = 0.9453`, so the true entry lies between
> them, but it is deliberately not reconstructed here; a fabricated number in a
> reference table is worse than a missing one. See `GHIA_KNOWN_GAPS` in
> `analysis/validation.py`. If you have the paper, restoring that Table II entry
> closes the gap.

### Poiseuille channel

| mode | centreline error | profile L2 (relative) |
|---|---:|---:|
| periodic + body force | 1.0e-07 % | 2.3e-04 |
| developing (inlet/outlet) | 0.10 % | 1.6e-03 |

The two configurations fix the flow rate differently, and comparing against the
wrong one manufactures a 50% "error" that is not there. The body force sets the
amplitude directly (`u_max = f h²/8ν`); a *uniform inlet* instead fixes the mass
flux, and a parabola with mean `U` peaks at `1.5U`. The developing case
reproduces that 1.5 factor to 0.1%, with the discrete mean velocity equal to the
inlet speed to machine precision.

### Flow past a cylinder

Drag is obtained from the immersed-boundary reaction force — the momentum the
direct forcing must remove to hold the body at rest. Both maskings in a substep
are counted: the one after the predictor carries the **advective and viscous**
flux, the one after the projection reduces to the discrete **pressure** surface
integral. Counting only the latter drops roughly half the drag at Re = 20.

Results on the default 16 D × 8 D domain at 256×128 (16 cells per diameter):

| Re | Cd | literature (unbounded) | St | literature | Cl peak-to-peak |
|---:|---:|---:|---:|---:|---:|
| 20 | 2.537 | 2.0–2.1 | — (steady) | — | 1.1e-14 |
| 100 | 1.460 | 1.32–1.40 | 0.1875 | 0.160–0.172 | 0.582 |

At Re = 20 the wake is steady and symmetric — the RMS lift is 5e-15, i.e. exactly
zero to round-off. At Re = 100 the symmetry breaks and a von Kármán street
develops, giving a clean periodic lift signal.

Both Cd and St come out roughly 10% above the unbounded values, and the reason is
measurable rather than mysterious. Refining the domain height at Re = 20 isolates
the confinement contribution:

| domain height | blockage `D/H` | Cd |
|---:|---:|---:|
| 8 D | 0.125 | 2.537 |
| 16 D | 0.0625 | 2.390 |
| 24 D | 0.0417 | 2.379 |

Cd converges to ≈ 2.38 as the blockage vanishes, so confinement explains part of
the gap and the mask explains the rest: zeroing every face that touches a solid
cell places the no-slip surface roughly half a cell outside the true circle, so
the effective diameter and wetted perimeter both exceed the nominal ones.
Confinement raises the shedding frequency the same way.

**Treat these drag coefficients as engineering estimates.** A mask-based
immersed boundary on a Cartesian grid is first-order accurate at the surface;
that is the price of a non-conforming mesh, and no amount of care in the force
integration removes it. The case's `domain_height` argument is exposed precisely
so this bias can be measured rather than assumed away.

---

## Performance

Measured on an 8-thread machine. The stencil kernel is JIT-compiled in two
builds — serial and thread-parallel — and selected by grid size, because
threading only pays for itself above roughly 12k cells.

| grid | NumPy stencil | Numba stencil | speed-up |
|---|---:|---:|---:|
| 64×64 | 0.123 ms | 0.055 ms | 2.25× |
| 128×128 | 0.415 ms | 0.223 ms | 1.86× |
| 192×192 | 1.112 ms | 0.505 ms | 2.20× |
| 256×256 | 2.110 ms | 0.359 ms | **5.87×** |

Full solver, wall-clock seconds per 1000 time steps:

| grid | NumPy | Numba | speed-up |
|---|---:|---:|---:|
| 128×128 | 7.46 s | 6.99 s | 1.07× |
| 256×256 | 33.6 s | 27.6 s | 1.22× |

**The end-to-end gain is small, and that is the honest headline.** Profiling
shows the sparse LU solve dominates a time step; the stencils are not the
bottleneck, so even a large speed-up there moves the total little. The task
specification's target of ≥3× applies to the stencil operations and is met at
256×256 but not at 128×128, where thread-overhead limits the gain to ~1.9×.

The Numba path is *bit-identical* to the NumPy path — same operations in the
same order, verified in the test suite for both blending modes and both boundary
families, and the serial and parallel builds agree bitwise with each other.

> If you modify `kernels.py`: the parallel build must not set `cache=True`.
> Numba keys its on-disk cache on the source function, so two builds of the same
> function collide and the second silently loads the first one's artifact —
> yielding a "parallel" kernel that runs serially. That cost a real 2.5× before
> it was caught.

---

## Testing

```bash
python -m pytest pycfd/tests -q
```

154 tests covering mesh geometry, obstacle construction from every supported
source, all six boundary condition types, Poisson assembly and all four linear
solvers, discrete conservation, the analytical benchmarks, and the export
round-trips.

Notable invariants under test:

- projected velocity is divergence-free to `1e-11` for six boundary
  configurations, with and without an obstacle;
- every boundary condition imposes **exactly** the prescribed value, checked on
  the specific staggered quantity it controls;
- uniform periodic flow and a quiescent box are preserved to machine precision;
- Taylor–Green kinetic energy decays at the analytical `exp(−4νt)` rate;
- the Poisson operator is exactly symmetric with vanishing row sums, and the
  obstacle mask decouples solid cells;
- the analytical reference solutions are verified against the governing
  equations themselves, not taken on trust;
- obstacle areas and centroids match closed-form values for circles, polygons,
  concave outlines and bitmaps, and a custom body keeps the solver
  divergence-free exactly as a built-in one does.

---

## Architecture

```
pycfd/
├── core/
│   ├── mesh.py          Structured mesh; the MAC index convention lives here
│   ├── fields.py        (u, v, p) container with ghost layers
│   ├── boundary.py      BC classes, periodic wrapping, global mass balance
│   ├── pressure.py      Poisson assembly + direct/CG/SOR/Jacobi solvers
│   ├── solver.py        Projection method: advection, diffusion, projection
│   ├── kernels.py       Optional fused Numba stencils
│   └── timestepper.py   Adaptive dt, run loop, divergence detection
├── physics/
│   ├── incompressible.py  High-level Simulation driver
│   └── turbulence.py      Smagorinsky SGS model
├── geometry/obstacles.py  Volume-fraction masks: primitives, polygons,
│                          bitmaps, predicates
├── analysis/
│   ├── postprocess.py   Vorticity, stream function, forces, probes
│   ├── validation.py    Analytical solutions, Ghia data, error norms
│   └── export.py        VTK / CSV / NPZ checkpoints
├── visualization/
│   ├── static_plot.py   Publication figures at 300 DPI
│   └── live_plot.py     FuncAnimation viewer
├── cases/               cavity, channel, cylinder, taylor_green
├── tests/               mesh, geometry, boundary, pressure, solver, validation
├── config.py            Dataclass configuration; every constant lives here
└── main.py              CLI
```

Two structural notes against the original specification: `core/fields.py` was
added so that `boundary.py` can operate on a field bundle without importing the
solver (which would be circular), and `core/kernels.py` holds the optional JIT
stencils. `visualization/` and `analysis/` never import from `core/`, and
`core/` never imports Matplotlib.

---

## Known limitations

- **`solver_type="simple"` is not implemented.** The configuration enum accepts
  it because the specification lists it, but constructing a solver with it
  raises `NotImplementedError`. Every build phase and success criterion in the
  specification targets the projection method; SIMPLE was left as a clearly
  flagged gap rather than a half-built alternative.
- **Stretched meshes are generated but not solved on.** `StructuredMesh`
  supports geometric stretching and is tested for it, but the finite-difference
  operators are derived for uniform spacing, so the solver raises
  `NonUniformMeshError` rather than quietly producing a first-order answer.
- **The stream function is integrated, not solved for.** The specification asks
  for `∇²ψ = −ω`; since the discrete divergence vanishes to machine precision,
  direct integration of `u = ∂ψ/∂y` is path-independent, exact, cheaper, and
  reproduces prescribed boundary values exactly. It is strictly the better
  choice here.
- **Immersed-boundary drag is first-order accurate** — see the cylinder section.
- **No 3D or CAD import.** Geometry is a 2D outline, bitmap or predicate;
  there is no STL, STEP or DXF reader, and none is planned — a 2D silhouette is
  the entire model this solver can use.
- **Immersed geometry is resolved by a staircase.** A body must span at least
  ~16 cells for forces to be meaningful; thin bodies in large domains are
  expensive on a uniform grid, which the usage guide's example illustrates
  rather than hides.
- **2D only**, uniform Cartesian grids, explicit time integration.

## References

- Harlow & Welch (1965), *Numerical calculation of time-dependent viscous
  incompressible flow of fluid with free surface*, Phys. Fluids 8, 2182.
- Chorin (1968), *Numerical solution of the Navier–Stokes equations*,
  Math. Comp. 22, 745.
- Ghia, Ghia & Shin (1982), *High-Re solutions for incompressible flow using the
  Navier–Stokes equations and a multigrid method*, J. Comput. Phys. 48, 387.
- Griebel, Dornseifer & Neunhoeffer (1998), *Numerical Simulation in Fluid
  Dynamics: A Practical Introduction*, SIAM.
