# Usage guide

The full workflow: bringing in a shape, meshing, boundary conditions and
settings, running, and reading the results — plus the complete CLI reference.

---

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
> it explicitly if you want coefficients based on chord instead — or pass
> [`--l-ref`](#reference-length-and-flight-conditions) on the command line,
> which does the same thing without touching the geometry file.

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

#### Stretched meshes

A stretched axis puts cells where the flow needs them. Two settings describe
one: `stretch_x` / `stretch_y`, the growth factor between neighbouring cells,
and `cluster_x` / `cluster_y`, which end the growth starts from.

| cluster | smallest cells | use it for |
|---|---|---|
| `low` (default) | at the low-coordinate end | a single wall, or a shear layer pinned to one side |
| `walls` | at **both** ends | a channel, or anything bounded by two walls |
| `centre` | in the **middle** | a body held in the interior of a large domain |

**The mode matters more than the ratio.** Growth has to begin somewhere, and
`low` refines one end while coarsening everything else. On Poiseuille flow —
which is symmetric — `low` at ratio 1.05 is *6× worse* than a uniform mesh of
the same cell count, because the wall it starves costs more than the wall it
refines wins. `walls` at the identical ratio beats uniform. Choose the mode from
where the flow has its gradients.

```bash
python3 -m pycfd.main --case cylinder --nx 128 --ny 64 --stretch-y 1.06 --cluster-y centre
```

That run resolves the cylinder with ~18 cells across it; a uniform mesh needs
`--ny 96` to reach 12. For an interior layer of half-width 0.03, 1% error takes
48 uniform cells per axis, 24 at 4× total growth and 16 at 10× — 4× and 9× fewer
cells in 2D, second order throughout.

Two things to know before turning it on:

- **The time step follows the smallest cell.** Clustering does not make the step
  smaller than the resolution you asked for — a stretched wall and a uniform
  wall of the same thinness take the same step — but the viscous limit scales as
  `h²`, so a mesh whose finest cell is 100× smaller than its coarsest takes
  steps set by that finest cell, not by the average.
- **A stretched axis cannot be periodic**, and the fused Numba kernel is
  uniform-only, so a stretched run uses the (slower, equally accurate) NumPy
  operators.

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
| `pressure_solver` | `direct` | **linear solver** for the pressure equation: `direct` (factorised once), `mgcg` (multigrid-preconditioned CG — see [Choosing a pressure solver](#choosing-a-pressure-solver)), or `cg` / `sor` / `jacobi` / `multigrid` |
| `mg_sweeps` | 1 | smoothing sweeps on each side of a multigrid V-cycle. Only read by `mgcg` and `multigrid` |
| `body_force` | (0, 0) | **force per unit mass** `(fx, fy)` applied everywhere — this is what drives the periodic channel |
| `use_les` | False | enable the **Smagorinsky sub-grid model** for under-resolved turbulence |
| `stretch_x`, `stretch_y` | 1.0, 1.0 | **geometric cell-growth ratios** between neighbouring cells; `1.0` is uniform. See [Stretched meshes](#stretched-meshes) |
| `cluster_x`, `cluster_y` | `"low"` | where the stretching puts its **smallest** cells: `low`, `walls` or `centre`. Ignored when the ratio is `1.0` |
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

### Choosing a pressure solver

Nothing else in a step costs what the pressure solve costs. Under the default
RK3 the projection runs three times per step, and on the cylinder case that is
55% of the wall time at 128×64 and 82% at 512×256. So this is the one numerical
choice worth making deliberately.

`direct` is the default and it is hard to beat. It factorises the operator once
and every later solve is a pair of triangular sweeps. What it cannot do is
scale, because the factors fill in. Measured on the cylinder case, RK3, this
machine:

| grid | cells | `direct` | `mgcg` | `direct` factors | `mgcg` hierarchy |
|---|---|---|---|---|---|
| 256×128 | 33k | **8.7** ms/step | 49.7 ms/step | 31 MiB | 7 MiB |
| 512×256 | 131k | **38** ms/step | 206 ms/step | 164 MiB | 28 MiB |
| 1024×512 | 524k | **188** ms/step | 829 ms/step | 840 MiB | 113 MiB |
| 1536×768 | 1.18M | **1130** ms/step | 1991 ms/step | 2106 MiB | 259 MiB |

Read the two trends rather than any one row. `direct` is 5.7× faster at 256×128
and only 1.8× faster at 1536×768, while its factors go from 4.7× the
hierarchy's size to 8.1×. The factorisation itself follows the same curve —
0.8 s against 0.06 s at the smallest grid, 12.2 s against 2.7 s at the largest.
Both solvers agree to within `3e-12` in velocity after thirteen steps at every
size.

One grid further makes the point. At 2048×1024 — 2.1M cells — `mgcg` builds a
nine-level hierarchy in 6.0 s, holds it in 464 MiB, and steps in 3.7 s at 11
iterations per solve. Extrapolating the measured fill above (1.28, 1.64 and
1.83 KiB per unknown) puts the direct factorisation for the same grid near
4 GiB before SuperLU's working space, which does not fit alongside the fields
on an 8 GiB machine. That row is not in the table because it was not run.

`mgcg` is conjugate gradient preconditioned by an aggregation-multigrid V-cycle.
Its cost is `O(N)` in both time and storage, and its iteration count does not
depend on the grid. Measured against the Jacobi-preconditioned `cg`, to the
same `1e-10` residual:

| grid | `cg` (Jacobi) | `mgcg` |
|---|---|---|
| 64×64 | 283 it | 13 it |
| 128×128 | 552 it | 13 it |
| 256×256 | 1128 it | 13 it |
| 512×512 | 2219 it | 13 it |

The left column doubles every time the grid doubles; the right one does not
move. That is what makes `mgcg` a different kind of solver rather than a
faster one.

**The honest summary: `direct` wins on time at every size where it fits, and
`mgcg` is what you use when it stops fitting.** Sparse LU is very strong on 2D
problems; a triangular solve makes one pass over its factors, while a V-cycle
makes a dozen passes over the operator. Multigrid does not close that gap in 2D
— it removes the memory wall instead, and the gap narrows as the wall gets
closer.

Two things to know before switching:

- **An iterative solver is divergence-free to its tolerance, not to machine
  precision**, and the default tolerance is looser than the `1e-11` the rest of
  the codebase assumes. This is not specific to multigrid — `cg`, `sor` and
  `jacobi` behave the same way — but it is the one thing that can surprise you
  when switching. Tightening `poisson_tol` buys it back cheaply; at 512×256:

  | solver | `poisson_tol` | ms/step | iterations | `max\|div\|` |
  |---|---|---|---|---|
  | `direct` | — | 38 | — | `2.8e-14` |
  | `mgcg` | `1e-10` (default) | 201 | 11.0 | `6.6e-11` |
  | `mgcg` | `1e-12` | 241 | 13.3 | `1.1e-12` |
  | `mgcg` | `1e-14` | 287 | 16.0 | `3.9e-14` |

  Machine-precision divergence costs 43% more time, not a different method.
- **`mg_sweeps` is a real knob and 1 and 2 are close.** More smoothing per cycle
  buys fewer cycles: at 512×256, `1` needs 11 iterations and `2` needs 8, and
  the two come out within a per cent of each other on wall time. `3` is slower
  than both. Leave it at 1 unless you measure otherwise on your own case.

`multigrid` runs the V-cycles on their own, without CG. It converges at a factor
of about 0.42 per cycle independent of the grid, needs roughly twice the cycles
`mgcg` needs iterations, and exists mainly because it makes the hierarchy's
behaviour visible. `mgcg` is the one to use.

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

### Several bodies at once

Repeat `--geometry` to put more than one body in the same domain — formation
aerodynamics, a store beside a pylon, a wake-interference study. Each body then
needs its own `--geometry-at X,Y`:

```bash
python -m pycfd.main --case cylinder --re 40 \
    --geometry pycfd/geometry_import/shield.csv --geometry-at 4,4 \
    --geometry pycfd/geometry_import/shield.csv --geometry-at 9,4 \
    --nx 128 --ny 64 --t-end 25
```

There is no default arrangement, because there is no defensible one: a guessed
formation does not fail, it quietly simulates something nobody asked for. So
placement is required as soon as there is a second body.

**Each body reports its own force.** A total would average away the only thing
a multi-body run is for. The report adds one entry per body, and the case's own
checks list them side by side:

```
  cd_shield#1                  2.07333
  cd_shield#2                  0.448144
    [PASS] force resolved per body: shield#1 Cd = 2.073, shield#2 Cd = 0.448
           (all against L = 1.95)
```

Two identical bodies loaded from one file would collide in that listing, so
repeated names are numbered `#1`, `#2` — otherwise one body's load would
silently overwrite another's.

Every coefficient is normalised by the *same* `l_ref`, which is what makes them
comparable; dividing each body by its own length would produce numbers that
cannot be set beside one another. With no `--l-ref`, that shared length is the
largest body's.

**Bodies may not touch.** Two that overlap, or that merely share a cell face,
are refused rather than run:

```
bodies 'fore' and 'aft' touch along 8 faces; there is no fluid between them to
carry a force, so a per-body coefficient would be meaningless -- merge them
into one outline, or leave at least one fluid cell between them
```

The force on a body is collected from the faces touching it, so each face has
to belong to exactly one body for the split to add up. Bodies meeting only at a
corner are fine — a corner is not a face. Two that genuinely are in contact are
one body as far as the flow is concerned: give them a single outline.

**Blockage counts all of them.** It is measured from the mask — the most
obstructed column of cells — rather than from any one body's nominal length.
Two bodies abreast block the sum of their spans; two in tandem block only one.
For a circular cylinder this is exactly the diameter, so the benchmark number
is unchanged.

The downstream boundary can be chosen at the same time with `--outlet-type` and
`--p-ref` — see [`PRESSURE_OUTLET`](#pressure_outlet--anchoring-the-pressure-instead-of-the-velocity)
above for what the two conditions differ in. For anything the flags cannot
express, build the `boundary_config` dict yourself as shown in step 3; the case
files under `cases/` are worked examples of exactly that.

#### Reference length and flight conditions

Two questions come up the moment a real body goes into the flow: *which* length
the Reynolds number is about, and what Reynolds number a real speed even
corresponds to. Three flags answer them.

**`--l-ref L` — which length the numbers are about.** A reference length is a
convention, not a measurement. A cylinder uses its diameter, an aerofoil its
chord, an aircraft its overall length — and the loader cannot tell which you
mean, so it defaults to the body's extent *across* the flow, the
cylinder-diameter convention. `--l-ref` names a different one, in the geometry
file's own units:

```bash
python -m pycfd.main --case cylinder --geometry pycfd/geometry_import/f22_side_profile.csv --l-ref 18.8 --re 1e6
```

It changes what `Re`, `Cd`, `Cl` and `St` are formed with — and therefore
`nu = u_ref * l_ref / re`, since the Reynolds number and the coefficients must
refer to the same length. It does **not** change the body. `blockage_ratio` and
`cells_across_body` keep reporting the body's real span, because those are
questions about the grid and the domain rather than about a convention. Both
lengths appear in the report, as `reference_length` and `characteristic_length`.

**`--wind-speed V` and `--altitude Z` — a Reynolds number from real
conditions.** `Re = V·L/nu` with `nu` from the International Standard
Atmosphere, treating the geometry's length unit as the metre:

```bash
python -m pycfd.main --case cylinder --geometry pycfd/geometry_import/f22_side_profile.csv \
    --l-ref 18.8 --wind-speed 70 --altitude 3000
```

`--re` and `--wind-speed` are mutually exclusive — two answers to one question
is a mistake, not a preference — and `--altitude` on its own is refused, since
it only selects the air properties the wind speed is converted with. Above
Mach 0.3 the run logs a warning: an incompressible solver has no density
equation at all, so past that point it is approximating a *different* flow, not
the same one slightly less well.

The solver stays non-dimensional throughout. `u_ref` remains 1.0, the free
stream still enters at 1.0, and the real speed lives in the Reynolds number.
The report carries the dimensional facts alongside — `wind_speed_m_s`,
`altitude_m`, `kinematic_viscosity`, `mach`, `dynamic_pressure_pa` — so the run
records what it was a simulation *of*.

**Writing results in SI.** `--rescale-to V` converts the field exports on the
way out, so nothing has to be converted by hand afterwards:

```bash
python -m pycfd.main --case cylinder --geometry pycfd/geometry_import/f22_side_profile.csv \
    --l-ref 18.8 --wind-speed 70 --altitude 3000 \
    --export-csv --export-vtk --checkpoint --rescale-to 70 --name f22
```

| output | units | why |
|---|---|---|
| `f22_SI.csv`, `f22_SI.vtk` | m, m/s, Pa, 1/s, s | what the flags asked for |
| `f22.npz` | solver units | a checkpoint restarts the solver, and the solver restarts in its own units |

Three things keep the two apart. The rescaled files carry an `_SI` suffix; the
CSV header names every unit (`pressure_Pa`, `u_m_s`) because a spreadsheet is
opened without the sidecar in view; and each `.provenance.json` records a
`units` block — including the exchange rate, so the conversion can be undone —
whether or not anything was rescaled.

`--rescale-to` takes the speed rather than reading it from `--wind-speed`,
because the two answer different questions: one sets the Reynolds number the
solver runs at, the other says what a solver velocity of 1 means on the way
out. They are usually the same number, and passing both says so explicitly.
Air density comes from `--altitude`.

**Reading results back in SI.** `pycfd/units.py` is the bridge:

```python
from pycfd.units import Scaling

s = Scaling.at_altitude(70.0, length=1.0, altitude=3000.0)
print(s.summary())
# 1 solver velocity = 70 m/s   1 solver length = 1 m   1 solver time = 0.01429 s
# rho = 0.9091 kg/m^3   q_inf = 2227 Pa   M = 0.213

s.to_speed(sim.fields.speed().max())   # solver velocity -> m/s
s.to_pascals(sim.fields.p_phys.mean()) # solver pressure -> Pa
s.to_seconds(sim.fields.t)             # solver time     -> s
```

It also exposes `atmosphere(z)` for the ISA properties on their own and
`reynolds_number(V, L, nu)` for the arithmetic, so a Reynolds number can be
worked out before there is a configuration to put it in.

> **Do not raise `u_ref` to a flight speed.** `u_ref` is the velocity scale the
> fields are *already* expressed in, and force coefficients are divided by its
> square. Setting it to 70 while the inlet still drives the flow at 1.0 does not
> rescale anything — it divides every coefficient by 4900 and reports a drag
> coefficient near zero. `SimulationConfig.validate()` refuses that combination
> outright. Put the speed in the Reynolds number and convert afterwards.

#### Complete CLI reference

Everything below is settable from the command line — you should not need to edit
any file to launch a run. `python -m pycfd.main --help` prints the same list.

**Choosing what to run**

| flag | default | meaning |
|---|---|---|
| `--case {cavity,channel,cylinder,taylor_green}` | `cavity` | which benchmark to run |
| `--list-cases` | — | print the available cases and exit |
| `--convergence` | — | run a grid-refinement study on `--case` and exit (defaults to `taylor_green`) |
| `--diagnose` | — | run the end-to-end sanity pass on `--case` and exit (defaults to `cylinder`); mutually exclusive with `--convergence` |
| `--resolutions N,N,N` | `16,32,64,128` (taylor_green), `32,64,128` (other studies), half and three-quarters of the case's own grid (`--diagnose`) | cell counts for the study, coarsest first |

**Grid and physics**

| flag | default | meaning |
|---|---|---|
| `--re RE` | case-specific | Reynolds number |
| `--nx N`, `--ny N` | case-specific | cells in x and y |
| `--stretch-x R`, `--stretch-y R` | 1.0 | geometric cell-growth ratio between neighbouring cells; `1.0` is uniform |
| `--cluster-x M`, `--cluster-y M` | `low` | where the stretching puts its smallest cells: `low`, `walls`, `centre`. See [Stretched meshes](#stretched-meshes) |
| `--dt DT` | case-specific | time step; **also the ceiling** under adaptive stepping |
| `--t-end T` | case-specific | stop time |
| `--max-steps N` | none | hard cap on iterations, whatever the clock says |
| `--cfl C` | case-specific | maximum Courant number |
| `--mode MODE` | `periodic` | channel only: `periodic` or `developing` |
| `--domain-length L` † | 16 (cylinder) | streamwise extent of the domain |
| `--domain-height H` † | 8 (cylinder) | cross-stream extent; with the body size this sets the blockage ratio |

**Flow conditions** †

| flag | default | meaning |
|---|---|---|
| `--l-ref L` | the body's span across the flow | reference length for `Re`, `Cd`, `Cl` and `St`, in the geometry's units |
| `--wind-speed V` | none | free-stream speed in m/s; derives `Re` from it and the ISA viscosity. Mutually exclusive with `--re` |
| `--altitude Z` | `0` (sea level) | altitude in metres, selecting the ISA air properties `--wind-speed` uses |
| `--transient F` | `0.5` | fraction of the record discarded before the force coefficients are averaged |

**Outflow boundary** †

| flag | default | meaning |
|---|---|---|
| `--outlet-type {outlet,pressure_outlet}` | case's own choice | velocity outlet (pressure floats) or pressure outlet (pressure anchored) |
| `--p-ref P` | `0.0` | pressure held on a pressure outlet |

**Custom geometry** †

| flag | default | meaning |
|---|---|---|
| `--geometry FILE` | none | 2D body: vertex file (`.csv`/`.txt`/`.dat`) or bitmap (`.png`/`.jpg`). **Repeatable** — see [several bodies at once](#several-bodies-at-once) |
| `--geometry-at X,Y` | ¼ along | centre the outline here; required once per body when several are given |
| `--geometry-scale S` | `1.0` | scale a vertex outline about its centroid; once for all bodies or once each |
| `--geometry-rotate DEG` | `0.0` | rotate a vertex outline, degrees anticlockwise; once for all bodies or once each |

**Numerics**

| flag | default | meaning |
|---|---|---|
| `--time-scheme {euler,rk2,rk3}` | `rk3` | time integrator; `euler`/`rk2` are unstable for central advection |
| `--advection {central,upwind}` | `central` | 2nd-order central, or blended upwind for stubborn high-Re runs |
| `--pressure-solver {direct,cg,jacobi,sor,multigrid,mgcg}` | `direct` | linear solver for the pressure equation; `mgcg` trades speed for O(N) memory on large grids |
| `--mg-sweeps N` | 1 | smoothing sweeps per side of a multigrid V-cycle |
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
| `--rescale-to V` | off | write those two in SI, treating solver velocity 1 as `V` m/s; adds an `_SI` suffix. Checkpoints stay in solver units |
| `--checkpoint` | off | save a restartable `.npz` |
| `--resume PATH` | none | continue from a checkpoint (pass a larger `--t-end`) |
| `--progress` | off | `tqdm` progress bar |
| `-v`/`--verbose`, `-q`/`--quiet` | INFO | DEBUG logging / warnings only |

† **Case-specific.** `--domain-length`/`--domain-height`, the three flow-condition
flags, the two outflow flags and the three geometry flags only apply to cases
that have the corresponding feature — an external flow with a body, an outlet
and a configurable domain, which today means `cylinder` (and the outflow flags
also on `channel --mode developing`). Using one where it cannot apply is an
**error naming the cases that do support it**, not a silent no-op.

Exit codes: `0` ran and validated, `1` ran but a validation check missed its
tolerance, `2` could not run (bad configuration, divergence, unsupported flag).

#### Averaged forces, and what the average is worth

A drag coefficient measured on a shedding wake is not a number, it is a
distribution sampled over time. The cylinder case discards a start-up window
and averages the rest; `--transient F` sets how much is discarded (0.5 by
default), and the report says what the remaining window was worth:

```
  cd_mean                      1.4081
  cd_uncertainty               0.000731784
  averaging_samples            251
  effective_samples            58.085
  autocorrelation_time         4.32125
  cd_drift_z                   0.562948
  validation:
    [PASS] Cd at Re=100: Cd = 1.408 ± 0.000732; unbounded literature range 1.32-1.4
    [PASS] force average is stationary: the two halves of the averaging window differ by 0.6 standard errors
```

**Why the error bar is not `s/√N`.** Consecutive samples of a shedding wake are
close to the same measurement — the force history is written every ten steps,
so a shedding cycle spans tens of writes. Using `√N` there would report an
uncertainty that shrinks with the *sampling rate*: write twice as often and the
error bar halves, which is plainly false. The band is built instead from the
integrated autocorrelation time `τ`, so `effective_samples = N/τ` counts
independent observations rather than writes — 251 samples above are worth 58.
Sampling more often raises `averaging_samples` and `τ` together and leaves the
band where it was, as it must.

**Was the transient long enough?** The retained window is split in half and the
two means compared against their own combined uncertainty. A record still
sliding down from start-up has a first half measurably above its second, and
`cd_drift_z` reports that in standard errors. Beyond ±2 the run fails the
stationarity check and tells you to raise `--transient` or run longer.

Two refinements keep the check honest in opposite directions. A drift smaller
than one part in a thousand of the mean is *not* reported as a failure however
many standard errors wide it is — a well-converged steady wake has almost no
scatter left, so parts-per-million of residual drift would otherwise condemn
exactly the runs that went best. And a record whose transient still dominates
it defeats the z-score entirely, because the trend is read as correlation and
inflates the very band it is being measured against. What gives that away is
`autocorrelation_time`: a settled record reports `τ` near 1, one still carrying
its transient reports hundreds, with a handful of effective samples. Read those
two together before trusting a mean.

#### Is it actually shedding?

`strouhal_number()` answers "where is the largest peak in the spectrum?" — and
it answers it for anything. Hand it white noise and it reports `St = 9.27`;
hand it a linear drift and it reports `0.0100`. Every signal has a strongest
frequency, so a peak-picker never says no.

The cylinder case now decides properly, and reports why:

```
  strouhal                     0.183294
  shedding_detected            1
  shedding_periodicity         0.991818
  shedding_concentration       0.970982
  shedding_periods             10.9635
    [PASS] vortex shedding developed: St = 0.1831 over 11.0 periods,
           periodicity 0.99, lift peak-to-peak 0.420; literature 0.16-0.172
```

Three things all have to hold, because each is easy to fool alone:

| measure | asks | fails on |
|---|---|---|
| `shedding_periodicity` | does the signal *repeat*? Autocorrelation at a lag of exactly one period — 1.0 for a clean tone, ~0 for noise | noise, drift, decay |
| `shedding_concentration` | is the power in a narrow band, or spread across the spectrum? | broadband wobble |
| `shedding_periods` | has enough been seen for a frequency to be measured at all? | a run stopped too early |

The periodicity test is the one a peak-picker lacks, and it is deliberately
independent of the spectrum that proposed the period in the first place.

When detection fails the report says which measure failed and what to do:

```
    [FAIL] vortex shedding developed: only 22 samples after discarding the
           transient, below the 32 a spectrum needs
```

That is the failure an F-22 run hit: a wake that had not begun shedding still
produced a confident Strouhal number, and nothing in the output said so. The
detector is available directly for any signal:

```python
from pycfd.analysis.shedding import detect_shedding

signal = detect_shedding(probe.t, probe.v, l_ref=1.0, u_ref=1.0)
print(signal.report())
if signal.detected:
    print(signal.strouhal, signal.periods_observed)
```

#### Grid-refinement studies

`--convergence` refines whichever case `--case` names and reports what moved:

```bash
python -m pycfd.main --convergence                                    # Taylor–Green
python -m pycfd.main --convergence --case cylinder --re 100 --resolutions 64,128,256
```

The two cases are not doing the same thing, and the report says which:

| case | what it can measure | what you get |
|---|---|---|
| `taylor_green` | a true **error** — the exact unsteady solution is known | error at each grid, and an observed order of accuracy |
| `cavity`, `channel` | a true **error** — against Ghia et al. and the analytical parabola | the same, per declared metric |
| `cylinder` | nothing exact — an immersed boundary on a Cartesian grid has no reference | the values, and whether they stopped moving |

A metric declared as a **quantity** rather than an error gets no observed
order. That is deliberate: fitting one to a sequence with nothing to be wrong
against is exactly how a study that never reached its asymptotic regime gets
extrapolated as though it had. What it gets instead is the relative change
between successive grids.

The verdict then depends on which of the two the metric is, because refinement
is supposed to do **opposite** things to them:

| kind | converging looks like | the verdict line |
|---|---|---|
| quantity | the number stops moving — the finest two grids agree to within 2% | *settled* / **STILL MOVING** |
| error | the number keeps moving — shrinking, at a plausible observed order (≥ 0.5) | *converging at order p* / **NOT CONVERGING** |

Asking the wrong one inverts the answer. Second-order convergence moves an
error by 75% per doubling; judged by "did it stop moving?", a solver working
perfectly reads as a failure — and an error that *has* settled is one that has
stopped converging, which is the reading that actually deserves attention. The
exit code follows the verdict: `0` when every tracked metric behaved, `1` when
one did not.

The case's own aspect ratio is preserved as the grid refines, so a 2:1 channel
stays 2:1 rather than being squared into a different problem. Each case
declares which of its numbers are worth refining against in
`CONVERGENCE_METRICS`; only the case knows which are results and which are
bookkeeping.

```
=== cylinder grid study (48, 64) ===
  cd_mean  (quantity)
       N          value     change
      48        1.68308
      64         2.3936     42.22%
    STILL MOVING at 2% between the two finest grids
```

A quantity that is *structurally* zero — the lift on a symmetric steady wake —
is not read as a grid refusing to converge; two different roundings of zero
differ by tens of per cent and mean nothing.

**With three or more grids** the study also reports a Richardson extrapolation
and a Grid Convergence Index, in the form standardised by ASME V&V 20:

```
    grids 64 -> 128 -> 256   R = +0.2509  (monotonic)
    order from successive differences 1.995   extrapolated 1.34124   finest is 0.33% from it
    GCI 0.41%  ->  1.34013 .. 1.35127
```

Two grids say whether a number moved; a third says whether it is *going*
anywhere. The order comes first:

| `R` | meaning | what happens |
|---|---|---|
| `0 < R < 1` | monotonic convergence | extrapolated |
| `-1 < R < 0` | oscillatory convergence | extrapolated, with a caveat |
| `\|R\| ≥ 1` | divergence | **refused** — no number is printed |

That refusal is the point of the whole thing. Richardson extrapolation assumes
the sequence is already in its asymptotic regime, and handed one that is not it
returns a number anyway — a diverging triplet and a beautifully converging one
both produce a float. An observed order outside `[0.5, 4]` is likewise not
believed: a second-order scheme apparently converging at order 7 is reporting
the ratio of two differences made mostly of noise.

The **GCI** is the band the converged answer is expected to lie in, at roughly
95% confidence. It is what turns "Cd = 1.346 on the finest grid I could afford"
into "Cd = 1.341, ± 0.4%" — and it is usually more sobering than expected: on
grids too coarse for the body, bands of tens of per cent are normal and
correct. The sample above is a clean second-order sequence; the cavity on
16–32 cells reports a GCI near 70%, which is the honest answer for grids that
far from resolved.

Grids do not have to double — an uneven ladder is solved by the ASME
fixed-point iteration for the order. `pycfd.analysis.richardson.richardson()`
takes any three values directly, if you want the estimate without rerunning
anything:

```python
from pycfd.analysis.richardson import richardson

estimate = richardson([64, 128, 256], [1.412, 1.359, 1.3457])
print(estimate.report())
if estimate.trustworthy:
    print(estimate.extrapolated, estimate.band())
```

#### The end-to-end sanity pass

`--diagnose` asks every question above at once, plus the ones about the setup
itself, and reports a verdict per question:

```bash
python -m pycfd.main --diagnose                    # the cylinder, ~90 s
python -m pycfd.main --diagnose --case cavity
```

It runs the case on a **coarse and a medium grid, both below the one the case
would otherwise use** — half and three-quarters resolution — because refining
is the only way to find out whether a number has converged and the cheapest
place to look is underneath the grid you were going to run anyway. Everything
else about the configuration is exactly what you passed, so the diagnosis is of
the run you actually intended.

```
=== cylinder diagnosis: grids 128, 192 (the case's own default is 256) ===

  [ WARN ] convergence of cl_rms
           moved 21.42% between N=128 and N=192
           -> still moving at more than 2%; refine past N=192 before quoting it; a third grid would say where it is heading

  [ WARN ] blockage
           the body spans 12.5% of the cross-stream extent
           -> above 10% the walls accelerate the flow past the body and bias Cd upward by several per cent -- more than the spread between published unbounded values. --domain-height 20 would bring it under 5%

  [ WARN ] body resolution
           the body spans 8.0 -> 12.0 cells across the study's grids
           -> a staircase immersed boundary needs about 16 cells before forces are quantitative -- roughly N=256 here

  [  OK  ] continuity
           max|div u| = 3.4e-15, 5.4e-15 across the grids

  [  OK  ] convergence of cd_mean
           moved 0.68% between N=128 and N=192

  [  OK  ] case validation
           all 3 of the case's own checks passed at N=192

  0 failed, 3 warned, 3 passed
  usable, with the caveats above
```

The worst finding is printed first, because that is what you ran this for.
Read the shipped cylinder's three warnings as what they are — the case's own
docstring already says its drag is an engineering estimate, and these are the
reasons why.

**The checks.** Each one is skipped in silence when the case does not report
what it needs, which is what lets one list serve a closed cavity and an
aircraft silhouette alike:

| check | asks | `warn` | `fail` |
|---|---|---|---|
| stability | did every grid finish? | — | the solver blew up |
| continuity | how much divergence survived the projection? | > `1e-8` | > `1e-6` |
| convergence of *m* | did each refined metric behave? | still moving, not falling, or `\|R\| ≥ 1` | — |
| blockage | how much of the cross-stream extent does the body fill? | ≥ 5% | — |
| body resolution | how many cells span the body? | < 16 | < 4 |
| compressibility | is `M` inside the incompressible limit? | > 0.3 | — |
| case validation | did the case's own checks pass on the finest grid? | any failed | — |

The shipped cylinder is a well-behaved setup. A custom body dropped into the
same domain at a real flight speed is not, and every check has something to say:

```bash
python -m pycfd.main --diagnose --case cylinder \
    --geometry pycfd/geometry_import/f22_side_profile.csv \
    --l-ref 18.8 --wind-speed 180 --altitude 8000 --resolutions 64,96 --t-end 4
```

```
=== cylinder diagnosis: grids 64, 96 (the case's own default is 256) ===

  [ WARN ] convergence of cl_rms
           moved 73.49% between N=64 and N=96
           -> still moving at more than 2%; refine past N=96 before quoting it; a third grid would say where it is heading

  [ WARN ] blockage
           the body spans 32.2% of the cross-stream extent
           -> above 10% the walls accelerate the flow past the body and bias Cd upward by several per cent -- more than the spread between published unbounded values. --domain-height 51.5596 would bring it under 5%

  [ WARN ] body resolution
           the body spans 10.3 -> 15.5 cells across the study's grids
           -> a staircase immersed boundary needs about 16 cells before forces are quantitative -- roughly N=100 here

  [ WARN ] compressibility
           M = 0.584 at the stated conditions
           -> this solver has no density equation at all, so above M = 0.3 the result approximates a different flow rather than the same one slightly less well

  [ WARN ] case validation
           1 of 2 of the case's own checks passed at N=96; force average is stationary: the two halves of the averaging window differ by -3.7 standard errors; raise --transient above 0.5 or run longer
           -> N=96 is a diagnostic grid, below the one this case runs at, so a check that only just failed here may pass at full resolution -- but a non-stationary average or an undetected wake will not fix itself

  [  OK  ] continuity
           max|div u| = 1.3e-15, 2.3e-15 across the grids

  [  OK  ] convergence of cd_mean
           moved 0.91% between N=64 and N=96

  0 failed, 5 warned, 2 passed
  usable, with the caveats above
```

Five findings, none of which is wrong: an 18.8 m aircraft in a domain sized for
a unit cylinder really does block a third of the channel, 180 m/s at 8 km really
is M = 0.58 in a solver with no density equation, and a 4-second record really
has not settled — that last one is the honest consequence of the `--t-end 4`
that made this example cheap to run. `cd_mean`, meanwhile, has already stopped
moving at 0.91%. **Warnings are the caller's judgement to make, not the tool's**
— which is why none of these is a failure.

**The exit code carries the worst verdict**, which makes this usable as a gate:

| code | meaning |
|---|---|
| `0` | nothing flagged |
| `1` | usable, but somebody has a judgement to make |
| `2` | the setup will not support a quantitative answer |

Remedies are arithmetic, not hand-waving: the domain height it suggests really
does bring blockage under 5%, and the resolution it suggests really does put 16
cells across the body — on the shipped cylinder that works out to `N=256`,
which is the grid the case already chose.

**What it deliberately does not do is shorten the run.** A truncated record is
a different flow, not a cheaper look at the same one, and the force coefficient
it reports would answer a question nobody asked. The saving comes from the grid
alone. Pass `--t-end` yourself for a faster screen, and read the stationarity
verdict knowing that is what you asked for.

**Two grids or three.** The default pair says whether a number moved. Naming a
third with `--resolutions` buys Richardson extrapolation on every refined
metric, which replaces *refine past N=192* with the limit the sequence is
heading for and a GCI band around it — and, when the triplet turns out not to be
converging at all, says that instead of extrapolating anyway:

```bash
python -m pycfd.main --diagnose --resolutions 96,128,192
```

The same pass is available directly, if you want the findings rather than the
printout:

```python
from pycfd.analysis.diagnose import diagnose

d = diagnose("cylinder", geometry="pycfd/geometry_import/f22_side_profile.csv", l_ref=18.8)
print(d.level, d.usable)
for finding in d.findings:
    if finding.level != "ok":
        print(finding.name, "--", finding.detail, "|", finding.advice)
```

#### Worked recipes

Custom body from a vertex file, in a larger domain, with the wake anchored:

```bash
python -m pycfd.main --case cylinder --geometry pycfd/geometry_import/shield.csv \
    --re 100 --nx 384 --ny 192 --domain-length 24 --domain-height 12 \
    --geometry-scale 1.0 --geometry-rotate 0 \
    --outlet-type pressure_outlet --p-ref 0 \
    --t-end 120 --cfl 0.4 --name shield_run1 \
    --export-vtk --checkpoint --progress
```

A body sized in metres, at a real speed and altitude, reported against the
aircraft-length convention rather than the body's height:

```bash
python -m pycfd.main --case cylinder --geometry pycfd/geometry_import/f22_side_profile.csv \
    --l-ref 18.8 --wind-speed 70 --altitude 3000 \
    --nx 320 --ny 240 --domain-length 80 --domain-height 60 \
    --t-end 40 --name f22_70ms_3km --checkpoint
```

The report then carries both lengths (`reference_length` 18.8 next to
`characteristic_length` 2.58, the profile's real height) and the conditions the
Reynolds number came from, so nothing about the run has to be reconstructed
later.

Watch it develop instead, then keep going from where it stopped:

```bash
python -m pycfd.main --case cylinder --geometry pycfd/geometry_import/shield.csv --re 100 --live --display vorticity
python -m pycfd.main --resume results/shield_run1.npz --t-end 240
```

A quick low-resolution sanity pass before committing to a long run:

```bash
python -m pycfd.main --case cylinder --geometry pycfd/geometry_import/shield.csv --re 100 \
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

---

[← Back to the README](../README.md)
