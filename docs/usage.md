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
python -m pycfd.main --case cylinder --geometry f22_side_profile.csv --l-ref 18.8 --re 1e6
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
python -m pycfd.main --case cylinder --geometry f22_side_profile.csv \
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
python -m pycfd.main --case cylinder --geometry f22_side_profile.csv \
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

**Flow conditions** †

| flag | default | meaning |
|---|---|---|
| `--l-ref L` | the body's span across the flow | reference length for `Re`, `Cd`, `Cl` and `St`, in the geometry's units |
| `--wind-speed V` | none | free-stream speed in m/s; derives `Re` from it and the ISA viscosity. Mutually exclusive with `--re` |
| `--altitude Z` | `0` (sea level) | altitude in metres, selecting the ISA air properties `--wind-speed` uses |

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

A body sized in metres, at a real speed and altitude, reported against the
aircraft-length convention rather than the body's height:

```bash
python -m pycfd.main --case cylinder --geometry f22_side_profile.csv \
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

---

[← Back to the README](../README.md)
