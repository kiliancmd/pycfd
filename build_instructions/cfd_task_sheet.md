# CFD Simulation Program — Claude Code Build Instructions

---

## Feasibility Assessment

**Verdict: FEASIBLE — Well-suited for Claude Code execution.**

Python is a proven platform for CFD at the educational and research-prototype tier.
NumPy provides vectorized array operations, SciPy supplies sparse linear solvers and
spatial routines, and Matplotlib handles publication-quality visualization — all
pre-installable via pip with no licensing friction. The finite-difference and
finite-volume methods this project targets are numerically straightforward and map
cleanly onto structured NumPy arrays.

**What this will be:** A 2D incompressible Navier-Stokes solver capable of running
classic benchmark problems (lid-driven cavity, channel flow, flow past a cylinder)
with real-time visualization, convergence diagnostics, and basic post-processing —
roughly comparable to an early-career research tool or a teaching code.

**What this will NOT be:** A production CFD suite. It will not match the speed of
compiled solvers (OpenFOAM, Fluent), handle complex 3D geometries, or include
industrial turbulence models. That is out of scope and not the goal.

**Key risks and mitigations:**

| Risk | Mitigation |
|---|---|
| Performance on large grids | Numba JIT for hot loops; keep default grids ≤ 256×256 |
| Numerical instability | CFL-based adaptive Δt; validated against analytical solutions |
| Scope creep | Strict phase gates below; each phase is independently testable |

---

## Architecture Overview

```
pycfd/
├── core/
│   ├── mesh.py              # Structured mesh generation
│   ├── solver.py            # Navier-Stokes solver (projection method)
│   ├── pressure.py          # Pressure Poisson equation solvers
│   ├── boundary.py          # Boundary condition handlers
│   └── timestepper.py       # Time integration + adaptive CFL
├── physics/
│   ├── incompressible.py    # Incompressible flow driver
│   └── turbulence.py        # Smagorinsky SGS model (optional)
├── geometry/
│   └── obstacles.py         # Immersed boundary / mask-based obstacles
├── analysis/
│   ├── postprocess.py       # Derived quantities (vorticity, stream function, forces)
│   ├── validation.py        # Analytical solution comparisons
│   └── export.py            # VTK / CSV / image export
├── visualization/
│   ├── live_plot.py         # Real-time Matplotlib animation
│   └── static_plot.py       # Publication-quality static figures
├── cases/
│   ├── lid_driven_cavity.py # Benchmark: lid-driven cavity
│   ├── channel_flow.py      # Benchmark: Poiseuille / channel flow
│   └── cylinder_flow.py     # Benchmark: flow past a circular cylinder
├── tests/
│   ├── test_mesh.py
│   ├── test_solver.py
│   ├── test_pressure.py
│   └── test_validation.py
├── main.py                  # CLI entry point
├── config.py                # Dataclass-based configuration
└── requirements.txt
```

---

## Build Phases

Each phase ends with a **gate check** — a concrete test that proves the phase works
before moving on. Do not skip gate checks.

---

### PHASE 1 — Mesh & Configuration Foundation

**Goal:** Structured 2D Cartesian mesh + typed configuration object.

**Tasks:**

1. Create `config.py` using Python `dataclasses`:
   - `nx`, `ny` (grid points, default 128×128)
   - `lx`, `ly` (domain length, default 1.0×1.0)
   - `dt` (time step), `t_end` (end time)
   - `re` (Reynolds number)
   - `cfl_max` (max CFL number, default 0.5)
   - `solver_type` (enum: `"projection"`, `"simple"`)
   - `boundary_config` (dict mapping each wall to a BC type)

2. Create `core/mesh.py`:
   - `StructuredMesh` class storing `x`, `y`, `dx`, `dy` as NumPy arrays
   - Cell-center and face-center coordinate generators
   - Support for uniform and basic geometric stretching

3. Write `tests/test_mesh.py`:
   - Verify coordinate array shapes, spacing uniformity, domain bounds

**Gate check:** `pytest tests/test_mesh.py` passes. Mesh coordinates print correctly
for a 32×32 and a 128×128 domain.

---

### PHASE 2 — Core Navier-Stokes Solver (Projection Method)

**Goal:** Working incompressible N-S solver using Chorin's projection method.

**Tasks:**

1. Create `core/solver.py` — implement the fractional-step algorithm:
   - **Step A — Advection:** Compute convective term `(u·∇)u`
     using 2nd-order central differences (or upwind for stability option)
   - **Step B — Diffusion:** Compute viscous term `(1/Re)∇²u`
     using 2nd-order central differences
   - **Step C — Intermediate velocity:** `u* = uⁿ + Δt(−Advection + Diffusion)`
   - **Step D — Pressure Poisson:** Solve `∇²p = (1/Δt)∇·u*`
   - **Step E — Projection:** `uⁿ⁺¹ = u* − Δt·∇p`

2. Create `core/pressure.py` — Pressure Poisson solver:
   - Primary: SciPy `spsolve` with pre-assembled sparse matrix (CSR format)
   - Secondary: Iterative Jacobi/Gauss-Seidel for educational transparency
   - Poisson matrix should be assembled once at init, not every timestep

3. Create `core/timestepper.py`:
   - Forward Euler (baseline)
   - CFL-based adaptive time step: `dt = cfl_max * min(dx, dy) / max(|u|, |v|)`
   - Timestep loop with iteration counter and elapsed-time tracking

**Gate check:** Solver runs for 100 timesteps on a 64×64 zero-initialized domain
without NaN or divergence. Velocity field remains bounded.

---

### PHASE 3 — Boundary Conditions

**Goal:** Modular boundary condition system supporting the standard types.

**Tasks:**

1. Create `core/boundary.py`:
   - `BoundaryCondition` base class with `apply(field, mesh)` method
   - `NoSlip` — velocity = 0 at wall (Dirichlet)
   - `MovingWall` — tangential velocity = U_wall (for lid-driven cavity)
   - `Inlet` — prescribed velocity profile (uniform or parabolic)
   - `Outlet` — zero-gradient / convective outflow (Neumann)
   - `Periodic` — wrap values across opposite boundaries
   - `Symmetry` — zero normal velocity, zero normal gradient of tangential

2. Apply BCs at the correct point in the projection algorithm:
   - After intermediate velocity computation (Step C)
   - After pressure correction (Step E)
   - Pressure BCs: Neumann (dp/dn = 0) at solid walls

3. Write `tests/test_solver.py`:
   - Lid-driven cavity at Re=100 for 1000 steps → check symmetry of flow
   - Channel flow at Re=10 → compare centerline velocity to Poiseuille solution

**Gate check:** Lid-driven cavity produces a recognizable primary vortex.
Channel flow centerline velocity matches Poiseuille analytical solution within 2%.

---

### PHASE 4 — Visualization

**Goal:** Both real-time animation and static publication-quality plots.

**Tasks:**

1. Create `visualization/live_plot.py`:
   - Uses `matplotlib.animation.FuncAnimation`
   - Configurable display: velocity magnitude contour, velocity vectors (quiver),
     pressure contour, or streamlines
   - Shows iteration count, simulation time, and max CFL in title bar
   - Colorbar with appropriate scaling
   - Pause/resume capability

2. Create `visualization/static_plot.py`:
   - Velocity magnitude contour with streamline overlay
   - Pressure field contour
   - Vorticity field contour
   - Centerline velocity profiles (u along vertical, v along horizontal)
   - Side-by-side comparison layout for validation plots
   - Save to PNG at 300 DPI

3. Integrate into `main.py`:
   - `--live` flag for real-time animation
   - `--plot-every N` to control update frequency (default: every 50 steps)

**Gate check:** Lid-driven cavity at Re=400 produces a clean animated flow and
a static 4-panel figure (velocity, pressure, vorticity, streamlines).

---

### PHASE 5 — Benchmark Cases

**Goal:** Three runnable, validated benchmark cases with one-command execution.

**Tasks:**

1. `cases/lid_driven_cavity.py`:
   - Pre-configured for Re = 100, 400, 1000
   - Compare against Ghia et al. (1982) reference data for centerline profiles
   - Auto-generates validation plots

2. `cases/channel_flow.py`:
   - Poiseuille flow between parallel plates
   - Validates against analytical parabolic profile
   - Reports L2 error norm

3. `cases/cylinder_flow.py`:
   - Flow past a circular cylinder using mask-based immersed boundary
   - `geometry/obstacles.py`: circle mask generator with interpolated boundary
   - Pre-configured for Re = 20 (steady), Re = 100 (vortex shedding)
   - Computes drag coefficient Cd

4. Each case callable as: `python main.py --case cavity --re 400 --live`

**Gate check:** All three cases run to completion. Cavity matches Ghia data.
Channel matches Poiseuille. Cylinder at Re=100 shows vortex shedding visually.

---

### PHASE 6 — Analysis & Post-Processing

**Goal:** Quantitative analysis tools beyond raw field visualization.

**Tasks:**

1. `analysis/postprocess.py`:
   - Vorticity computation: `ω = ∂v/∂x − ∂u/∂y`
   - Stream function computation (solve Poisson: `∇²ψ = −ω`)
   - Drag and lift force integration (pressure + viscous contributions)
   - Kinetic energy: `KE = 0.5 * Σ(u² + v²) * dx * dy`
   - Enstrophy: `E = 0.5 * Σ(ω²) * dx * dy`
   - Point probe: time series of velocity/pressure at a specified (x,y)

2. `analysis/validation.py`:
   - Poiseuille analytical profile generator
   - Ghia et al. reference data (hardcoded arrays for Re=100,400,1000)
   - L2 and L∞ error norm computation
   - Convergence rate estimation (run at 2–3 grid resolutions, compute order)

3. `analysis/export.py`:
   - VTK structured grid export (for ParaView visualization)
   - CSV export of field data
   - NumPy `.npz` checkpoint save/load for resuming simulations

**Gate check:** Cavity case exports to VTK, opens in ParaView (or verifiable via
vtk Python reader). Grid convergence study shows 2nd-order convergence rate.

---

### PHASE 7 — Performance & Polish

**Goal:** Optimize hot paths, add CLI ergonomics, and harden edge cases.

**Tasks:**

1. **Performance:**
   - Profile with `cProfile` to identify bottlenecks
   - Add Numba `@njit` to advection and diffusion stencil loops
   - Benchmark: report wall-clock seconds per 1000 timesteps at 128×128 and 256×256
   - Target: ≥ 3× speedup on stencil operations with Numba

2. **CLI (`main.py`):**
   - Full `argparse` interface:
     `--case`, `--re`, `--nx`, `--ny`, `--dt`, `--t-end`,
     `--live`, `--plot-every`, `--export-vtk`, `--export-csv`,
     `--checkpoint`, `--resume`
   - `--list-cases` to show available benchmarks
   - Progress bar using `tqdm`

3. **Robustness:**
   - Divergence detection: abort with clear message if `max(|u|) > 1e6`
   - CFL warning if adaptive step drops below `1e-8`
   - Input validation on all config parameters (Re > 0, nx ≥ 4, etc.)
   - Logging via Python `logging` module (INFO default, DEBUG for diagnostics)

4. **Documentation:**
   - `README.md` with installation, quickstart, and example output images
   - Docstrings on all public classes and functions
   - `requirements.txt`: `numpy`, `scipy`, `matplotlib`, `numba`, `tqdm`

**Gate check:** Full test suite passes. `python main.py --case cavity --re 1000 --nx 128 --live`
runs without errors and produces physically correct results. README is self-contained.

---

## Dependency Stack

| Package | Purpose | Version Constraint |
|---|---|---|
| `numpy` | Array operations, vectorized math | ≥ 1.24 |
| `scipy` | Sparse linear solvers, spatial routines | ≥ 1.10 |
| `matplotlib` | Visualization (static + animated) | ≥ 3.7 |
| `numba` | JIT compilation for stencil loops | ≥ 0.57 |
| `tqdm` | Progress bars | ≥ 4.65 |
| `pytest` | Testing | ≥ 7.0 |

No compiled dependencies beyond what pip installs. No GPU requirement.

---

## Quality Rules (Enforce Throughout)

- **No magic numbers.** Every physical or numerical constant lives in `config.py`
  or is a named variable with a comment.
- **Vectorize first.** Write NumPy array operations before reaching for explicit loops.
  Only use Numba for loops that genuinely can't be vectorized.
- **Test at boundaries.** Every BC type gets a unit test. Every solver test checks
  that boundary values are exactly what was prescribed.
- **Fail loud.** Divergence, invalid config, or NaN should raise with a clear message,
  never silently produce garbage.
- **Separate physics from I/O.** Solver functions return arrays. Visualization and export
  are separate layers that receive arrays. No Matplotlib imports inside `core/`.

---

## Success Criteria

The project is done when:

1. `python main.py --case cavity --re 1000 --nx 128 --live` runs and shows
   correct vortex structure in real time
2. `python main.py --case cylinder --re 100 --live` shows vortex shedding
3. Validation plots match Ghia et al. reference data within published tolerances
4. Grid convergence study confirms 2nd-order spatial accuracy
5. All tests pass, README documents installation and usage, and the whole project
   is runnable from a clean `pip install -r requirements.txt`
