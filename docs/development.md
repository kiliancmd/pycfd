# Development

Testing, continuous integration, project layout, and the limits of what this
solver represents.

---

## Testing

```bash
python -m pytest pycfd/tests -q
```

517 tests covering mesh geometry, obstacle construction from every supported
source, all six boundary condition types, Poisson assembly and all four linear
solvers, discrete conservation, the analytical benchmarks, dimensional
bookkeeping, CLI plumbing, output provenance, and the export round-trips — plus
6 full-fidelity benchmark regressions behind `--runslow`.

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
  divergence-free exactly as a built-in one does;
- the standard atmosphere reproduces the *published* ISA table rather than its
  own output, so a rearranged formula fails instead of being re-recorded, and a
  configuration whose `u_ref` disagrees with its own inlet is refused;
- Richardson extrapolation recovers a *constructed* answer exactly — build
  `f(h) = f_exact + C h^p`, hand it three grids, and require `p` and `f_exact`
  back — on uneven grid ladders as well as doubling ones, since only an uneven
  ladder can catch a swapped refinement ratio; and a diverging triplet is
  refused rather than extrapolated;
- the autocorrelation time of an AR(1) process comes back as its analytical
  `(1+φ)/(1-φ)`, and doubling a record's sampling rate leaves the reported
  error bar unchanged — the property a naive `s/√N` gets wrong;
- shedding detection **refuses** white noise, a linear drift, an exponential
  decay and a record spanning too few periods. Each is a signal
  `strouhal_number()` answers with a confident frequency, so the negative cases
  are the ones under test;
- the sanity pass reads an error that fell 75% under refinement as a
  second-order scheme working, and one that has *settled* as a solver that
  stopped converging — the two readings are opposite for an error and a measured
  quantity, and asking the wrong question inverts the verdict. Its remedies are
  checked arithmetically rather than textually: the domain height it suggests is
  asserted to bring blockage under the threshold, and the resolution it suggests
  to put 16 cells across the body;
- the channel's centreline error is pinned as being set by `steady_tol` and
  **not** by the grid — tightening the tolerance 100× tightens it 100×, while
  refining leaves it at 1.01e-7 %. That is why it is not one of the channel's
  convergence metrics, and the test asserts both halves so the exclusion cannot
  be undone by accident;
- the cavity and the channel accept a caller-supplied `steady_tol` and fall
  back to their own default when none is given — both cases used to pass it as
  an explicit keyword alongside `**overrides`, so any caller who supplied it
  collided with the case's own default instead of overriding it.

---

## Regression baselines

The benchmark tests assert *tolerances* — convergence order above 1.8, cavity
profiles within 0.02 of Ghia. Those catch a break but not a drift: a change that
quietly moved second-order convergence from 1.9997 to 1.85 would pass every one
of them.

`pycfd/tests/baselines.json` therefore records the measured values themselves,
and `tests/test_regression.py` pins them. The file is self-describing — it
records the version, platform, Python and commit that produced each number.

Two tiers, because fidelity costs time:

| tier | resolution | runtime | when it runs |
|---|---|---|---|
| **fast** | reduced | ~10 s | every push, part of the default suite |
| **full** | as published in [Validation](validation.md) | ~9 min | `--runslow`, and on `main` in CI |

```bash
python -m pytest pycfd/tests -q                 # fast tier included
python -m pytest pycfd/tests --runslow -q       # plus full-fidelity benchmarks
```

The solver is deterministic — repeated runs are bit-identical on one machine —
so the tolerance exists only to absorb float variation between platforms and
BLAS builds. Quantities that are *structurally* zero (a divergence the
projection drives out, the lift on a symmetric steady wake) are checked against
an absolute bound instead, since comparing them relatively is meaningless.

**A failure here is not automatically a bug.** It means a number moved, and
someone has to decide whether that was a regression or an improvement.
Re-recording is deliberately a separate command rather than a pytest flag — a
baseline that refreshes itself records whatever the code currently does, which
is the opposite of what a regression test is for:

```bash
python tools/record_baselines.py --fast     # ~10 s
python tools/record_baselines.py --full     # ~9 min
```

Commit the regenerated file in the same change that caused the movement, so the
diff shows the old and new numbers side by side.

## Continuous integration

`.github/workflows/tests.yml` runs on every push to `main`, every pull request,
and on demand:

- **suite** — the full test suite across Python 3.10, 3.11 and 3.12, with
  `fail-fast: false` so one version failing does not hide the others. The
  installed `numpy` / `scipy` / `matplotlib` / `numba` versions are printed
  before the run, which makes a platform-specific baseline failure diagnosable
  from the log alone.
- **benchmarks** — the full-fidelity regressions. **Opt-in only**: tick
  *Also rerun the benchmarks at their published resolutions* when starting a run
  from the Actions tab. It never fires on a push.

No install step is needed: `pycfd/conftest.py` puts the repository root on
`sys.path`, so a fresh checkout runs directly.

### Why the benchmarks are opt-in

They cost minutes of solver time on a warm 8-core laptop and considerably more
on a shared two-core runner — the first automatic attempt ran for 64 minutes
without finishing and had to be cancelled by hand. Since the fast tier already
covers the same cases on every push, only the *published numbers* need the
expensive rerun, and that is worth asking for explicitly: after a numerics
change, or before tagging a release.

Both jobs carry a `timeout-minutes` so a hang fails loudly rather than
occupying a runner for GitHub's six-hour default. The benchmark job's 90-minute
bound is deliberately generous because nobody has yet seen it finish on a
runner; it runs pytest with `--durations=0`, so the first completed run
replaces that guess with a measurement and the bound can be tightened.

The concurrency key includes the event name. Without that, pushing to `main`
while a hand-requested benchmark run was in flight would cancel it — which is
exactly what `cancel-in-progress` is supposed to do for redundant pushes, and
exactly what it must not do to a run someone deliberately started.

## Provenance in outputs

A results directory accumulates figures, VTK dumps and checkpoints that look
alike and are impossible to tell apart a week later. Every export therefore
records the run that produced it — the exact command, the configuration, the
version, and the commit — through two channels:

| channel | formats | holds |
|---|---|---|
| embedded | PNG `tEXt` chunks, VTK title line, a field in the `.npz` | a digest, travelling inside the file |
| `<name>.provenance.json` sidecar | all of them, including CSV | the full record with the complete config |

CSV is deliberately left pure — a `#` banner would break readers that do not
expect one — so for that format the sidecar is the only channel.

```bash
exiftool -Comment results/cavity/cavity_Re100_fields.png
python -c "import json; print(json.load(open('results/run.provenance.json'))['command'])"
```

```python
from pycfd.analysis.export import checkpoint_provenance
record = checkpoint_provenance("results/run.npz")
print(record["git_commit"], record["extra"]["cd"])
```

The record also carries the run's own diagnostics, so an exported file says not
just how the run was configured but what it had converged to. A `+` on the
commit hash means the working tree had uncommitted changes — a hash alone would
imply a reproducibility it does not have.

## Architecture

```
.
├── README.md            Project front page
├── requirements.txt     numpy, scipy, matplotlib, numba, tqdm
├── docs/                This guide and its siblings
├── tools/               record_baselines.py
├── .github/workflows/   tests.yml
└── pycfd/               The package itself

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
│   ├── timeseries.py    Correlated-sample averaging: autocorrelation time,
│   │                    effective sample size, stationarity
│   ├── shedding.py      Is a wake periodic, or does it merely have a peak?
│   ├── richardson.py    Richardson extrapolation + GCI, and the regime check
│   │                    that decides whether either is worth reporting
│   ├── diagnose.py      The end-to-end sanity pass: every question above at
│   │                    once, one verdict each, worst first
│   ├── provenance.py    What produced a given output file
│   └── export.py        VTK / CSV / NPZ checkpoints
├── visualization/
│   ├── static_plot.py   Publication figures at 300 DPI
│   └── live_plot.py     FuncAnimation viewer
├── cases/               cavity, channel, cylinder, taylor_green; plus the
│                        grid-study driver they share
├── tests/               mesh, geometry, boundary, pressure, solver, validation,
│                        units, timeseries, shedding, gridstudy, richardson,
│                        diagnose, cli, provenance, regression
│                        + baselines.json
├── config.py            Dataclass configuration; every constant lives here
├── units.py             ISA atmosphere and the solver-unit <-> SI bridge
└── main.py              CLI
```

`pycfd/` holds nothing but the package, so everything importable is under one
roof and everything else — the README GitHub renders, the docs, the CI
workflow, the baseline recorder — sits at the repository root where each is
conventionally looked for.

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

---

[← Back to the README](../README.md)
