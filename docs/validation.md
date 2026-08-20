# Validation

Measured agreement against analytical solutions and published reference data.
Every number here is pinned by an executable test — see
[Development](development.md#regression-baselines).

---

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

This is the only case with an exact solution, and so the only one that can
report a true order of accuracy. Any other case can still be refined and
watched — see [grid-refinement studies](usage.md#grid-refinement-studies) —
but a body with no reference to be wrong against gets a verdict on whether its
numbers stopped moving, not an extrapolated order.

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

---

[← Back to the README](../README.md)
