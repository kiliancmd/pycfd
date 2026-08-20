# Numerical method

How the solver is discretised, and why each choice was made.

---

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

---

[← Back to the README](../README.md)
