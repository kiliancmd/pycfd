# Task: Pressure Outlet BC — Confirm, Implement, Validate, Document

Execute the steps below in order. Do not proceed past a GATE CHECK until it passes.

---

## STEP 1 — Confirm inlet pressure implementation

Read `pycfd/core/pressure.py` in full.

Find the section that assembles the Poisson matrix rows corresponding to the
**left (inlet) boundary**. Confirm one of the following is true:

- The inlet rows use a **Neumann stencil** (one-sided finite difference forcing
  `dp/dn = 0`), OR
- The inlet boundary is **excluded from Dirichlet treatment entirely** (interior
  stencil extended to the boundary, or rows left as Neumann by default).

**If inlet uses Dirichlet pressure (p = constant on left wall):**
Before doing anything else, fix this: replace the inlet rows with a one-sided
Neumann stencil (`p[0,j] = p[1,j]`, i.e. row sets `-p[1,j] + p[0,j] = 0`).
This is a bug and must be corrected regardless of the rest of this task.

**Report:** Print a clear one-line statement of what you found, quoting the
relevant line numbers and stencil type. Example:
> "Inlet (left boundary): Neumann, lines 84–91 — no change needed."
> "Inlet (left boundary): Dirichlet — corrected to Neumann at lines 84–91."

**GATE CHECK 1:** Inlet boundary confirmed as Neumann (either pre-existing or
just corrected). Do not continue until this is true.

---

## STEP 2 — Implement PressureOutlet boundary condition

### 2a — `pycfd/core/boundary.py`

Add the following class. Insert it directly after the existing `Outlet` class,
preserving all existing code:

```python
class PressureOutlet(BoundaryCondition):
    """
    Pressure outlet: Dirichlet pressure (p = p_ref) at the right boundary.
    Velocity uses zero-gradient outflow (same as Outlet).
    Physically correct for external aerodynamics — anchors the pressure
    level at the outlet rather than letting it float.
    """
    def __init__(self, p_ref: float = 0.0):
        self.p_ref = p_ref

    def apply_velocity(self, u, v, mesh):
        """Zero-gradient outflow — copy second-to-last column to last."""
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]

    def apply_pressure(self, p, mesh):
        """Dirichlet: fix outlet pressure to p_ref."""
        p[:, -1] = self.p_ref
```

### 2b — `pycfd/core/pressure.py`

Locate the Poisson matrix assembly method. Find the block that handles the
**right (outlet) boundary** — currently a Neumann stencil.

Add a branch: when `boundary_config.get("right") == "pressure_outlet"`, replace
those rows with Dirichlet identity rows instead:

```python
if boundary_config.get("right") == "pressure_outlet":
    for j in range(mesh.ny):
        idx = self._idx(mesh.nx - 1, j)
        # Zero the entire row, set diagonal to 1, RHS to p_ref
        self.A[idx, :] = 0.0
        self.A[idx, idx] = 1.0
        self.rhs_bc[idx] = self.p_ref
else:
    # existing Neumann stencil for right boundary — leave unchanged
    ...
```

Store `self.p_ref` as an instance attribute (read from boundary config or
default to 0.0). The matrix is assembled once at init — confirm this is the
case and that the branch is evaluated at assembly time, not each timestep.

### 2c — `pycfd/config.py`

Add `"pressure_outlet"` as a valid string value in the boundary config
documentation / type hint / enum (however valid BC types are currently
registered). Do not change any default case configs — only register the new type.

### 2d — `pycfd/cases/cylinder_flow.py`

Locate the boundary config for the cylinder case. Change:
```python
"right": "outlet"
```
to:
```python
"right": "pressure_outlet"
```
Leave all other boundaries unchanged.

---

## STEP 3 — Validate

### 3a — Unit test: Dirichlet pressure at outlet

Add the following test to `pycfd/tests/test_pressure.py`:

```python
def test_pressure_outlet_dirichlet():
    """
    After one solver step with pressure_outlet BC, the rightmost pressure
    column must equal p_ref (0.0) to within floating-point tolerance.
    """
    from pycfd.core.mesh import StructuredMesh
    from pycfd.core.solver import NavierStokesSolver
    from pycfd.config import SimConfig

    cfg = SimConfig(
        nx=32, ny=32, lx=4.0, ly=4.0,
        re=100.0, dt=1e-3, t_end=1e-3,
        boundary_config={
            "left":   "inlet",
            "right":  "pressure_outlet",
            "top":    "symmetry",
            "bottom": "symmetry",
        }
    )
    mesh = StructuredMesh(cfg)
    solver = NavierStokesSolver(cfg, mesh)
    solver.step()

    assert np.allclose(solver.p[:, -1], 0.0, atol=1e-10), \
        f"Outlet pressure not anchored: max deviation = {np.max(np.abs(solver.p[:, -1])):.2e}"
```

### 3b — Unit test: inlet remains Neumann

Add the following test to `pycfd/tests/test_pressure.py`:

```python
def test_inlet_pressure_neumann():
    """
    Inlet pressure must satisfy dp/dx = 0 (Neumann), i.e. p[:, 0] ≈ p[:, 1].
    If inlet were Dirichlet, p[:, 0] would be fixed regardless of interior —
    the Neumann condition allows it to adjust freely.
    """
    from pycfd.core.mesh import StructuredMesh
    from pycfd.core.solver import NavierStokesSolver
    from pycfd.config import SimConfig

    cfg = SimConfig(
        nx=32, ny=32, lx=4.0, ly=4.0,
        re=100.0, dt=1e-3, t_end=5e-3,
        boundary_config={
            "left":   "inlet",
            "right":  "pressure_outlet",
            "top":    "symmetry",
            "bottom": "symmetry",
        }
    )
    mesh = StructuredMesh(cfg)
    solver = NavierStokesSolver(cfg, mesh)
    for _ in range(5):
        solver.step()

    dp_dx_inlet = np.abs(solver.p[:, 0] - solver.p[:, 1])
    assert np.all(dp_dx_inlet < 1e-6), \
        f"Inlet Neumann condition violated: max dp/dx = {dp_dx_inlet.max():.2e}"
```

### 3c — Run full test suite

```bash
cd <project_root>
pytest pycfd/tests/ -v
```

**GATE CHECK 2:** Both new tests pass. No previously passing tests regress.
If any existing test fails, fix the regression before continuing.

### 3d — Smoke test: cylinder case

Run the cylinder case for 200 timesteps with the pressure outlet active:

```bash
python3 -m pycfd.main --case cylinder --re 100 --nx 64 --ny 64 --t-end 0.1
```

Confirm:
- No NaN or divergence in output
- Outlet pressure column printed/logged near 0.0
- Simulation completes without error

**GATE CHECK 3:** Smoke test passes cleanly.

---

## STEP 4 — Update README.md

Locate the section that documents boundary conditions (create one under a
`## Boundary Conditions` heading if it does not exist).

Add an entry for `pressure_outlet` in the same style as existing BC entries.
The entry must cover:

- What it is: Dirichlet pressure at the outlet (`p = p_ref`, default 0.0)
- When to use it: external aerodynamics, open-domain flows
- How it differs from `outlet`: `outlet` is Neumann (`dp/dn = 0`, pressure
  floats); `pressure_outlet` is Dirichlet (pressure anchored)
- Configuration example:

```python
boundary_config = {
    "left":   "inlet",
    "right":  "pressure_outlet",   # anchors p = 0 at outlet
    "top":    "symmetry",
    "bottom": "symmetry",
}
```

- Note that inlet must remain Neumann when using pressure_outlet on the right —
  combining Dirichlet pressure on both inlet and outlet over-constrains the
  Poisson system.

Do not rewrite or reformat any other part of the README. Add only the new entry.

---

## Completion Checklist

Before reporting done, confirm every item:

- [ ] GATE CHECK 1: Inlet confirmed as Neumann (or corrected)
- [ ] `PressureOutlet` class added to `boundary.py`
- [ ] Poisson matrix assembly branches correctly on `"pressure_outlet"`
- [ ] `"pressure_outlet"` registered as valid BC type in `config.py`
- [ ] Cylinder case updated to use `"pressure_outlet"` on right boundary
- [ ] GATE CHECK 2: Both new pressure tests pass, no regressions
- [ ] GATE CHECK 3: Cylinder smoke test completes without NaN or error
- [ ] README updated with `pressure_outlet` documentation
