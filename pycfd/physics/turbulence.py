"""Smagorinsky sub-grid-scale model (optional LES closure).

The eddy viscosity follows the standard algebraic form

    nu_t = (Cs * Delta)^2 * |S| ,      |S| = sqrt(2 S_ij S_ij)

with the filter width taken as the geometric mean of the cell dimensions.  The
strain-rate components sit naturally on the staggered grid: the normal rates
``du/dx`` and ``dv/dy`` at cell centres, the shear rate ``S_12`` at cell corners.
``S_12`` is averaged to the centres to form ``|S|``, and the resulting viscosity
is interpolated back to the corners for the shear-stress term.

The model is off by default.  When it is on, the solver switches from the
constant-viscosity Laplacian to the full stress-divergence form of the viscous
term -- the only form that is correct for a spatially varying viscosity.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import StructuredMesh


class SmagorinskyModel:
    """Algebraic eddy-viscosity closure on the staggered grid.

    Parameters
    ----------
    mesh:
        Uniform structured mesh.
    cs:
        Smagorinsky constant; 0.17 is the classical homogeneous-turbulence
        value, 0.1 is common for wall-bounded flows.
    """

    def __init__(self, mesh: StructuredMesh, cs: float = 0.17) -> None:
        mesh.require_uniform("Smagorinsky model")
        if cs < 0:
            raise ValueError(f"Smagorinsky constant must be non-negative, got {cs}")
        self.mesh = mesh
        self.cs = float(cs)
        #: Filter width: geometric mean of the cell dimensions.
        self.delta = float(np.sqrt(mesh.dx * mesh.dy))
        self._coeff = (self.cs * self.delta) ** 2
        self.last_nu_t_max = 0.0

    # ------------------------------------------------------------------ #
    def strain_rate_magnitude(self, u: np.ndarray, v: np.ndarray):
        """Return ``(|S| at cell centres, S_12 at corners)``.

        The cell-centred array carries the ghosted pressure shape so it can be
        indexed exactly like ``p``; the corner array has shape ``(nx+1, ny+1)``.
        """
        nx, ny = self.mesh.shape
        dx, dy = self.mesh.dx, self.mesh.dy

        s11 = (u[1:nx + 3, :] - u[0:nx + 2, :]) / dx
        s22 = (v[:, 1:ny + 3] - v[:, 0:ny + 2]) / dy

        dudy = (u[1:nx + 2, 1:ny + 2] - u[1:nx + 2, 0:ny + 1]) / dy
        dvdx = (v[1:nx + 2, 1:ny + 2] - v[0:nx + 1, 1:ny + 2]) / dx
        s12_corner = 0.5 * (dudy + dvdx)

        # Average the four surrounding corners onto each interior cell centre,
        # then extend to the ghost ring so the viscosity is defined everywhere.
        s12_cc = np.zeros_like(s11)
        s12_cc[1:nx + 1, 1:ny + 1] = 0.25 * (
            s12_corner[0:nx, 0:ny] + s12_corner[1:nx + 1, 0:ny]
            + s12_corner[0:nx, 1:ny + 1] + s12_corner[1:nx + 1, 1:ny + 1]
        )
        s12_cc[0, :] = s12_cc[1, :]
        s12_cc[nx + 1, :] = s12_cc[nx, :]
        s12_cc[:, 0] = s12_cc[:, 1]
        s12_cc[:, ny + 1] = s12_cc[:, ny]

        s_mag = np.sqrt(2.0 * s11 ** 2 + 2.0 * s22 ** 2 + 4.0 * s12_cc ** 2)
        return s_mag, s12_corner

    # ------------------------------------------------------------------ #
    def eddy_viscosity(self, u: np.ndarray, v: np.ndarray, nu_molecular: float):
        """Effective viscosity ``nu + nu_t`` at cell centres and at corners.

        Returns ``(nu_c, nu_corner)`` shaped for
        :meth:`~pycfd.core.solver.ProjectionSolver._diffusion_variable`.
        """
        nx, ny = self.mesh.shape
        s_mag, _ = self.strain_rate_magnitude(u, v)
        nu_t = self._coeff * s_mag
        self.last_nu_t_max = float(nu_t.max())
        nu_c = nu_molecular + nu_t

        nu_corner = 0.25 * (
            nu_c[0:nx + 1, 0:ny + 1] + nu_c[1:nx + 2, 0:ny + 1]
            + nu_c[0:nx + 1, 1:ny + 2] + nu_c[1:nx + 2, 1:ny + 2]
        )
        return nu_c, nu_corner

    def __repr__(self) -> str:
        return f"SmagorinskyModel(cs={self.cs:g}, delta={self.delta:.4g})"
