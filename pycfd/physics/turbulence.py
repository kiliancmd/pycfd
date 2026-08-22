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
        if cs < 0:
            raise ValueError(f"Smagorinsky constant must be non-negative, got {cs}")
        self.mesh = mesh
        self.metrics = mesh.metrics
        self.cs = float(cs)
        #: Filter width: geometric mean of the cell dimensions.  On a stretched
        #: mesh this is per-cell -- the filter width *is* the local grid, so a
        #: single number would over-damp the fine region and under-damp the
        #: coarse one, which is the whole reason the grid was stretched.
        if mesh.is_uniform:
            self.delta: float | np.ndarray = float(np.sqrt(mesh.dx * mesh.dy))
            self._coeff: float | np.ndarray = (self.cs * self.delta) ** 2
        else:
            delta = np.sqrt(mesh.dx_cells[:, None] * mesh.dy_cells[None, :])
            self.delta = delta
            # Cell-centred quantities carry the ghosted pressure shape, so the
            # coefficient has to as well; the ghost ring repeats its neighbour.
            coeff = np.empty((mesh.nx + 2, mesh.ny + 2))
            coeff[1:mesh.nx + 1, 1:mesh.ny + 1] = (self.cs * delta) ** 2
            coeff[0, :], coeff[mesh.nx + 1, :] = coeff[1, :], coeff[mesh.nx, :]
            coeff[:, 0], coeff[:, mesh.ny + 1] = coeff[:, 1], coeff[:, mesh.ny]
            self._coeff = coeff
        self.last_nu_t_max = 0.0

    # ------------------------------------------------------------------ #
    def strain_rate_magnitude(self, u: np.ndarray, v: np.ndarray):
        """Return ``(|S| at cell centres, S_12 at corners)``.

        The cell-centred array carries the ghosted pressure shape so it can be
        indexed exactly like ``p``; the corner array has shape ``(nx+1, ny+1)``.
        """
        nx, ny = self.mesh.shape
        m = self.metrics

        # Normal strains are differenced across a cell (ghost-extended), shear
        # strains across a face -- the same two spacings the viscous term uses.
        s11 = (u[1:nx + 3, :] - u[0:nx + 2, :]) / m.hx_ext
        s22 = (v[:, 1:ny + 3] - v[:, 0:ny + 2]) / m.hy_ext

        dudy = (u[1:nx + 2, 1:ny + 2] - u[1:nx + 2, 0:ny + 1]) / m.hyv
        dvdx = (v[1:nx + 2, 1:ny + 2] - v[0:nx + 1, 1:ny + 2]) / m.hxu
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
        delta = (f"{self.delta:.4g}" if np.isscalar(self.delta)
                 else f"[{self.delta.min():.4g}, {self.delta.max():.4g}]")
        return f"SmagorinskyModel(cs={self.cs:g}, delta={delta})"
