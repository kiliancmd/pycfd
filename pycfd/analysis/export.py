"""Field export: VTK for ParaView, CSV for spreadsheets, NPZ checkpoints.

All exporters work from cell-centred data so that every quantity shares one
grid, which is what external tools expect.  The staggered face values are
interpolated to centres on the way out; checkpoints instead store the raw
ghosted arrays so a resumed run continues bit-for-bit.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..config import SimulationConfig
from ..core.fields import FlowField
from ..core.mesh import StructuredMesh
from .postprocess import vorticity
from .provenance import provenance_record, summary_line, write_sidecar


def _cell_data(fields: FlowField) -> dict[str, np.ndarray]:
    """Cell-centred bundle shared by the VTK and CSV writers."""
    uc, vc = fields.cell_velocities()
    return {
        "u": uc,
        "v": vc,
        "speed": np.hypot(uc, vc),
        "pressure": fields.p_phys,
        "vorticity": vorticity(fields),
    }


# --------------------------------------------------------------------------- #
# VTK
# --------------------------------------------------------------------------- #
def export_vtk(fields: FlowField, path: str | Path, name: str = "pycfd",
               provenance: dict | None = None) -> Path:
    """Write a legacy ASCII VTK ``RECTILINEAR_GRID`` file.

    Cell-centre coordinates become grid points carrying ``POINT_DATA``, which is
    what ParaView's contour and streamline filters expect.  The legacy format is
    written directly so the package needs no ``vtk`` dependency.

    The format's single free-text title line carries a provenance digest, and
    the full record is written to a ``.provenance.json`` sidecar.
    """
    path = Path(path)
    record = provenance_record() if provenance is None else provenance
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = fields.mesh
    data = _cell_data(fields)
    nx, ny = mesh.shape

    def flat(a: np.ndarray) -> np.ndarray:
        # VTK orders points with x fastest, then y, then z.
        return np.asarray(a, dtype=float).T.ravel()

    with path.open("w", encoding="ascii") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        # The title is the one free-text slot the legacy format offers.
        fh.write(f"{name} t={fields.t:.6g} step={fields.step} "
                 f"[{summary_line(record, limit=180)}]\n")
        fh.write("ASCII\nDATASET RECTILINEAR_GRID\n")
        fh.write(f"DIMENSIONS {nx} {ny} 1\n")
        fh.write(f"X_COORDINATES {nx} float\n")
        fh.write(" ".join(f"{x:.7g}" for x in mesh.xc) + "\n")
        fh.write(f"Y_COORDINATES {ny} float\n")
        fh.write(" ".join(f"{y:.7g}" for y in mesh.yc) + "\n")
        fh.write("Z_COORDINATES 1 float\n0\n")
        fh.write(f"POINT_DATA {nx * ny}\n")

        for field_name in ("pressure", "speed", "vorticity"):
            fh.write(f"SCALARS {field_name} float 1\nLOOKUP_TABLE default\n")
            fh.write("\n".join(f"{val:.7g}" for val in flat(data[field_name])) + "\n")

        fh.write("VECTORS velocity float\n")
        uu, vv = flat(data["u"]), flat(data["v"])
        fh.write("\n".join(f"{a:.7g} {b:.7g} 0" for a, b in zip(uu, vv)) + "\n")

    write_sidecar(path, record)
    return path


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def export_csv(fields: FlowField, path: str | Path,
               provenance: dict | None = None) -> Path:
    """Write one row per cell: ``x, y, u, v, speed, pressure, vorticity``.

    The file itself stays pure data -- a comment banner would break readers that
    do not expect one -- so provenance goes to a ``.provenance.json`` sidecar.
    """
    path = Path(path)
    record = provenance_record() if provenance is None else provenance
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = fields.mesh
    data = _cell_data(fields)
    X, Y = mesh.cell_center_grid()

    columns = ["x", "y", "u", "v", "speed", "pressure", "vorticity"]
    stacked = np.column_stack(
        [X.ravel(), Y.ravel()] + [data[c].ravel() for c in columns[2:]]
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in stacked:
            writer.writerow([f"{val:.9g}" for val in row])
    write_sidecar(path, record)
    return path


def export_profile_csv(path: str | Path, columns: dict[str, np.ndarray]) -> Path:
    """Write named 1D columns (a profile or a time series) to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    lengths = {len(np.asarray(columns[n])) for n in names}
    if len(lengths) != 1:
        raise ValueError(f"all columns must have equal length, got {lengths}")

    rows = np.column_stack([np.asarray(columns[n], dtype=float) for n in names])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for row in rows:
            writer.writerow([f"{val:.9g}" for val in row])
    return path


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(fields: FlowField, config: SimulationConfig,
                    path: str | Path, provenance: dict | None = None) -> Path:
    """Save the raw ghosted state plus the configuration for an exact restart.

    The provenance record is stored inside the archive as well as beside it, so
    a checkpoint moved on its own still knows where it came from.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = provenance_record(config) if provenance is None else provenance
    np.savez_compressed(
        path,
        u=fields.u, v=fields.v, p=fields.p,
        t=np.array(fields.t), step=np.array(fields.step),
        config=np.array(config.to_json()),
        provenance=np.array(json.dumps(record)),
    )
    write_sidecar(path, record)
    return path


def checkpoint_provenance(path: str | Path) -> dict | None:
    """Read the provenance embedded in a checkpoint, if it carries one.

    Returns ``None`` for checkpoints written before provenance was recorded, so
    older files stay loadable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    with np.load(path, allow_pickle=False) as npz:
        if "provenance" not in npz.files:
            return None
        return json.loads(str(npz["provenance"]))


def load_checkpoint(path: str | Path) -> tuple[FlowField, SimulationConfig]:
    """Restore a state written by :func:`save_checkpoint`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        config = SimulationConfig.from_dict(json.loads(str(npz["config"])))
        mesh = StructuredMesh.from_config(config)
        fields = FlowField(
            mesh, npz["u"], npz["v"], npz["p"],
            float(npz["t"]), int(npz["step"]),
        )
    for arr, expected, label in (
        (fields.u, mesh.u_shape, "u"),
        (fields.v, mesh.v_shape, "v"),
        (fields.p, mesh.p_shape, "p"),
    ):
        if arr.shape != expected:
            raise ValueError(
                f"checkpoint {label} has shape {arr.shape}, expected {expected}; "
                "the file does not match its stored configuration"
            )
    return fields, config
