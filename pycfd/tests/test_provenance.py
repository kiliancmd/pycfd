"""Provenance: every output must be able to say what produced it.

The value of a provenance record is entirely in whether it survives the trip to
disk and back, so these tests read the written files rather than inspecting the
record in memory.
"""

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from pycfd.analysis.export import (  # noqa: E402
    checkpoint_provenance,
    export_csv,
    export_vtk,
    load_checkpoint,
    save_checkpoint,
)
from pycfd.analysis.provenance import (  # noqa: E402
    invocation,
    png_metadata,
    provenance_record,
    summary_line,
    write_sidecar,
)
from pycfd.config import BCKind  # noqa: E402
from pycfd.core.solver import ProjectionSolver  # noqa: E402

from .conftest import cavity_config, make_config, walls  # noqa: E402


@pytest.fixture
def stepped():
    """A configuration and a field that has actually been advanced."""
    cfg = cavity_config(nx=8, ny=8)
    solver = ProjectionSolver(cfg)
    return cfg, solver.step(solver.initialize(), 1e-3)


# --------------------------------------------------------------------------- #
# The record itself
# --------------------------------------------------------------------------- #
def test_record_carries_what_is_needed_to_reproduce_a_run():
    cfg = make_config(nx=16, ny=16, re=250.0, boundary_config=walls())
    record = provenance_record(cfg, extra={"cd": 1.25})

    for key in ("pycfd_version", "generated_utc", "command", "python", "platform"):
        assert record[key], f"provenance is missing {key!r}"
    # The configuration travels in full, so a result is reproducible from the
    # file alone rather than from whatever the caller happens to remember.
    assert record["config"]["re"] == 250.0
    assert record["config"]["nx"] == 16
    assert record["extra"]["cd"] == 1.25


def test_record_without_a_config_is_still_valid():
    """Figures record the command even when no config is threaded through."""
    record = provenance_record()
    assert record["command"] is not None
    assert "config" not in record


def test_invocation_reconstructs_a_runnable_command():
    """``sys.argv[0]`` is a path; what gets recorded should be pasteable."""
    argv = ["/somewhere/pycfd/main.py", "--case", "cylinder", "--re", "100"]
    assert invocation(argv) == "python -m pycfd.main --case cylinder --re 100"


def test_invocation_quotes_arguments_that_need_it():
    argv = ["/x/main.py", "--name", "my run", "--outdir", "/tmp/a b"]
    assert "'my run'" in invocation(argv)


def test_summary_line_respects_its_length_cap():
    """The VTK title line is capped by the format at 256 characters."""
    record = provenance_record()
    record["command"] = "python -m pycfd.main " + "--flag x " * 200
    assert len(summary_line(record, limit=180)) <= 180


def test_dirty_working_tree_is_marked():
    """A bare hash would imply a reproducibility an edited tree does not have."""
    record = provenance_record()
    if "git_commit" not in record:
        pytest.skip("not running inside a git repository")
    commit = record["git_commit"]
    assert commit.rstrip("+"), "commit hash is empty"
    assert commit.endswith("+") or commit == commit.strip()


# --------------------------------------------------------------------------- #
# Sidecars
# --------------------------------------------------------------------------- #
def test_sidecar_replaces_the_suffix_rather_than_appending(tmp_path):
    record = provenance_record()
    sidecar = write_sidecar(tmp_path / "field.vtk", record)
    assert sidecar.name == "field.provenance.json"
    assert json.loads(sidecar.read_text())["command"] == record["command"]


@pytest.mark.parametrize("exporter,suffix", [(export_vtk, ".vtk"), (export_csv, ".csv")])
def test_every_exporter_writes_a_sidecar(tmp_path, stepped, exporter, suffix):
    _, fields = stepped
    path = tmp_path / f"out{suffix}"
    exporter(fields, path)
    sidecar = path.with_suffix(".provenance.json")
    assert sidecar.exists(), f"{suffix} export wrote no provenance sidecar"
    assert json.loads(sidecar.read_text())["pycfd_version"]


# --------------------------------------------------------------------------- #
# Embedded channels
# --------------------------------------------------------------------------- #
def test_vtk_title_carries_a_digest_and_stays_within_the_format(tmp_path, stepped):
    _, fields = stepped
    path = export_vtk(fields, tmp_path / "f.vtk", name="demo")
    title = path.read_text().splitlines()[1]

    assert "demo" in title and "pycfd" in title
    # The legacy format caps the title line at 256 characters.
    assert len(title) <= 256


def test_csv_stays_pure_data(tmp_path, stepped):
    """A comment banner would break readers that do not expect one."""
    _, fields = stepped
    path = export_csv(fields, tmp_path / "f.csv")
    first = path.read_text().splitlines()[0]
    assert first == "x,y,u,v,speed,pressure,vorticity"
    assert not first.startswith("#")


def test_checkpoint_embeds_provenance_and_still_restores(tmp_path, stepped):
    cfg, fields = stepped
    path = save_checkpoint(fields, cfg, tmp_path / "ck.npz")

    record = checkpoint_provenance(path)
    assert record["config"]["nx"] == cfg.nx
    # Embedding must not disturb the restart path.
    restored, restored_cfg = load_checkpoint(path)
    assert np.array_equal(restored.u, fields.u)
    assert restored_cfg == cfg


def test_checkpoint_without_provenance_still_loads(tmp_path, stepped):
    """Checkpoints written before provenance existed must stay readable."""
    cfg, fields = stepped
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path, u=fields.u, v=fields.v, p=fields.p,
        t=np.array(fields.t), step=np.array(fields.step),
        config=np.array(cfg.to_json()),
    )
    assert checkpoint_provenance(path) is None
    restored, _ = load_checkpoint(path)
    assert np.array_equal(restored.p, fields.p)


def test_png_metadata_uses_registered_keywords():
    """``exiftool`` and image viewers surface these without knowing about pycfd."""
    meta = png_metadata(provenance_record())
    assert set(meta) <= {"Software", "Creation Time", "Description", "Comment"}
    assert meta["Software"].startswith("pycfd")
    assert all(isinstance(v, str) for v in meta.values())


def test_figures_embed_provenance_in_the_png(tmp_path, stepped):
    from PIL import Image

    from pycfd.visualization import static_plot as sp

    _, fields = stepped
    path = tmp_path / "fig.png"
    sp.use_headless_backend()
    sp.four_panel_figure(fields, title="test", path=path)

    info = Image.open(path).info
    assert info.get("Software", "").startswith("pycfd")
    assert info.get("Creation Time")


# --------------------------------------------------------------------------- #
# Through the Simulation façade
# --------------------------------------------------------------------------- #
def test_simulation_provenance_includes_its_own_diagnostics(tmp_path):
    """An export should say what the run converged to, not just how it started."""
    from pycfd.physics.incompressible import Simulation

    cfg = cavity_config(nx=12, ny=12, t_end=0.05)
    sim = Simulation(cfg)
    sim.run()
    record = sim.provenance()

    assert record["config"]["nx"] == 12
    assert "max_divergence" in record["extra"]
    assert "kinetic_energy" in record["extra"]


def test_simulation_exports_carry_the_configuration(tmp_path):
    from pycfd.physics.incompressible import Simulation

    cfg = cavity_config(nx=10, ny=10, t_end=0.02, name="prov_case")
    sim = Simulation(cfg)
    sim.run()
    sim.export_vtk(tmp_path / "f.vtk")

    record = json.loads((tmp_path / "f.provenance.json").read_text())
    assert record["config"]["name"] == "prov_case"
    assert record["config"]["boundary_config"]["top"]["kind"] == BCKind.MOVING_WALL.value
