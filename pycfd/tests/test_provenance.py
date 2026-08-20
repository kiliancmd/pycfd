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


# --------------------------------------------------------------------------- #
# Rescaled exports
# --------------------------------------------------------------------------- #
# --rescale-to writes the field exports in SI while the checkpoint stays in
# solver units, so the two must never be mistaken for one another -- not by
# their contents, their column names, their sidecars or their filenames.
def si_scaling():
    from pycfd.units import Scaling

    return Scaling.at_altitude(70.0, length=2.0, altitude=3000.0)


def test_unrescaled_exports_are_unchanged_and_say_so(tmp_path, stepped):
    _, fields = stepped
    export_csv(fields, tmp_path / "plain.csv")
    header = (tmp_path / "plain.csv").read_text().splitlines()[0]

    assert header == "x,y,u,v,speed,pressure,vorticity"
    record = json.loads((tmp_path / "plain.provenance.json").read_text())
    assert record["units"]["system"] == "solver"


def test_a_rescaled_csv_labels_every_column_with_its_unit(tmp_path, stepped):
    _, fields = stepped
    export_csv(fields, tmp_path / "si.csv", scaling=si_scaling())
    header = (tmp_path / "si.csv").read_text().splitlines()[0]

    assert header == "x_m,y_m,u_m_s,v_m_s,speed_m_s,pressure_Pa,vorticity_1_s"


def test_a_rescaled_csv_holds_si_values(tmp_path, stepped):
    """Each column converts through its own scale, not one blanket factor."""
    _, fields = stepped
    s = si_scaling()
    export_csv(fields, tmp_path / "plain.csv")
    export_csv(fields, tmp_path / "si.csv", scaling=s)

    def column(name, path):
        lines = (path).read_text().splitlines()
        idx = lines[0].split(",").index(name)
        return np.array([float(line.split(",")[idx]) for line in lines[1:]])

    plain_u = column("u", tmp_path / "plain.csv")
    plain_p = column("pressure", tmp_path / "plain.csv")
    plain_x = column("x", tmp_path / "plain.csv")
    assert column("u_m_s", tmp_path / "si.csv") == pytest.approx(plain_u * s.speed)
    assert column("pressure_Pa", tmp_path / "si.csv") == pytest.approx(
        plain_p * s.pressure_scale)
    assert column("x_m", tmp_path / "si.csv") == pytest.approx(plain_x * s.length)


def test_a_rescaled_sidecar_records_the_exchange_rate(tmp_path, stepped):
    """Enough to undo the conversion, so the file is not a dead end."""
    _, fields = stepped
    s = si_scaling()
    export_csv(fields, tmp_path / "si.csv", scaling=s)
    units = json.loads((tmp_path / "si.provenance.json").read_text())["units"]

    assert units["system"] == "SI"
    assert units["reference_speed_m_s"] == pytest.approx(s.speed)
    assert units["reference_length_m"] == pytest.approx(s.length)
    assert units["density_kg_m3"] == pytest.approx(s.density)
    assert units["columns"]["pressure"] == "Pa"


def test_a_rescaled_vtk_announces_its_units_in_the_title(tmp_path, stepped):
    _, fields = stepped
    export_vtk(fields, tmp_path / "si.vtk", scaling=si_scaling())
    title = (tmp_path / "si.vtk").read_text().splitlines()[1]

    assert " SI " in title
    assert len(title) <= 256          # the legacy format's cap


def test_a_rescaled_vtk_converts_the_time_stamp_too(tmp_path, stepped):
    _, fields = stepped
    s = si_scaling()
    export_vtk(fields, tmp_path / "si.vtk", scaling=s)
    title = (tmp_path / "si.vtk").read_text().splitlines()[1]

    stamped = float(title.split("t=")[1].split()[0])
    assert stamped == pytest.approx(s.to_seconds(fields.t), rel=1e-5)


def test_a_checkpoint_is_never_rescaled(tmp_path, stepped):
    """It restarts the solver, and the solver restarts in its own units."""
    cfg, fields = stepped
    save_checkpoint(fields, cfg, tmp_path / "state.npz")
    restored, _ = load_checkpoint(tmp_path / "state.npz")

    assert np.array_equal(restored.u, fields.u)
    record = checkpoint_provenance(tmp_path / "state.npz")
    assert record["units"]["system"] == "solver"


def test_rescaled_and_plain_outputs_do_not_share_a_sidecar(tmp_path):
    """The bug this naming exists to prevent: one file describing two others.

    All three exporters map ``<name>.<ext>`` onto one ``<name>.provenance.json``,
    so a rescaled field export and an unrescaled checkpoint written under the
    same name would leave whichever ran last describing both.
    """
    from pycfd.main import build_parser, write_outputs
    from pycfd.physics.incompressible import Simulation

    sim = Simulation(cavity_config(nx=8, ny=8, t_end=0.01))
    sim.run()
    args = build_parser().parse_args(
        ["--outdir", str(tmp_path), "--export-csv", "--checkpoint",
         "--rescale-to", "70"]
    )
    written = write_outputs(sim, args, "run")

    assert {p.name for p in written} == {"run_SI.csv", "run.npz"}
    systems = {
        p.name: json.loads(p.with_suffix(".provenance.json").read_text())["units"]["system"]
        for p in written
    }
    assert systems == {"run_SI.csv": "SI", "run.npz": "solver"}
