"""Several static bodies in one domain, each with its own force.

A multi-body run that only reported a total would be a differently-shaped
single obstacle.  The point is the *split* -- what the trailing body in a wake
feels, as against the one making the wake -- so most of what is tested here is
that the split is well defined and that it adds up: the per-body forces are
collected from disjoint sets of faces whose union is the set the total is
collected from, which makes exact additivity a property the code either has or
does not.

The geometries that would break that -- bodies overlapping, or touching with no
fluid between them -- are refused on construction rather than reported as
numbers that look per-body and are not.
"""

import numpy as np
import pytest

from pycfd.cases import cylinder_flow
from pycfd.core.mesh import StructuredMesh
from pycfd.geometry.obstacles import (
    Obstacle,
    ObstacleContactError,
    ObstacleGroup,
    as_obstacle_group,
    circle_mask,
    rectangle_mask,
)
from pycfd.cases.cylinder_flow import file_tag
from pycfd.main import (
    EXIT_ERROR,
    build_parser,
    check_orphan_geometry_flags,
    geometry_spec,
    main,
    parse_placement,
)

from .conftest import DIVERGENCE_TOL


def mesh(nx: int = 96, ny: int = 48, lx: float = 24.0, ly: float = 12.0):
    return StructuredMesh(nx, ny, lx, ly)


def tandem(m=None, gap: float = 4.0):
    """Two equal cylinders on the centreline, ``gap`` diameters apart."""
    m = m if m is not None else mesh()
    return (circle_mask(m, (6.0, 6.0), 0.5, name="lead"),
            circle_mask(m, (6.0 + gap, 6.0), 0.5, name="trail"))


# --------------------------------------------------------------------------- #
# The group is a body as far as everything else is concerned
# --------------------------------------------------------------------------- #
def test_the_union_is_every_body_and_nothing_else():
    m = mesh()
    a, b = tandem(m)
    group = ObstacleGroup((a, b))
    assert np.array_equal(group.mask, a.mask | b.mask)
    # Disjoint bodies, so the cell counts add exactly.
    assert group.mask.sum() == a.mask.sum() + b.mask.sum()


def test_the_solid_fractions_add_because_no_cell_is_shared():
    m = mesh()
    a, b = tandem(m)
    group = ObstacleGroup((a, b))
    assert group.fraction == pytest.approx(a.fraction + b.fraction)
    assert group.area == pytest.approx(a.area + b.area)


def test_the_reference_length_is_the_largest_body():
    m = mesh()
    small = circle_mask(m, (5.0, 6.0), 0.5, name="small")
    large = circle_mask(m, (12.0, 6.0), 1.5, name="large")
    group = ObstacleGroup((small, large))
    assert group.characteristic_length == large.characteristic_length


def test_a_group_names_its_members():
    a, b = tandem()
    assert ObstacleGroup((a, b)).name == "lead + trail"


def test_a_crowd_is_counted_rather_than_listed():
    m = mesh()
    bodies = tuple(
        circle_mask(m, (3.0 + 3.0 * k, 6.0), 0.5, name=f"b{k}") for k in range(4)
    )
    assert ObstacleGroup(bodies).name == "4 bodies"


# --------------------------------------------------------------------------- #
# Geometries that would make a per-body force meaningless
# --------------------------------------------------------------------------- #
def test_overlapping_bodies_are_refused_and_both_are_named():
    m = mesh()
    a = circle_mask(m, (6.0, 6.0), 1.0, name="lead")
    b = circle_mask(m, (6.5, 6.0), 1.0, name="clash")
    with pytest.raises(ObstacleContactError, match="'lead' and 'clash' overlap"):
        ObstacleGroup((a, b))


def test_touching_bodies_are_refused_because_no_fluid_separates_them():
    m = mesh(64, 32, 16.0, 8.0)
    fore = rectangle_mask(m, (4.0, 3.0), (6.0, 5.0), name="fore")
    aft = rectangle_mask(m, (6.0, 3.0), (8.0, 5.0), name="aft")
    with pytest.raises(ObstacleContactError, match="touch along"):
        ObstacleGroup((fore, aft))


def test_bodies_meeting_at_a_corner_are_allowed():
    """A corner is not a face, so nothing about the bookkeeping is ambiguous."""
    m = mesh(64, 32, 16.0, 8.0)
    lower = rectangle_mask(m, (4.0, 3.0), (5.0, 4.0), name="lower")
    upper = rectangle_mask(m, (5.0, 4.0), (6.0, 5.0), name="upper")
    assert len(ObstacleGroup((lower, upper))) == 2


def test_one_fluid_cell_between_them_is_enough():
    m = mesh(64, 32, 16.0, 8.0)
    fore = rectangle_mask(m, (4.0, 3.0), (6.0, 5.0), name="fore")
    aft = rectangle_mask(m, (6.5, 3.0), (8.0, 5.0), name="aft")
    assert len(ObstacleGroup((fore, aft))) == 2


def test_bodies_from_different_meshes_are_refused():
    a = circle_mask(mesh(96, 48), (6.0, 6.0), 0.5, name="a")
    b = circle_mask(mesh(64, 32), (6.0, 6.0), 0.5, name="b")
    with pytest.raises(ValueError, match="same mesh"):
        ObstacleGroup((a, b))


def test_an_empty_group_is_refused():
    with pytest.raises(ValueError, match="at least one body"):
        ObstacleGroup(())


# --------------------------------------------------------------------------- #
# Identical bodies are the expected case, not a mistake
# --------------------------------------------------------------------------- #
def test_repeated_names_are_numbered_so_no_body_is_lost():
    """A formation is loaded from one file, so duplicate names are normal --
    and forces keyed by name would silently collapse onto a single entry."""
    m = mesh()
    a = circle_mask(m, (6.0, 6.0), 0.5, name="jet")
    b = circle_mask(m, (10.0, 6.0), 0.5, name="jet")
    assert ObstacleGroup((a, b)).names == ("jet#1", "jet#2")


def test_a_number_that_would_itself_collide_is_bumped():
    """Landing back on a duplicate would lose a body in exactly the way the
    numbering exists to prevent."""
    m = mesh(128, 64, 32.0, 12.0)
    bodies = (
        circle_mask(m, (4.0, 6.0), 0.5, name="jet"),
        circle_mask(m, (8.0, 6.0), 0.5, name="jet"),
        circle_mask(m, (12.0, 6.0), 0.5, name="jet#1"),
    )
    names = ObstacleGroup(bodies).names
    assert len(set(names)) == 3
    assert "jet#1" in names


def test_a_name_used_once_is_left_alone():
    m = mesh()
    a = circle_mask(m, (6.0, 6.0), 0.5, name="jet")
    b = circle_mask(m, (10.0, 6.0), 0.5, name="probe")
    assert ObstacleGroup((a, b)).names == ("jet", "probe")


def test_renaming_does_not_touch_the_callers_own_objects():
    m = mesh()
    a = circle_mask(m, (6.0, 6.0), 0.5, name="jet")
    b = circle_mask(m, (10.0, 6.0), 0.5, name="jet")
    ObstacleGroup((a, b))
    assert (a.name, b.name) == ("jet", "jet")


# --------------------------------------------------------------------------- #
# Blockage is not any one body's size
# --------------------------------------------------------------------------- #
def test_two_bodies_abreast_block_the_sum_of_their_spans():
    m = mesh()
    a = circle_mask(m, (6.0, 4.0), 0.5, name="a")
    b = circle_mask(m, (6.0, 8.0), 0.5, name="b")
    assert ObstacleGroup((a, b)).blocked_span(m.dy) == pytest.approx(2.0)


def test_two_bodies_in_tandem_block_only_one_span():
    m = mesh()
    a, b = tandem(m)
    assert ObstacleGroup((a, b)).blocked_span(m.dy) == pytest.approx(1.0)


@pytest.mark.parametrize("nx,ny", [(256, 128), (192, 96), (128, 64), (96, 48)])
def test_one_cylinder_blocks_exactly_its_diameter(nx, ny):
    """One measure serves both paths, so it has to reproduce the single-body
    number a benchmark is already pinned to -- exactly, and on every grid."""
    m = mesh(nx, ny, 16.0, 8.0)
    a = circle_mask(m, (4.0, 4.0), 0.5, name="solo")
    assert ObstacleGroup((a,)).blocked_span(m.dy) == pytest.approx(
        a.characteristic_length
    )


def test_a_second_body_can_never_lower_the_blockage():
    """Reporting one body's length as what the group obstructs let a body
    added *behind* another reduce the blockage the walls supposedly feel."""
    m = mesh()
    a, b = tandem(m)
    solo = ObstacleGroup((a,)).blocked_span(m.dy)
    assert ObstacleGroup((a, b)).blocked_span(m.dy) >= solo


# --------------------------------------------------------------------------- #
# Normalising whatever a caller passed
# --------------------------------------------------------------------------- #
def test_as_obstacle_group_accepts_the_three_spellings():
    m = mesh()
    a, b = tandem(m)
    assert as_obstacle_group(None) is None
    assert as_obstacle_group(a).names == ("lead",)
    assert as_obstacle_group([a, b]).names == ("lead", "trail")
    group = ObstacleGroup((a, b))
    assert as_obstacle_group(group) is group


def test_a_sequence_of_non_obstacles_is_refused():
    with pytest.raises(TypeError, match="sequence"):
        as_obstacle_group(["not-an-obstacle"])


# --------------------------------------------------------------------------- #
# The solver's per-body bookkeeping
# --------------------------------------------------------------------------- #
def two_body_sim(t_end: float = 12.0, re: float = 40.0):
    m = mesh()
    a, b = tandem(m)
    sim = cylinder_flow.build(
        re=re, nx=m.nx, ny=m.ny, domain_length=m.lx, domain_height=m.ly,
        obstacle=ObstacleGroup((a, b)), t_end=t_end,
    )
    sim.run(progress=False)
    return sim


def test_each_body_gets_its_own_face_set_and_together_they_are_the_whole():
    m = mesh()
    a, b = tandem(m)
    sim = cylinder_flow.build(
        re=40.0, nx=m.nx, ny=m.ny, domain_length=m.lx, domain_height=m.ly,
        obstacle=ObstacleGroup((a, b)), t_end=0.1,
    )
    s = sim.solver
    (u_a, u_b), (v_a, v_b) = s.u_face_body, s.v_face_body
    # Disjoint: no face is charged to two bodies...
    assert not (u_a & u_b).any()
    assert not (v_a & v_b).any()
    # ...and complete: no face is charged to none.
    assert np.array_equal(u_a | u_b, s.u_face_solid)
    assert np.array_equal(v_a | v_b, s.v_face_solid)


def test_the_per_body_forces_sum_to_the_total():
    """The invariant the whole split rests on. Disjoint face sets whose union
    is the total set means this holds to round-off, not to a tolerance."""
    sim = two_body_sim()
    total_cd, total_cl = sim.force_coefficients()
    per = sim.force_coefficients_by_body()
    assert sum(c[0] for c in per.values()) == pytest.approx(total_cd, abs=1e-12)
    assert sum(c[1] for c in per.values()) == pytest.approx(total_cl, abs=1e-12)


def test_a_body_in_a_wake_feels_less_drag_than_the_body_making_it():
    """Why anyone runs two bodies at once. At four diameters the trailing
    cylinder sits inside the leading one's wake and sees a fraction of its
    drag -- a total would report their average and hide it entirely."""
    per = two_body_sim().force_coefficients_by_body()
    assert per["trail"][0] < 0.5 * per["lead"][0]
    assert per["lead"][0] > 0.0


def test_several_bodies_leave_the_flow_divergence_free():
    """The projection has to stay consistent with a mask made of several
    disconnected pieces, not just one."""
    sim = two_body_sim(t_end=6.0)
    assert sim.solver.max_divergence(sim.fields) < DIVERGENCE_TOL


def test_one_body_reports_itself_and_nothing_else():
    m = mesh()
    solo = circle_mask(m, (6.0, 6.0), 0.5, name="solo")
    sim = cylinder_flow.build(
        re=40.0, nx=m.nx, ny=m.ny, domain_length=m.lx, domain_height=m.ly,
        obstacle=solo, t_end=0.1,
    )
    sim.run(progress=False)
    per = sim.force_coefficients_by_body()
    assert list(per) == ["solo"]
    assert per["solo"] == pytest.approx(sim.force_coefficients())


def test_a_single_body_keeps_the_cheaper_path():
    """Per-body masks buy nothing when there is one body, and the reduction
    they cost would be paid on every substep of every existing run."""
    m = mesh()
    solo = circle_mask(m, (6.0, 6.0), 0.5, name="solo")
    sim = cylinder_flow.build(
        re=40.0, nx=m.nx, ny=m.ny, domain_length=m.lx, domain_height=m.ly,
        obstacle=solo, t_end=0.1,
    )
    assert sim.solver.body_masks == ()
    assert sim.solver.body_force_reactions == ()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("4,3.5", (4.0, 3.5)),
    ("4 3.5", (4.0, 3.5)),
    (" 4 , 3.5 ", (4.0, 3.5)),
])
def test_a_placement_parses(text, expected):
    assert parse_placement(text) == expected


@pytest.mark.parametrize("text", ["4", "4,5,6", "left,4"])
def test_a_bad_placement_is_refused_with_a_reason(text):
    with pytest.raises(ValueError, match="--geometry-at"):
        parse_placement(text)


def test_geometry_repeats_into_one_body_each():
    args = build_parser().parse_args([
        "--case", "cylinder",
        "--geometry", "a.csv", "--geometry-at", "4,4",
        "--geometry", "b.csv", "--geometry-at", "9,4",
    ])
    spec = geometry_spec(args)
    assert spec["geometry"] == ["a.csv", "b.csv"]
    assert spec["geometry_at"] == [(4.0, 4.0), (9.0, 4.0)]


def test_one_scale_applies_to_every_body():
    args = build_parser().parse_args([
        "--case", "cylinder",
        "--geometry", "a.csv", "--geometry-at", "4,4",
        "--geometry", "b.csv", "--geometry-at", "9,4",
        "--geometry-scale", "2.0",
    ])
    assert geometry_spec(args)["geometry_scale"] == [2.0, 2.0]


def test_a_scale_per_body_is_kept_as_given():
    args = build_parser().parse_args([
        "--case", "cylinder",
        "--geometry", "a.csv", "--geometry-at", "4,4",
        "--geometry", "b.csv", "--geometry-at", "9,4",
        "--geometry-scale", "2.0", "--geometry-scale", "3.0",
    ])
    assert geometry_spec(args)["geometry_scale"] == [2.0, 3.0]


def test_a_scale_count_that_matches_neither_is_refused():
    args = build_parser().parse_args([
        "--case", "cylinder",
        "--geometry", "a.csv", "--geometry-at", "4,4",
        "--geometry", "b.csv", "--geometry-at", "9,4",
        "--geometry", "c.csv", "--geometry-at", "14,4",
        "--geometry-scale", "2.0", "--geometry-scale", "3.0",
    ])
    with pytest.raises(ValueError, match="--geometry-scale was given 2"):
        geometry_spec(args)


def test_several_bodies_without_placement_are_refused():
    """Arranging them automatically would be a guess, and a wrong guess here
    does not fail -- it simulates a formation nobody asked for."""
    args = build_parser().parse_args([
        "--case", "cylinder", "--geometry", "a.csv", "--geometry", "b.csv",
    ])
    with pytest.raises(ValueError, match="no --geometry-at"):
        geometry_spec(args)


def test_a_placement_count_that_does_not_match_is_refused():
    args = build_parser().parse_args([
        "--case", "cylinder",
        "--geometry", "a.csv", "--geometry", "b.csv", "--geometry-at", "4,4",
    ])
    with pytest.raises(ValueError, match="--geometry-at was given 1"):
        geometry_spec(args)


def test_a_single_body_still_needs_no_placement():
    args = build_parser().parse_args(["--case", "cylinder", "--geometry", "a.csv"])
    spec = geometry_spec(args)
    assert spec["geometry"] == ["a.csv"]
    assert spec["geometry_at"] == [None]


# --------------------------------------------------------------------------- #
# A group's name has to survive being put in a path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expected", [
    ("shield#1 + shield#2", "shield-1-shield-2"),
    ("cylinder", "cylinder"),
    ("f22_side_profile", "f22_side_profile"),
    ("my body", "my-body"),
    ("###", "body"),
])
def test_a_display_name_folds_into_a_usable_filename(name, expected):
    """A group names its members, which reads well in a report and badly in a
    shell -- `shield#1 + shield#2_Re40.png` has two shell metacharacters and a
    space in it."""
    assert file_tag(name) == expected


@pytest.mark.parametrize("flag,value", [
    ("--geometry-at", "4,4"),
    ("--geometry-scale", "2.0"),
    ("--geometry-rotate", "10"),
])
def test_a_placement_flag_without_a_body_is_refused(flag, value):
    """Dropped in silence, the run that came back would look like the one that
    was asked for."""
    args = build_parser().parse_args(["--case", "cylinder", flag, value])
    with pytest.raises(ValueError, match="no --geometry was given"):
        check_orphan_geometry_flags(args)


def test_the_orphan_check_reaches_the_command_line(capsys):
    code = main(["--case", "cylinder", "--geometry-at", "4,4", "--no-plots"])
    assert code == EXIT_ERROR
    assert "no --geometry was given" in capsys.readouterr().err
