"""Command-line plumbing: argument parsing, case wiring, exit codes.

These tests cover the layer between the CLI and the case modules -- the part
that decides *which* configuration the solver ends up with.  The numerics
themselves are covered elsewhere; what matters here is that a flag reaches the
place it claims to, and that a flag which cannot apply says so instead of being
quietly dropped.
"""

import numpy as np
import pytest

from pycfd.cases import OUTLET_KINDS, available_cases, load_case, override_outlet
from pycfd.config import BCKind, BCSpec
from pycfd.main import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_VALIDATION_FAILED,
    build_parser,
    main,
)


def external_walls(p_ref: float = 0.0) -> dict[str, BCSpec]:
    """Inlet / outlet / slip-sides, the configuration that has an outflow."""
    return {
        "left": BCSpec(BCKind.INLET, velocity=1.0),
        "right": BCSpec(BCKind.PRESSURE_OUTLET, p_ref=p_ref),
        "bottom": BCSpec(BCKind.SYMMETRY),
        "top": BCSpec(BCKind.SYMMETRY),
    }


def closed_walls() -> dict[str, BCSpec]:
    return {w: BCSpec(BCKind.NO_SLIP) for w in ("left", "right", "bottom", "top")}


# --------------------------------------------------------------------------- #
# override_outlet
# --------------------------------------------------------------------------- #
def test_no_override_leaves_the_config_untouched():
    walls = external_walls()
    assert override_outlet(walls) is walls


def test_override_retypes_only_the_outflow_wall():
    walls = override_outlet(external_walls(), "outlet")
    assert walls["right"].kind is BCKind.OUTLET
    # Everything else is preserved exactly.
    assert walls["left"].kind is BCKind.INLET
    assert walls["left"].velocity == 1.0
    assert walls["bottom"].kind is BCKind.SYMMETRY


def test_override_can_switch_a_velocity_outlet_to_a_pressure_outlet():
    walls = dict(external_walls())
    walls["right"] = BCSpec(BCKind.OUTLET)
    updated = override_outlet(walls, "pressure_outlet", 1.75)
    assert updated["right"].kind is BCKind.PRESSURE_OUTLET
    assert updated["right"].p_ref == 1.75


def test_p_ref_alone_keeps_the_existing_outlet_kind():
    walls = override_outlet(external_walls(p_ref=0.0), p_ref=3.5)
    assert walls["right"].kind is BCKind.PRESSURE_OUTLET
    assert walls["right"].p_ref == 3.5


def test_p_ref_on_a_velocity_outlet_warns(caplog):
    """It cannot take effect, so it must not pass silently."""
    with caplog.at_level("WARNING"):
        walls = override_outlet(external_walls(), "outlet", p_ref=2.0)
    assert walls["right"].kind is BCKind.OUTLET
    assert "ignored" in caplog.text.lower()


def test_override_rejects_a_case_with_no_outflow():
    with pytest.raises(ValueError, match="no outflow boundary"):
        override_outlet(closed_walls(), "outlet")


def test_override_rejects_a_non_outlet_kind():
    with pytest.raises(ValueError, match="outlet_type must be one of"):
        override_outlet(external_walls(), "inlet")


def test_outlet_kinds_are_exactly_the_two_outflow_conditions():
    assert set(OUTLET_KINDS) == {BCKind.OUTLET, BCKind.PRESSURE_OUTLET}


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def test_parser_accepts_both_outlet_flags():
    args = build_parser().parse_args(
        ["--case", "cylinder", "--outlet-type", "pressure_outlet", "--p-ref", "2.5"]
    )
    assert args.outlet_type == "pressure_outlet"
    assert args.p_ref == 2.5


def test_parser_defaults_leave_the_case_choice_alone():
    args = build_parser().parse_args(["--case", "cylinder"])
    assert args.outlet_type is None and args.p_ref is None


def test_parser_rejects_an_unknown_outlet_type():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--outlet-type", "wide_open"])


# --------------------------------------------------------------------------- #
# Case wiring
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", ["cylinder", "channel"])
def test_cases_with_an_outflow_accept_the_flags(case):
    import inspect

    params = inspect.signature(load_case(case).run).parameters
    assert "outlet_type" in params and "p_ref" in params


@pytest.mark.parametrize("case", ["cavity", "taylor_green"])
def test_cases_without_an_outflow_do_not_advertise_the_flags(case):
    import inspect

    assert "outlet_type" not in inspect.signature(load_case(case).run).parameters


@pytest.mark.parametrize("outlet_type,expected", [
    (None, BCKind.PRESSURE_OUTLET),          # the cylinder case's own choice
    ("outlet", BCKind.OUTLET),
    ("pressure_outlet", BCKind.PRESSURE_OUTLET),
])
def test_cylinder_honours_the_requested_outlet(outlet_type, expected):
    sim = load_case("cylinder").build(re=20, nx=48, ny=24, outlet_type=outlet_type)
    assert sim.config.boundary_config["right"].kind is expected


def test_cylinder_honours_a_non_zero_p_ref():
    sim = load_case("cylinder").build(re=20, nx=48, ny=24, p_ref=1.25)
    assert sim.config.boundary_config["right"].p_ref == 1.25
    assert sim.solver.dirichlet_pressure == {"right": 1.25}


def test_periodic_channel_rejects_the_flags():
    with pytest.raises(ValueError, match="no outflow boundary"):
        load_case("channel").build(re=10, nx=32, ny=32, mode="periodic",
                                   outlet_type="outlet")


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def _argv(tmp_path, *extra):
    return ["--case", "cylinder", "--re", "20", "--nx", "48", "--ny", "24",
            "--t-end", "0.2", "--no-plots", "-q",
            "--outdir", str(tmp_path), *extra]


@pytest.mark.parametrize("extra", [
    [],
    ["--outlet-type", "outlet"],
    ["--outlet-type", "pressure_outlet", "--p-ref", "2.0"],
])
def test_cli_runs_with_each_outlet_setting(tmp_path, extra):
    assert main(_argv(tmp_path, *extra)) == EXIT_OK


def test_cli_reports_the_anchor_it_applied(tmp_path, capsys):
    main(_argv(tmp_path, "--outlet-type", "pressure_outlet", "--p-ref", "2.0"))
    out = capsys.readouterr().out
    assert "outlet_p_ref" in out and "2" in out
    assert "outlet_p_deviation" in out


def test_cli_omits_the_anchor_for_a_velocity_outlet(tmp_path, capsys):
    main(_argv(tmp_path, "--outlet-type", "outlet"))
    assert "outlet_p_ref" not in capsys.readouterr().out


@pytest.mark.parametrize("case", ["cavity", "taylor_green"])
def test_cli_refuses_the_flags_on_a_case_without_an_outflow(tmp_path, case, capsys):
    code = main(["--case", case, "--nx", "24", "--ny", "24", "--t-end", "0.02",
                 "--no-plots", "-q", "--outdir", str(tmp_path),
                 "--outlet-type", "outlet"])
    assert code == EXIT_ERROR
    assert "no outflow boundary" in capsys.readouterr().err


def test_cli_error_names_the_cases_that_do_support_it(tmp_path, capsys):
    main(["--case", "cavity", "--nx", "24", "--ny", "24", "--t-end", "0.02",
          "--no-plots", "-q", "--outdir", str(tmp_path), "--p-ref", "1.0"])
    err = capsys.readouterr().err
    assert "cylinder" in err and "channel" in err


def test_switching_the_outlet_does_not_change_the_flow(tmp_path):
    """The outlet fixes the pressure datum, not the physics.

    Drag must be insensitive to the choice; only the level of the reported
    pressure field moves.
    """
    forces, means = [], []
    for outlet_type in ("outlet", "pressure_outlet"):
        sim = load_case("cylinder").build(re=20, nx=128, ny=64,
                                          outlet_type=outlet_type)
        sim.run(t_end=6.0)
        forces.append(sim.force_coefficients()[0])
        means.append(float(sim.fields.p_phys.mean()))

    assert forces[0] == pytest.approx(forces[1], rel=2e-3)
    # The velocity outlet has its mean removed by construction; the pressure
    # outlet instead pins the face, so the mean is free to sit elsewhere.
    assert abs(means[0]) < 1e-12


# --------------------------------------------------------------------------- #
# Domain size, naming and the turbulence switch
# --------------------------------------------------------------------------- #
def test_domain_flags_change_the_blockage_ratio():
    tall = load_case("cylinder").build(re=20, nx=96, ny=48,
                                       domain_length=16.0, domain_height=16.0)
    short = load_case("cylinder").build(re=20, nx=96, ny=48,
                                        domain_length=16.0, domain_height=8.0)
    assert tall.config.ly == 16.0 and short.config.ly == 8.0
    # Halving the height doubles the blockage the body presents.
    assert (short.obstacle.characteristic_length / short.config.ly
            == pytest.approx(2 * tall.obstacle.characteristic_length / tall.config.ly))


def test_domain_flags_reposition_the_body_consistently():
    """The obstacle must stay inside the domain it was told to use."""
    sim = load_case("cylinder").build(re=20, nx=96, ny=48,
                                      domain_length=24.0, domain_height=12.0)
    assert sim.mesh.lx == 24.0 and sim.mesh.ly == 12.0
    assert sim.obstacle.mask.shape == (96, 48)
    assert sim.obstacle.mask.any()


# --------------------------------------------------------------------------- #
# Reference length and flight conditions
# --------------------------------------------------------------------------- #
def test_parser_accepts_the_flow_condition_flags():
    args = build_parser().parse_args(
        ["--case", "cylinder", "--l-ref", "2.6", "--wind-speed", "70",
         "--altitude", "3000"]
    )
    assert args.l_ref == 2.6 and args.wind_speed == 70.0 and args.altitude == 3000.0


def test_flow_condition_flags_default_to_the_bodys_own_convention():
    args = build_parser().parse_args(["--case", "cylinder"])
    assert args.l_ref is None and args.wind_speed is None and args.altitude is None


def test_reference_length_defaults_to_the_bodys_span():
    sim = load_case("cylinder").build(re=20, nx=48, ny=24)
    assert sim.config.l_ref == pytest.approx(sim.obstacle.characteristic_length)


def test_l_ref_overrides_the_reference_length_only():
    """The body does not change size; only what its forces are divided by."""
    sim = load_case("cylinder").build(re=20, nx=48, ny=24, l_ref=2.6)
    assert sim.config.l_ref == 2.6
    # Blockage and grid resolution still describe the real body.
    assert sim.obstacle.characteristic_length == pytest.approx(1.0)


def test_l_ref_reaches_the_force_coefficients():
    """The coefficients divide by the reference length, not by the body's span.

    Comparing two runs would not show this: ``l_ref`` also feeds
    ``nu = u_ref l_ref / re``, so changing it changes the flow as well as the
    normalisation.  The wiring is therefore checked against the definition on a
    single run.
    """
    from pycfd.analysis.postprocess import force_coefficients

    sim = load_case("cylinder").build(re=20, nx=48, ny=24, l_ref=2.6)
    sim.run(t_end=0.4)
    assert sim.obstacle.characteristic_length == pytest.approx(1.0)
    assert sim.force_coefficients() == pytest.approx(
        force_coefficients(sim.solver.body_force_reaction, 1.0, 2.6))


def test_the_reference_length_defines_the_viscosity_too():
    """``nu = u_ref l_ref / re``: Re and the coefficients share one length."""
    sim = load_case("cylinder").build(re=20, nx=48, ny=24, l_ref=2.6)
    assert sim.config.nu == pytest.approx(1.0 * 2.6 / 20.0)


def test_l_ref_must_be_positive():
    with pytest.raises(ValueError, match="l_ref must be positive"):
        load_case("cylinder").build(re=20, nx=48, ny=24, l_ref=0.0)


def test_wind_speed_derives_the_reynolds_number():
    from pycfd.units import atmosphere, reynolds_number

    sim = load_case("cylinder").build(nx=48, ny=24, wind_speed=70.0, l_ref=2.6)
    expected = reynolds_number(70.0, 2.6, atmosphere(0.0).kinematic_viscosity)
    assert sim.config.re == pytest.approx(expected)
    # The solver itself stays non-dimensional: the speed went into Re, not u_ref.
    assert sim.config.u_ref == 1.0


def test_wind_speed_without_l_ref_uses_the_bodys_own_length():
    from pycfd.units import atmosphere, reynolds_number

    sim = load_case("cylinder").build(nx=48, ny=24, wind_speed=70.0)
    expected = reynolds_number(70.0, sim.obstacle.characteristic_length,
                               atmosphere(0.0).kinematic_viscosity)
    assert sim.config.re == pytest.approx(expected)


def test_wind_speed_puts_the_real_viscosity_into_the_solver():
    """The whole chain in one identity.

    ``nu_solver = u_ref l_ref / re`` and ``re = V l_ref / nu_air`` together give
    ``nu_solver = nu_air / V``, independent of which length convention was
    chosen.  If any link were formed with a different length the reference
    length would fail to cancel and this would not hold.
    """
    from pycfd.units import atmosphere

    nu_air = atmosphere(3000.0).kinematic_viscosity
    for l_ref in (None, 2.6, 11.5):
        sim = load_case("cylinder").build(nx=48, ny=24, wind_speed=70.0,
                                          altitude=3000.0, l_ref=l_ref)
        assert sim.config.nu == pytest.approx(nu_air / 70.0, rel=1e-12)


def test_altitude_thins_the_air_and_lowers_the_reynolds_number():
    low = load_case("cylinder").build(nx=48, ny=24, wind_speed=70.0)
    high = load_case("cylinder").build(nx=48, ny=24, wind_speed=70.0, altitude=10000.0)
    assert high.config.re < low.config.re


def test_re_and_wind_speed_together_are_refused():
    """Two answers to the same question is a mistake, not a preference."""
    with pytest.raises(ValueError, match="both set the Reynolds number"):
        load_case("cylinder").build(nx=48, ny=24, re=100.0, wind_speed=70.0)


def test_altitude_without_a_wind_speed_is_refused():
    with pytest.raises(ValueError, match="pass --wind-speed as well"):
        load_case("cylinder").build(nx=48, ny=24, altitude=3000.0)


def test_a_transonic_wind_speed_warns_that_the_equations_do_not_apply(caplog):
    with caplog.at_level("WARNING"):
        load_case("cylinder").build(nx=48, ny=24, wind_speed=272.0)
    assert "incompressible limit" in caplog.text


def test_cli_reports_the_flight_conditions_it_used(tmp_path, capsys):
    code = main(["--case", "cylinder", "--nx", "48", "--ny", "24", "--t-end", "0.1",
                 "--no-plots", "-q", "--outdir", str(tmp_path),
                 "--wind-speed", "70", "--altitude", "3000", "--l-ref", "2.6"])
    # A 0.1 s run is far too short for the shedding check to pass; what is under
    # test here is the report, not the physics.
    assert code in (EXIT_OK, EXIT_VALIDATION_FAILED)
    out = capsys.readouterr().out
    for key in ("wind_speed_m_s", "altitude_m", "mach", "reference_length"):
        assert key in out


def test_cli_refuses_flight_conditions_on_a_case_without_a_body(tmp_path, capsys):
    code = main(["--case", "cavity", "--nx", "24", "--ny", "24", "--t-end", "0.02",
                 "--no-plots", "-q", "--outdir", str(tmp_path), "--wind-speed", "70"])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "Reynolds number against" in err and "cylinder" in err


def test_rescale_to_builds_a_scaling_from_the_altitude():
    from pycfd.main import export_scaling
    from pycfd.units import atmosphere

    args = build_parser().parse_args(["--rescale-to", "70", "--altitude", "3000"])
    scaling = export_scaling(args)
    assert scaling.speed == 70.0
    assert scaling.density == pytest.approx(atmosphere(3000.0).density)


def test_rescale_to_defaults_to_sea_level():
    from pycfd.main import export_scaling
    from pycfd.units import atmosphere

    args = build_parser().parse_args(["--rescale-to", "70"])
    assert export_scaling(args).density == pytest.approx(atmosphere(0.0).density)


def test_no_rescaling_asked_for_means_no_scaling():
    from pycfd.main import export_scaling

    assert export_scaling(build_parser().parse_args([])) is None


def test_altitude_reaches_the_rescaling_on_a_case_with_no_body(tmp_path):
    """--altitude describes the air, so it must not need a Reynolds number.

    It is refused by a case that has no body to size one against, which is why
    it is withheld from the case unless a wind speed came with it.
    """
    from pycfd.main import case_specific_kwargs

    args = build_parser().parse_args(
        ["--case", "cavity", "--rescale-to", "70", "--altitude", "3000"]
    )
    assert "altitude" not in case_specific_kwargs(args)

    code = main(["--case", "cavity", "--nx", "16", "--ny", "16", "--t-end", "0.02",
                 "--no-plots", "-q", "--outdir", str(tmp_path),
                 "--rescale-to", "70", "--altitude", "3000", "--export-csv"])
    # Two solver steps miss the Ghia tolerance; the export is what is under test.
    assert code in (EXIT_OK, EXIT_VALIDATION_FAILED)
    header = (tmp_path / "cavity_Re100_SI.csv").read_text().splitlines()[0]
    assert header.startswith("x_m,y_m,u_m_s")


def test_an_altitude_that_converts_nothing_is_refused(tmp_path, capsys):
    code = main(["--case", "cavity", "--nx", "16", "--ny", "16", "--t-end", "0.02",
                 "--no-plots", "-q", "--outdir", str(tmp_path), "--altitude", "3000"])
    assert code == EXIT_ERROR
    assert "--rescale-to" in capsys.readouterr().err


def test_rescaling_is_refused_before_the_run_not_after(tmp_path, capsys):
    """A long run must not finish only to discover its export was invalid."""
    code = main(["--case", "cavity", "--nx", "16", "--ny", "16", "--t-end", "1e9",
                 "--no-plots", "-q", "--outdir", str(tmp_path), "--altitude", "3000"])
    assert code == EXIT_ERROR
    assert not list(tmp_path.glob("*"))


@pytest.mark.parametrize("case", ["cavity", "channel", "taylor_green"])
def test_flow_condition_flags_are_refused_where_they_do_not_apply(case):
    import inspect

    params = inspect.signature(load_case(case).build).parameters
    assert not {"l_ref", "wind_speed", "altitude"} & set(params)


@pytest.mark.parametrize("case", ["cavity", "channel", "taylor_green"])
def test_domain_flags_are_refused_where_they_do_not_apply(case):
    import inspect

    assert "domain_length" not in inspect.signature(load_case(case).build).parameters


def test_cli_domain_error_names_the_supporting_case(tmp_path, capsys):
    code = main(["--case", "cavity", "--nx", "24", "--ny", "24", "--t-end", "0.02",
                 "--no-plots", "-q", "--outdir", str(tmp_path), "--domain-length", "10"])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "configurable domain size" in err and "cylinder" in err
    assert "no a " not in err            # the article must not collide


@pytest.mark.parametrize("case", ["cavity", "channel", "cylinder", "taylor_green"])
def test_name_is_overridable_on_every_case(case):
    sim = load_case(case).build(name="my_run")
    assert sim.config.name == "my_run"


@pytest.mark.parametrize("case,expected", [
    ("cavity", "cavity_Re100"),
    ("cylinder", "cylinder_Re100"),
    ("channel", "channel_periodic_Re10"),
])
def test_name_defaults_are_unchanged(case, expected):
    assert load_case(case).build().config.name == expected


def test_name_drives_the_files_the_cli_writes(tmp_path):
    main(["--case", "cylinder", "--re", "20", "--nx", "48", "--ny", "24",
          "--t-end", "0.1", "--no-plots", "-q", "--outdir", str(tmp_path),
          "--name", "shield_120kmh", "--export-csv", "--checkpoint"])
    written = {f.name for f in tmp_path.iterdir()}
    assert "shield_120kmh.csv" in written
    assert "shield_120kmh.npz" in written


def test_filenames_are_unchanged_without_name(tmp_path):
    main(["--case", "cylinder", "--re", "20", "--nx", "48", "--ny", "24",
          "--t-end", "0.1", "--no-plots", "-q", "--outdir", str(tmp_path),
          "--export-csv"])
    assert "cylinder_Re20.csv" in {f.name for f in tmp_path.iterdir()}


@pytest.mark.parametrize("argv,expected", [
    (["--les"], True),
    (["--no-les"], False),
    ([], None),
])
def test_les_switch_parses_three_ways(argv, expected):
    assert build_parser().parse_args(argv).use_les is expected


def test_les_and_no_les_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--les", "--no-les"])


@pytest.mark.parametrize("flag,expected", [("--les", True), ("--no-les", False)])
def test_les_switch_reaches_the_solver(tmp_path, flag, expected):
    args = build_parser().parse_args(
        ["--case", "cylinder", "--nx", "48", "--ny", "24", flag]
    )
    from pycfd.main import case_kwargs

    assert case_kwargs(args)["use_les"] is expected
    sim = load_case("cylinder").build(re=20, nx=48, ny=24, use_les=expected)
    assert sim.config.use_les is expected
    assert (sim.solver.turbulence is not None) is expected


def test_benchmarks_stay_laminar_unless_asked():
    """The package default may be True; a benchmark must not inherit it."""
    for case in ("cavity", "channel", "cylinder", "taylor_green"):
        assert load_case(case).build().config.use_les is False
