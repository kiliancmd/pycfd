"""Grid-refinement studies on any case, not just the one with an exact solution.

The Taylor--Green vortex can measure a true error because its answer is known.
Every other case can only be refined and watched, and the distinction matters:
fitting an observed order to a sequence that has no reference is how a study
extrapolates past its own asymptotic regime. These tests pin which of the two a
case gets, and that a quantity which is structurally zero is not mistaken for
one that refuses to converge.
"""

import math

import pytest

from pycfd.cases import (
    DEFAULT_RESOLUTIONS,
    MIN_CONVERGENCE_ORDER,
    NOISE_FLOOR,
    SETTLED_TOLERANCE,
    GridStudy,
    available_cases,
    case_aspect,
    grid_study,
    load_case,
    parse_resolutions,
)
from pycfd.main import EXIT_ERROR, EXIT_OK, EXIT_VALIDATION_FAILED, build_parser, main


# --------------------------------------------------------------------------- #
# Parsing the resolution list
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("64,128,256", (64, 128, 256)),
    ("16, 32", (16, 32)),
    ("16 32 64", (16, 32, 64)),          # whitespace is as good as a comma
])
def test_resolutions_parse(text, expected):
    assert parse_resolutions(text) == expected


@pytest.mark.parametrize("text,message", [
    ("64", "at least two resolutions"),
    ("128,64", "must increase"),
    ("64,64", "must increase"),
    ("2,8", "at least 4"),
    ("64,abc", "comma-separated list of integers"),
])
def test_bad_resolutions_are_refused_with_a_reason(text, message):
    with pytest.raises(ValueError, match=message):
        parse_resolutions(text)


# --------------------------------------------------------------------------- #
# Refining preserves the shape of the problem
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case,aspect", [
    ("cavity", 1.0),          # 128x128
    ("channel", 2.0),         # 32x64
    ("cylinder", 0.5),        # 256x128
])
def test_case_aspect_reads_the_cases_own_shape(case, aspect):
    assert case_aspect(case) == pytest.approx(aspect)


def test_a_case_that_derives_its_own_ny_has_no_fixed_aspect():
    """Taylor-Green defaults ``ny`` to ``nx``; nothing to preserve."""
    assert case_aspect("taylor_green") is None


def test_refinement_keeps_the_domain_shape(tmp_path):
    """A 2:1 channel refined into a square would be a different problem."""
    import functools

    seen = []
    module = load_case("channel")
    real_run = module.run

    # wraps keeps the real signature visible, which is what case_aspect reads
    # the domain shape from -- a bare **kwargs spy would erase the thing under
    # test and pass vacuously.
    @functools.wraps(real_run)
    def spy(**kwargs):
        seen.append((kwargs["nx"], kwargs["ny"]))
        return real_run(**kwargs)

    module.run = spy
    try:
        grid_study("channel", (16, 24), re=10, t_end=0.05)
    finally:
        module.run = real_run

    assert seen == [(16, 32), (24, 48)]


# --------------------------------------------------------------------------- #
# Reading a study
# --------------------------------------------------------------------------- #
def study(values, kind="error", resolutions=(16, 32, 64)):
    return GridStudy("test", list(resolutions), {"m": list(values)}, {"m": kind})


def test_changes_are_relative_to_the_coarser_grid():
    assert study([1.0, 0.5, 0.25]).changes("m") == pytest.approx([0.5, 0.5])


def test_a_settled_metric_is_one_whose_finest_pair_agree():
    assert study([1.0, 0.5, 0.5 * (1 + 0.5 * SETTLED_TOLERANCE)]).settled("m")
    assert not study([1.0, 0.5, 0.25]).settled("m")


def test_an_error_metric_gets_an_observed_order():
    """Halving the error on each doubling is second order."""
    assert study([1.0, 0.25, 0.0625]).observed_order("m") == pytest.approx(2.0)


def test_a_measured_quantity_gets_no_observed_order():
    """Without a reference there is no error to halve, so none is claimed."""
    assert math.isnan(study([1.0, 0.25, 0.0625], kind="quantity").observed_order("m"))


def test_a_structurally_zero_metric_is_not_a_failure_to_converge():
    """Two roundings of zero differ by tens of per cent and mean nothing."""
    noise = study([1.8e-15, 1.4e-15, 1.6e-15], kind="quantity")
    assert noise.changes("m") == [0.0, 0.0]
    assert noise.settled("m")
    assert noise.passed


def test_a_metric_that_starts_at_zero_and_grows_is_not_settled():
    """The floor covers noise, not a quantity that actually turns on."""
    grew = study([0.0, 0.0, 0.5], kind="quantity")
    assert not grew.settled("m")


def test_passed_requires_every_metric_to_settle():
    mixed = GridStudy(
        "test", [16, 32], {"a": [1.0, 1.0], "b": [1.0, 2.0]},
        {"a": "quantity", "b": "quantity"},
    )
    assert mixed.settled("a") and not mixed.settled("b")
    assert not mixed.passed


def test_the_report_says_which_metric_is_still_moving():
    text = study([1.0, 0.5, 0.25], kind="quantity").report()
    assert "STILL MOVING" in text and "m" in text
    assert "refine further" in text


# --------------------------------------------------------------------------- #
# What refinement is supposed to do depends on what the metric is
# --------------------------------------------------------------------------- #
def test_an_error_converges_by_moving_not_by_settling():
    """Halving the error on each doubling is a scheme working, not one stuck.

    Judged by the settled test -- which is the right question for a measured
    quantity -- a first-order solver converging perfectly moves 50% per grid and
    reads as a failure. Asking the wrong one of the two inverts the answer.
    """
    healthy = study([1.0, 0.5, 0.25])
    assert healthy.converging("m") and not healthy.settled("m")
    assert healthy.passed
    assert "converging at order 1.00" in healthy.report()


def test_an_error_that_has_settled_has_stopped_converging():
    """The same reading that means success for a quantity means failure here."""
    stalled = study([1.0, 0.5, 0.499])
    assert stalled.settled("m") and not stalled.converging("m")
    assert not stalled.passed
    assert "NOT CONVERGING" in stalled.report()


def test_an_error_that_grows_under_refinement_is_not_converging():
    assert not study([0.25, 0.5, 0.9]).converging("m")


def test_an_error_driven_to_zero_is_converged():
    """Zero is the destination; there is no order left to fit to it."""
    assert study([1e-3, 1e-6, 0.0]).converging("m")


def test_a_quantity_converges_by_settling():
    """And it gets no order, so the error branch cannot be reached for one."""
    quiet = study([1.0, 0.5, 0.5], kind="quantity")
    assert quiet.converging("m") and quiet.settled("m")
    assert math.isnan(quiet.observed_order("m"))


def test_an_error_creeping_downward_too_slowly_is_not_converging():
    """Half of first order: below that a sequence is decreasing, not converging."""
    creep = study([1.0, 0.95, 0.90])
    assert creep.observed_order("m") < MIN_CONVERGENCE_ORDER
    assert not creep.converging("m")


def test_the_report_names_the_resolutions_it_used():
    assert "16, 32, 64" in study([1.0, 0.5, 0.25]).report()


# --------------------------------------------------------------------------- #
# Running one for real
# --------------------------------------------------------------------------- #
def test_every_case_declares_what_it_can_be_refined_against():
    """A case with no declared metric cannot be studied, so all four must."""
    for name in available_cases():
        metrics = getattr(load_case(name), "CONVERGENCE_METRICS", {})
        assert metrics, f"{name} declares no CONVERGENCE_METRICS"
        assert set(metrics.values()) <= {"error", "quantity"}


def test_a_case_reports_every_metric_it_declares():
    """The declaration and the report must not drift apart."""
    result = load_case("cavity").run(nx=16, ny=16, t_end=0.05, make_plots=False)
    for metric in load_case("cavity").CONVERGENCE_METRICS:
        assert metric in result.metrics


def test_a_real_study_tracks_the_declared_metrics():
    result = grid_study("cavity", (16, 24), t_end=0.5)
    assert set(result.values) == set(load_case("cavity").CONVERGENCE_METRICS)
    assert result.resolutions == [16, 24]
    assert all(len(v) == 2 for v in result.values.values())


def test_refining_the_cavity_reduces_its_error_against_ghia():
    """The point of the whole exercise, on the cheapest case that shows it."""
    result = grid_study("cavity", (16, 32), t_end=6.0)
    errors = result.values["ghia_u_L2"]
    assert errors[1] < errors[0]


def test_a_case_without_declared_metrics_is_refused(monkeypatch):
    module = load_case("cavity")
    monkeypatch.setattr(module, "CONVERGENCE_METRICS", {}, raising=False)
    with pytest.raises(ValueError, match="does not declare CONVERGENCE_METRICS"):
        grid_study("cavity", (16, 24))


# --------------------------------------------------------------------------- #
# Through the CLI
# --------------------------------------------------------------------------- #
def test_convergence_alone_still_means_taylor_green():
    """The documented command predates --case being honoured; it must not move."""
    args = build_parser().parse_args(["--convergence"])
    assert args.case is None            # resolved in main(), not by argparse


def test_a_plain_run_still_defaults_to_the_cavity():
    from pycfd.main import _run

    args = build_parser().parse_args(["--list-cases"])
    assert args.case is None
    # main() resolves it; --list-cases exits before the case is used.
    assert _run(args) == EXIT_OK


def test_cli_runs_a_grid_study_on_a_named_case(tmp_path, capsys):
    code = main(["--convergence", "--case", "cavity", "--resolutions", "16,24",
                 "--t-end", "0.5", "--no-plots", "-q", "--outdir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "cavity grid study (16, 24)" in out
    assert "ghia_u_L2" in out
    assert code in (EXIT_OK, EXIT_VALIDATION_FAILED)


def test_cli_reports_a_bad_resolution_list_as_a_usage_error(tmp_path, capsys):
    code = main(["--convergence", "--case", "cavity", "--resolutions", "128,64",
                 "--no-plots", "-q", "--outdir", str(tmp_path)])
    assert code == EXIT_ERROR
    assert "must increase" in capsys.readouterr().err


def test_default_resolutions_are_a_usable_study():
    assert len(DEFAULT_RESOLUTIONS) >= 2
    assert parse_resolutions(",".join(str(n) for n in DEFAULT_RESOLUTIONS)) \
        == DEFAULT_RESOLUTIONS


def test_the_noise_floor_is_far_below_any_real_measurement():
    assert NOISE_FLOOR < 1e-6
