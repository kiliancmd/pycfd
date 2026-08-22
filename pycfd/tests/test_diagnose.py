"""The end-to-end sanity pass: what it flags, and what it refuses to flag.

A diagnostic tool earns its place by being right about the boring cases. If it
cries wolf on a healthy run nobody reads its output, and if it stays quiet on a
body four cells wide it has actively cost somebody an afternoon. So most of what
is pinned here is the *level* each check returns rather than its prose, and the
sharpest tests are the ones where the naive answer is the wrong one:

- an error falling 55% under refinement is a scheme converging beautifully, not
  a number that refuses to settle -- which is what the first version of this
  said about the channel, at an observed order of 2.00;
- a quantity that stops moving has converged, and the same test applied to an
  error would call a stalled solver healthy;
- a run that blows up has to come back as a finding rather than a traceback,
  because a diagnostic that raises is one people route around.

The remedies are checked arithmetically, not textually: a warning that says to
run at N=256 is only useful if N=256 actually delivers what it promises.
"""

import math

import pytest

from pycfd.analysis.diagnose import (
    BLOCKAGE_NOTICEABLE,
    CELLS_QUANTITATIVE,
    DIVERGENCE_FAIL,
    DIVERGENCE_WARN,
    FAIL,
    OK,
    WARN,
    Diagnosis,
    Finding,
    _check_blockage,
    _check_body_resolution,
    _check_case_validation,
    _check_compressibility,
    _check_continuity,
    _check_convergence,
    diagnose,
    diagnostic_resolutions,
)
from pycfd.cases import CaseResult, GridStudy, available_cases, case_resolution
from pycfd.core.timestepper import DivergenceError
from pycfd.main import EXIT_ERROR, EXIT_OK, EXIT_VALIDATION_FAILED, build_parser, main


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def result(metrics=None, checks=()):
    """A CaseResult carrying only what the checks actually read."""
    return CaseResult("test", None, dict(metrics or {}), [], list(checks))


def study(values=(1.0, 1.0), kind="quantity", resolutions=(128, 192),
          metrics=None, checks=()):
    """A GridStudy with one metric and one set of per-grid metrics."""
    per_grid = [result(m, checks) for m in (metrics or [{}] * len(resolutions))]
    return GridStudy("test", list(resolutions), {"m": list(values)},
                     {"m": kind}, per_grid)


def levels(findings):
    return [f.level for f in findings]


# --------------------------------------------------------------------------- #
# Choosing the grids
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", sorted(available_cases()))
def test_diagnostic_grids_sit_below_the_case_own_default(case):
    """The premise of the whole pass: refine *up to* the grid you were going to
    use, so the verdict says something about that run rather than a finer one."""
    grids = diagnostic_resolutions(case)
    production = case_resolution(case)
    assert len(grids) >= 2
    assert list(grids) == sorted(set(grids))
    assert max(grids) < production, f"{case}: {grids} is not below {production}"


def test_a_tiny_case_still_gets_two_distinct_grids():
    """Fractions of a small default collide; one grid is not a refinement."""
    grids = diagnostic_resolutions("channel", fractions=(0.01, 0.02))
    assert len(grids) == 2 and grids[0] < grids[1]


def test_an_explicit_pair_is_honoured():
    assert diagnostic_resolutions("cylinder", fractions=(0.25, 1.0)) == (64, 256)


# --------------------------------------------------------------------------- #
# Severity bookkeeping
# --------------------------------------------------------------------------- #
def test_the_diagnosis_reports_its_worst_finding():
    d = Diagnosis("test", [16, 32], [
        Finding("a", OK, ""), Finding("b", WARN, ""), Finding("c", OK, ""),
    ])
    assert d.level == WARN and d.usable
    d.findings.append(Finding("d", FAIL, ""))
    assert d.level == FAIL and not d.usable


def test_a_diagnosis_with_nothing_to_say_is_clean():
    d = Diagnosis("test", [16, 32])
    assert d.level == OK and d.usable
    assert "nothing to flag" in d.report()


def test_the_report_leads_with_the_worst_finding():
    """Somebody who ran this wants the problem at the top, not after the passes."""
    d = Diagnosis("test", [16, 32], [
        Finding("fine", OK, "all good"), Finding("broken", FAIL, "it blew up"),
    ])
    text = d.report()
    assert text.index("broken") < text.index("fine")
    assert "1 failed, 0 warned, 1 passed" in text
    assert "will not support a quantitative answer" in text


# --------------------------------------------------------------------------- #
# Continuity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("worst,level", [
    (1.0e-14, OK),                       # what a healthy projection leaves
    (DIVERGENCE_WARN * 10, WARN),
    (DIVERGENCE_FAIL * 10, FAIL),
])
def test_continuity_is_graded_by_how_much_divergence_survived(worst, level):
    s = study(metrics=[{"max_divergence": 1e-15}, {"max_divergence": worst}])
    assert levels(_check_continuity(s)) == [level]


def test_the_worst_grid_sets_the_continuity_verdict():
    """A fine grid that behaves does not excuse a coarse one that did not."""
    s = study(metrics=[{"max_divergence": DIVERGENCE_FAIL * 10},
                       {"max_divergence": 1e-15}])
    assert levels(_check_continuity(s)) == [FAIL]


# --------------------------------------------------------------------------- #
# Blockage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio,level", [
    (0.02, OK), (0.07, WARN), (0.125, WARN),
])
def test_blockage_is_graded_by_how_much_of_the_channel_the_body_fills(ratio, level):
    s = study(metrics=[{"blockage_ratio": ratio, "characteristic_length": 1.0}] * 2)
    assert levels(_check_blockage(s)) == [level]


def test_the_blockage_remedy_would_actually_fix_the_blockage():
    """The advised domain height has to deliver the ratio it promises."""
    span = 2.5
    s = study(metrics=[{"blockage_ratio": 0.3, "characteristic_length": span}] * 2)
    advice = _check_blockage(s)[0].advice
    height = span / BLOCKAGE_NOTICEABLE
    assert f"--domain-height {height:.6g}" in advice
    assert span / height <= BLOCKAGE_NOTICEABLE


# --------------------------------------------------------------------------- #
# Body resolution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cells,level", [
    (20.0, OK), (8.0, WARN), (2.0, FAIL), (0.0, FAIL),
])
def test_body_resolution_is_graded_by_cells_across_the_body(cells, level):
    s = study(metrics=[{"cells_across_body": cells / 2}, {"cells_across_body": cells}])
    assert levels(_check_body_resolution(s)) == [level]


def test_the_resolution_remedy_would_actually_resolve_the_body():
    """`N=256` is only useful advice if N=256 really spans 16 cells."""
    s = study(resolutions=(128, 192),
              metrics=[{"cells_across_body": 8.0}, {"cells_across_body": 12.0}])
    advice = _check_body_resolution(s)[0].advice
    assert "N=256" in advice
    # Cells scale with the grid, so 192 -> 256 takes 12 cells to 16.
    assert 12.0 * 256 / 192 >= CELLS_QUANTITATIVE


def test_the_ladder_is_reported_not_just_the_finest_grid():
    """Resolution is the thing under study; one number hides the trend."""
    s = study(metrics=[{"cells_across_body": 8.0}, {"cells_across_body": 12.0}])
    assert "8.0 -> 12.0" in _check_body_resolution(s)[0].detail


# --------------------------------------------------------------------------- #
# A case that reports none of it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("check", [
    _check_blockage, _check_body_resolution, _check_compressibility,
    _check_continuity, _check_case_validation,
])
def test_a_check_says_nothing_when_the_case_reports_nothing(check):
    """One list of checks has to serve a closed cavity and an aircraft alike."""
    assert check(study()) == []


def test_a_partial_metric_is_not_read_as_a_measurement():
    """A metric one grid reported and another did not is not a ladder."""
    s = study(metrics=[{"blockage_ratio": 0.02}, {}])
    assert _check_blockage(s) == []


# --------------------------------------------------------------------------- #
# Compressibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mach,level", [(0.1, OK), (0.3, OK), (0.6, WARN)])
def test_compressibility_is_flagged_past_the_incompressible_limit(mach, level):
    s = study(metrics=[{"mach": mach}] * 2)
    assert levels(_check_compressibility(s)) == [level]


# --------------------------------------------------------------------------- #
# Convergence -- where the naive answer is the wrong one
# --------------------------------------------------------------------------- #
def test_an_error_falling_under_refinement_is_converging_not_moving():
    """The bug this pass found in itself.

    Second-order convergence on a doubling ladder moves the error by 75%. Judged
    by "did it stop moving?", a perfectly healthy solver fails.
    """
    s = study([1.0, 0.25], kind="error", resolutions=(16, 32))
    assert s.converging("m") and not s.settled("m")
    finding = _check_convergence(s)[0]
    assert finding.level == OK
    assert "fell 75.00%" in finding.detail and "order of 2.00" in finding.detail


def test_an_error_that_has_stopped_falling_is_flagged():
    """A settled error is a solver that has stopped converging -- the opposite
    of the same reading on a measured quantity."""
    s = study([1.0, 0.999], kind="error", resolutions=(16, 32))
    assert s.settled("m") and not s.converging("m")
    finding = _check_convergence(s)[0]
    assert finding.level == WARN
    assert "not falling under refinement" in finding.advice


def test_an_error_that_grew_is_not_converging():
    s = study([0.5, 0.8], kind="error", resolutions=(16, 32))
    assert not s.converging("m")


def test_an_exact_error_is_converged_not_stalled():
    """Zero error is the destination, not a failure to move toward it."""
    assert study([1e-3, 0.0], kind="error", resolutions=(16, 32)).converging("m")


def test_a_quantity_that_stopped_moving_has_converged():
    s = study([1.320, 1.322], resolutions=(128, 192))
    assert levels(_check_convergence(s)) == [OK]


def test_a_quantity_still_moving_is_flagged_with_where_to_go_next():
    s = study([1.20, 1.35], resolutions=(128, 192))
    finding = _check_convergence(s)[0]
    assert finding.level == WARN
    assert "a third grid would say where it is heading" in finding.advice


def quadratic(resolutions, limit=1.32, coefficient=20.0):
    """``f(h) = limit + C h^2`` -- a sequence with a known destination."""
    return [limit + coefficient * (1.0 / n) ** 2 for n in resolutions]


def test_a_third_grid_replaces_the_advice_with_an_extrapolated_value():
    """What #18 buys the pass: not 'refine more' but 'it is heading here'."""
    grids = (8, 16, 32)
    finding = _check_convergence(study(quadratic(grids), resolutions=grids))[0]
    assert finding.level == WARN
    assert "extrapolates to 1.32" in finding.advice
    assert "refine past" not in finding.advice


def test_a_settled_quantity_still_says_where_it_extrapolated_to():
    """Converged is not the same as converged-to-what; the limit is the answer."""
    grids = (32, 64, 128)
    finding = _check_convergence(study(quadratic(grids), resolutions=grids))[0]
    assert finding.level == OK
    assert "extrapolates to 1.32" in finding.detail and "GCI" in finding.detail


def test_a_diverging_sequence_is_flagged_even_when_the_last_step_looked_small():
    s = study([1.0, 1.5, 2.6], resolutions=(32, 64, 128))
    finding = _check_convergence(s)[0]
    assert finding.level == WARN
    assert "R = " in finding.detail and "not converging" in finding.advice


def test_every_refined_metric_gets_its_own_finding():
    s = GridStudy("test", [128, 192], {"cd": [1.32, 1.32], "cl": [0.1, 0.5]},
                  {"cd": "quantity", "cl": "quantity"}, [result(), result()])
    findings = _check_convergence(s)
    assert {f.name for f in findings} == {"convergence of cd", "convergence of cl"}
    assert sorted(levels(findings)) == [OK, WARN]


# --------------------------------------------------------------------------- #
# The case's own verdict
# --------------------------------------------------------------------------- #
def test_a_failed_case_check_is_surfaced_by_name():
    """The case is the authority on its own numbers; this reports, not re-derives."""
    s = study(checks=[("Cd at Re=100", True, "1.34"),
                      ("force average is stationary", False, "3.6 standard errors")])
    finding = _check_case_validation(s)[0]
    assert finding.level == WARN
    assert "force average is stationary" in finding.detail
    assert "1 of 2" in finding.detail
    assert "Cd at Re=100" not in finding.detail       # passes are not restated


def test_all_checks_passing_is_reported_at_the_finest_grid():
    s = study(checks=[("Cd at Re=100", True, "1.34")])
    finding = _check_case_validation(s)[0]
    assert finding.level == OK and "N=192" in finding.detail


# --------------------------------------------------------------------------- #
# A run that blows up
# --------------------------------------------------------------------------- #
def test_a_diverged_run_comes_back_as_a_finding_not_a_traceback(monkeypatch):
    def boom(*args, **kwargs):
        raise DivergenceError("the 'cylinder' case diverged at N=128: |u| = 1e9")

    monkeypatch.setattr("pycfd.analysis.diagnose.grid_study", boom)
    d = diagnose("cylinder")
    assert d.level == FAIL and not d.usable
    assert "diverged at N=128" in d.findings[0].detail
    assert "--cfl" in d.findings[0].advice


def test_a_divergence_says_which_grid_it_happened_on(monkeypatch):
    """A study that fails only on its coarsest grid is a different problem from
    one that fails everywhere, and the bare exception says neither."""
    from pycfd.cases import grid_study, load_case

    module = load_case("cavity")
    real_run = module.run

    def explode(nx, **kwargs):
        if nx >= 24:
            raise DivergenceError("|u| exceeded 1e6")
        return real_run(nx=nx, **kwargs)

    monkeypatch.setattr(module, "run", explode)
    with pytest.raises(DivergenceError, match=r"'cavity' case diverged at N=24"):
        grid_study("cavity", (16, 24), t_end=0.05)


# --------------------------------------------------------------------------- #
# Through the CLI
# --------------------------------------------------------------------------- #
def test_diagnose_alone_means_the_case_with_a_body_in_the_flow():
    """--convergence means Taylor-Green because that one has an exact solution;
    --diagnose means the cylinder because that one has a body to confine."""
    args = build_parser().parse_args(["--diagnose"])
    assert args.case is None                  # resolved in main(), not argparse
    assert args.diagnose and not args.convergence


def test_a_study_and_a_diagnosis_are_not_the_same_request(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--diagnose", "--convergence"])
    assert "not allowed with" in capsys.readouterr().err


@pytest.mark.parametrize("level,code", [
    (OK, EXIT_OK), (WARN, EXIT_VALIDATION_FAILED), (FAIL, EXIT_ERROR),
])
def test_the_exit_code_carries_the_worst_verdict(monkeypatch, tmp_path, level, code):
    """Usable as a gate: 0 clean, 1 somebody has a judgement to make, 2 broken."""
    monkeypatch.setattr(
        "pycfd.analysis.diagnose.diagnose",
        lambda *a, **k: Diagnosis("cavity", [16, 24], [Finding("x", level, "d")]),
    )
    assert main(["--diagnose", "--case", "cavity", "-q",
                 "--outdir", str(tmp_path)]) == code


def test_the_cli_runs_a_real_diagnosis(tmp_path, capsys):
    code = main(["--diagnose", "--case", "cavity", "--resolutions", "16,24",
                 "--t-end", "0.5", "--no-plots", "-q", "--outdir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "cavity diagnosis: grids 16, 24" in out
    assert "the case's own default is 128" in out
    assert "continuity" in out and "convergence of ghia_u_L2" in out
    assert code in (EXIT_OK, EXIT_VALIDATION_FAILED)


def test_a_geometry_reaches_the_study_instead_of_being_dropped(tmp_path, monkeypatch):
    """A study that quietly studies a cylinder when it was handed an aircraft is
    the worst kind of wrong answer: a complete, plausible one."""
    from pycfd.main import study_kwargs

    args = build_parser().parse_args(
        ["--diagnose", "--case", "cylinder", "--geometry", "wing.csv",
         "--geometry-scale", "3.0"]
    )
    assert study_kwargs(args)["geometry"] == "wing.csv"
    assert study_kwargs(args)["geometry_scale"] == 3.0


def test_a_geometry_on_a_case_with_no_body_is_refused(tmp_path, capsys):
    code = main(["--diagnose", "--case", "cavity", "--geometry", "wing.csv",
                 "-q", "--outdir", str(tmp_path)])
    assert code == EXIT_ERROR
    assert "--case cylinder" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# What belongs in a grid study at all
# --------------------------------------------------------------------------- #
def test_the_channel_centreline_error_is_set_by_the_clock_not_the_grid():
    """Why `centerline_error_pct` is not one of the channel's grid metrics.

    It looks like the obvious second entry -- it is an error against a known
    peak -- but refinement cannot move it. What sets it is the tolerance at
    which the time integration is allowed to stop, and a study that tracked it
    would report the channel as failing to converge for a reason that has
    nothing to do with convergence. Both halves are asserted, because only the
    pair rules out a coincidence.
    """
    import pycfd.cases.channel_flow as channel

    assert "centerline_error_pct" not in channel.CONVERGENCE_METRICS

    original = channel.STEADY_TOLERANCE
    errors = {}
    try:
        for tol in (1e-7, 1e-9):
            channel.STEADY_TOLERANCE = tol
            errors[tol] = channel.run(nx=16, ny=32,
                                      make_plots=False).metrics["centerline_error_pct"]
        channel.STEADY_TOLERANCE = original
        coarse = channel.run(nx=16, ny=32, make_plots=False)
        fine = channel.run(nx=24, ny=48, make_plots=False)
    finally:
        channel.STEADY_TOLERANCE = original

    # Tightening the clock by 100x tightens the error by 100x ...
    assert errors[1e-7] / errors[1e-9] == pytest.approx(100.0, rel=0.05)
    # ... while refining the grid by half as much again does essentially nothing.
    a, b = (coarse.metrics["centerline_error_pct"],
            fine.metrics["centerline_error_pct"])
    assert abs(b - a) / a < 0.05
    # The metric the channel *is* studied on behaves the other way round.
    assert fine.metrics["profile_L2_relative"] < 0.6 * coarse.metrics["profile_L2_relative"]


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
def test_the_divergence_thresholds_sit_above_a_healthy_projection():
    """A run that behaves must not trip them: the solver lands near round-off."""
    assert DIVERGENCE_WARN > 1e-11 and DIVERGENCE_FAIL > DIVERGENCE_WARN


def test_a_finding_renders_its_advice_only_when_it_has_some():
    assert len(Finding("x", OK, "fine").lines()) == 2
    assert len(Finding("x", WARN, "iffy", "do this").lines()) == 3


def test_the_levels_are_ordered_by_severity():
    assert Finding("x", OK, "").severity < Finding("x", WARN, "").severity
    assert Finding("x", WARN, "").severity < Finding("x", FAIL, "").severity
