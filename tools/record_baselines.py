#!/usr/bin/env python3
"""Re-record the benchmark baselines in ``pycfd/tests/baselines.json``.

Run this only when a number has moved *and you have decided the new value is
correct*.  Re-recording is deliberately a separate command rather than a pytest
flag: a baseline that refreshes itself records whatever the code currently does,
which is the opposite of what a regression test is for.

    python tools/record_baselines.py --fast          # ~10 s, the CI tier
    python tools/record_baselines.py --full          # ~9 min, README numbers
    python tools/record_baselines.py --fast --full   # both

Commit the regenerated file in the same change that caused the movement, so the
diff shows the old and new numbers side by side.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)

import logging  # noqa: E402
logging.basicConfig(level=logging.ERROR)

from pycfd import __version__  # noqa: E402
from pycfd.analysis.provenance import git_commit  # noqa: E402
from pycfd.cases import load_case  # noqa: E402

BASELINE_PATH = REPO / "pycfd" / "tests" / "baselines.json"

CHANNEL_KEYS = ("centerline_error_pct", "profile_L2_relative", "u_max_numerical",
                "mean_velocity", "max_divergence", "steps", "final_time")
CAVITY_KEYS = ("ghia_u_L2", "ghia_u_Linf", "ghia_v_L2", "ghia_v_Linf",
               "max_divergence", "steps", "final_time", "reached_steady_state")
CYLINDER_KEYS = ("cd_mean", "cl_rms", "characteristic_length", "blockage_ratio",
                 "max_divergence", "steps", "final_time")


def timed(fn):
    """Run ``fn``, returning ``(result, elapsed_seconds)``."""
    start = time.perf_counter()
    return fn(), round(time.perf_counter() - start, 1)


def entry(params: dict, metrics: dict, keys, seconds: float) -> dict:
    return {"params": params,
            "metrics": {k: metrics[k] for k in keys},
            "seconds": seconds}


def record_fast() -> dict:
    """The tier that runs on every push."""
    out = {}

    def convergence():
        l2, linf, _ = load_case("taylor_green").convergence_study(
            (16, 32, 64, 128), progress=False)
        return {"observed_order_L2": l2.observed_order,
                "observed_order_Linf": linf.observed_order,
                "finest_L2": l2.errors[-1],
                "finest_Linf": linf.errors[-1]}

    metrics, seconds = timed(convergence)
    out["taylor_green_convergence"] = {
        "params": {"resolutions": [16, 32, 64, 128]},
        "metrics": metrics, "seconds": seconds}
    print(f"  taylor_green_convergence  {seconds:6.1f}s")

    channel = load_case("channel")
    for name, p in (("channel_periodic", {"re": 10, "nx": 16, "ny": 32, "mode": "periodic"}),
                    ("channel_developing", {"re": 10, "nx": 48, "ny": 24, "mode": "developing"})):
        r, seconds = timed(lambda p=p: channel.run(**p, make_plots=False))
        out[name] = entry(p, r.metrics, CHANNEL_KEYS, seconds)
        print(f"  {name:25s} {seconds:6.1f}s")

    p = {"re": 100, "nx": 48, "ny": 48}
    r, seconds = timed(lambda: load_case("cavity").run(**p, make_plots=False))
    out["cavity_re100"] = entry(p, r.metrics, CAVITY_KEYS, seconds)
    print(f"  {'cavity_re100':25s} {seconds:6.1f}s")

    p = {"re": 20, "nx": 128, "ny": 64, "t_end": 20.0}
    r, seconds = timed(lambda: load_case("cylinder").run(**p, make_plots=False))
    out["cylinder_re20"] = entry(p, r.metrics, CYLINDER_KEYS, seconds)
    print(f"  {'cylinder_re20':25s} {seconds:6.1f}s")
    return out


def record_full() -> dict:
    """The tier that pins the resolutions the README quotes."""
    out = {}
    channel = load_case("channel")
    for name, p in (("channel_periodic_full", {"re": 10, "nx": 32, "ny": 64, "mode": "periodic"}),
                    ("channel_developing_full", {"re": 10, "nx": 96, "ny": 48, "mode": "developing"})):
        r, seconds = timed(lambda p=p: channel.run(**p, make_plots=False))
        out[name] = entry(p, r.metrics,
                          ("centerline_error_pct", "profile_L2_relative",
                           "max_divergence", "steps"), seconds)
        print(f"  {name:25s} {seconds:6.1f}s")

    cavity = load_case("cavity")
    for re in (100, 400, 1000):
        p = {"re": re, "nx": 128, "ny": 128}
        r, seconds = timed(lambda p=p: cavity.run(**p, make_plots=False))
        out[f"cavity_re{re}_full"] = entry(p, r.metrics,
                                           ("ghia_u_L2", "ghia_u_Linf", "ghia_v_L2",
                                            "ghia_v_Linf", "max_divergence", "steps"),
                                           seconds)
        print(f"  cavity_re{re}_full{'':11s} {seconds:6.1f}s")

    p = {"re": 20, "nx": 256, "ny": 128}
    r, seconds = timed(lambda: load_case("cylinder").run(**p, make_plots=False))
    out["cylinder_re20_full"] = entry(p, r.metrics,
                                      ("cd_mean", "cl_rms", "max_divergence", "steps"),
                                      seconds)
    print(f"  {'cylinder_re20_full':25s} {seconds:6.1f}s")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fast", action="store_true",
                        help="re-record the fast tier (~10 s)")
    parser.add_argument("--full", action="store_true",
                        help="re-record the full-fidelity tier (~9 min)")
    args = parser.parse_args(argv)
    if not (args.fast or args.full):
        parser.error("choose at least one of --fast / --full")

    doc = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}

    if args.fast:
        print("recording fast tier:")
        doc["fast"] = record_fast()
    if args.full:
        print("recording full-fidelity tier:")
        doc["full"] = record_full()

    doc["_meta"] = {
        "purpose": "Recorded benchmark results. A test failure means a number moved, "
                   "which is either a regression or an improvement -- decide which, "
                   "then re-record deliberately with tools/record_baselines.py.",
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pycfd_version": __version__,
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "determinism": "Verified bit-identical across repeated runs on the recording "
                       "platform; tolerances in test_regression.py allow for "
                       "cross-platform float variation.",
    }
    BASELINE_PATH.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nwrote {BASELINE_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
