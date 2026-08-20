"""Command-line entry point for pycfd.

Examples
--------
::

    python main.py --list-cases
    python main.py --case cavity --re 400
    python main.py --case cavity --re 1000 --nx 128 --live
    python main.py --case cylinder --re 100 --live
    python main.py --case channel --re 10 --export-csv
    python main.py --convergence
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from pathlib import Path

# Allow ``python main.py`` from inside the package directory as well as
# ``python -m pycfd.main`` from its parent.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "pycfd"

from .cases import OUTLET_KINDS, available_cases, load_case  # noqa: E402
from .config import AdvectionScheme, PressureSolver, TimeScheme  # noqa: E402
from .core.timestepper import DivergenceError  # noqa: E402

log = logging.getLogger("pycfd")

DEFAULT_OUTPUT_DIR = "results"


def build_parser() -> argparse.ArgumentParser:
    """Assemble the full argument parser."""
    p = argparse.ArgumentParser(
        prog="pycfd",
        description="2D incompressible Navier-Stokes solver (projection method).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--case", default="cavity",
                   choices=sorted(available_cases()),
                   help="benchmark case to run")
    p.add_argument("--list-cases", action="store_true",
                   help="list the available benchmark cases and exit")
    p.add_argument("--convergence", action="store_true",
                   help="run the Taylor-Green grid-convergence study and exit")

    grid = p.add_argument_group("grid and physics")
    grid.add_argument("--re", type=float, default=None, help="Reynolds number")
    grid.add_argument("--nx", type=int, default=None, help="cells in x")
    grid.add_argument("--ny", type=int, default=None, help="cells in y")
    grid.add_argument("--dt", type=float, default=None, help="initial/fixed time step")
    grid.add_argument("--t-end", type=float, default=None, help="stop time")
    grid.add_argument("--max-steps", type=int, default=None,
                      help="hard cap on the number of time steps")
    grid.add_argument("--cfl", type=float, default=None, dest="cfl_max",
                      help="maximum Courant number for adaptive stepping")
    grid.add_argument("--mode", default=None,
                      help="case-specific variant (channel: periodic | developing)")
    grid.add_argument("--domain-length", type=float, default=None, metavar="L",
                      help="streamwise extent of the domain, in the same units as "
                           "the geometry (external-flow cases only)")
    grid.add_argument("--domain-height", type=float, default=None, metavar="H",
                      help="cross-stream extent of the domain; together with the "
                           "body size this sets the blockage ratio "
                           "(external-flow cases only)")

    bnd = p.add_argument_group("outflow boundary")
    bnd.add_argument("--outlet-type", default=None,
                     choices=[k.value for k in OUTLET_KINDS],
                     help="retype the downstream boundary: 'outlet' extrapolates "
                          "the velocity and lets the pressure float; "
                          "'pressure_outlet' holds the pressure at --p-ref and "
                          "solves for the velocity. Default: the case's own choice")
    bnd.add_argument("--p-ref", type=float, default=None, metavar="P",
                     help="pressure held on a pressure outlet (default 0.0). "
                          "Ignored, with a warning, by a velocity outlet")

    flow = p.add_argument_group("flow conditions (external-flow / cylinder case)")
    flow.add_argument("--l-ref", type=float, default=None, dest="l_ref", metavar="L",
                      help="reference length for the Reynolds number and the force "
                           "coefficients, in the geometry's own units. Default: the "
                           "body's extent across the flow, i.e. the cylinder-diameter "
                           "convention. Use it for the aircraft-length convention "
                           "instead. Blockage and grid-resolution figures keep "
                           "reporting the body's real span either way")
    flow.add_argument("--wind-speed", type=float, default=None, dest="wind_speed",
                      metavar="V",
                      help="free-stream speed in m/s. Derives the Reynolds number "
                           "from it and the ISA viscosity, treating the geometry's "
                           "length unit as the metre; mutually exclusive with --re")
    flow.add_argument("--altitude", type=float, default=None, metavar="Z",
                      help="altitude in metres, selecting the ISA air properties "
                           "--wind-speed uses (default: sea level)")

    geom = p.add_argument_group("custom geometry (external-flow / cylinder case)")
    geom.add_argument("--geometry", default=None, metavar="FILE",
                      help="2D body to place in the flow: a vertex file "
                           "(.csv/.txt/.dat, two columns x y) or a bitmap "
                           "silhouette (.png/.jpg, dark = solid)")
    geom.add_argument("--geometry-scale", type=float, default=1.0,
                      help="scale factor applied to a vertex outline")
    geom.add_argument("--geometry-rotate", type=float, default=0.0, metavar="DEG",
                      help="rotation applied to a vertex outline, in degrees")

    num = p.add_argument_group("numerics")
    num.add_argument("--time-scheme", default=None,
                     choices=[s.value for s in TimeScheme],
                     help="time integrator")
    num.add_argument("--advection", default=None,
                     choices=[s.value for s in AdvectionScheme],
                     help="convective discretisation")
    num.add_argument("--pressure-solver", default=None,
                     choices=[s.value for s in PressureSolver],
                     help="pressure Poisson solver")
    les = num.add_mutually_exclusive_group()
    les.add_argument("--les", dest="use_les", action="store_true", default=None,
                     help="enable the Smagorinsky sub-grid-scale model")
    les.add_argument("--no-les", dest="use_les", action="store_false", default=None,
                     help="force the laminar solver. Benchmarks are laminar by "
                          "default, so this only matters if a case or config "
                          "turns the model on")

    vis = p.add_argument_group("visualisation")
    vis.add_argument("--live", action="store_true",
                     help="show a real-time animation while the solver runs")
    vis.add_argument("--display", default="speed",
                     choices=("speed", "pressure", "vorticity", "streamlines"),
                     help="field shown by the live viewer")
    vis.add_argument("--plot-every", type=int, default=50,
                     help="solver steps between rendered frames")
    vis.add_argument("--quiver", action="store_true",
                     help="overlay velocity vectors on the live view")
    vis.add_argument("--save-animation", default=None, metavar="PATH",
                     help="write the animation to a file (e.g. wake.gif)")
    vis.add_argument("--frames", type=int, default=200,
                     help="frames to render when saving an animation")
    vis.add_argument("--no-plots", action="store_true",
                     help="skip the static figures a case would otherwise write")

    io = p.add_argument_group("input and output")
    io.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR, help="output directory")
    io.add_argument("--name", default=None,
                    help="label used in logs, figure titles and output filenames "
                         "(default: derived from the case and Reynolds number)")
    io.add_argument("--export-vtk", action="store_true",
                    help="write the final field as a legacy VTK file")
    io.add_argument("--export-csv", action="store_true",
                    help="write the final field as CSV")
    io.add_argument("--checkpoint", action="store_true",
                    help="save a restartable .npz checkpoint at the end")
    io.add_argument("--resume", default=None, metavar="PATH",
                    help="resume from a checkpoint instead of starting fresh")
    io.add_argument("--progress", action="store_true",
                    help="show a tqdm progress bar")

    p.add_argument("-v", "--verbose", action="store_true",
                   help="enable DEBUG logging")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="only report warnings and errors")
    return p


def configure_logging(verbose: bool, quiet: bool) -> None:
    """Set up the root logger (INFO by default, DEBUG with ``-v``)."""
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)


#: Options only some cases understand, mapped to the flag that sets them and the
#: property a case must have for the flag to mean anything.
CASE_SPECIFIC_FLAGS = {
    "outlet_type": ("--outlet-type", "outflow boundary to retype"),
    "p_ref": ("--p-ref", "outflow boundary to retype"),
    "domain_length": ("--domain-length", "configurable domain size"),
    "domain_height": ("--domain-height", "configurable domain size"),
    "l_ref": ("--l-ref", "body in the flow to measure a reference length on"),
    "wind_speed": ("--wind-speed", "body in the flow to size a Reynolds number against"),
    "altitude": ("--altitude", "body in the flow to size a Reynolds number against"),
}


def _case_accepts(case_name: str, param: str) -> bool:
    """Whether a case's ``build`` declares ``param``."""
    return param in inspect.signature(load_case(case_name).build).parameters


def case_specific_kwargs(args: argparse.Namespace) -> dict:
    """Forward the case-specific flags, refusing those the case cannot use.

    Support is read from each case's ``build`` signature rather than a hard-coded
    list, so a case that later gains an outflow boundary or a configurable domain
    picks the flags up with no change here.  A flag that cannot apply is an error
    rather than a silent no-op -- asking for a domain size and getting the
    default one back without a word is exactly the kind of thing that wastes an
    afternoon.
    """
    requested = {p: getattr(args, p) for p in CASE_SPECIFIC_FLAGS
                 if getattr(args, p, None) is not None}
    unsupported = [p for p in requested if not _case_accepts(args.case, p)]
    if unsupported:
        flags = ", ".join(CASE_SPECIFIC_FLAGS[p][0] for p in sorted(set(unsupported)))
        reason = CASE_SPECIFIC_FLAGS[unsupported[0]][1]
        supported = sorted(name for name in available_cases()
                           if _case_accepts(name, unsupported[0]))
        raise ValueError(
            f"{flags} cannot be used with case '{args.case}', which has no "
            f"{reason}; cases that do: {', '.join(supported)}"
        )
    return requested


def case_kwargs(args: argparse.Namespace) -> dict:
    """Collect the CLI overrides that a case's ``build``/``run`` understands."""
    mapping = {
        "re": args.re, "nx": args.nx, "ny": args.ny,
        "dt": args.dt, "t_end": args.t_end, "mode": args.mode,
    }
    kwargs = {k: v for k, v in mapping.items() if v is not None}

    overrides = {}
    if args.cfl_max is not None:
        overrides["cfl_max"] = args.cfl_max
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.time_scheme is not None:
        overrides["time_scheme"] = TimeScheme(args.time_scheme)
    if args.advection is not None:
        overrides["advection_scheme"] = AdvectionScheme(args.advection)
    if args.pressure_solver is not None:
        overrides["pressure_solver"] = PressureSolver(args.pressure_solver)
    if args.use_les is not None:
        overrides["use_les"] = args.use_les
    if args.name is not None:
        overrides["name"] = args.name
    kwargs.update(overrides)
    return kwargs


def write_outputs(sim, args: argparse.Namespace, tag: str) -> list[Path]:
    """Honour ``--export-vtk``, ``--export-csv`` and ``--checkpoint``."""
    outdir = Path(args.outdir)
    written: list[Path] = []
    if args.export_vtk:
        written.append(sim.export_vtk(outdir / f"{tag}.vtk"))
    if args.export_csv:
        written.append(sim.export_csv(outdir / f"{tag}.csv"))
    if args.checkpoint:
        written.append(sim.save_checkpoint(outdir / f"{tag}.npz"))
    return written


def run_live(args: argparse.Namespace) -> int:
    """Build the case and animate it instead of running it headlessly."""
    from .visualization.live_plot import LiveViewer

    module = load_case(args.case)
    kwargs = case_kwargs(args)
    if args.case != "channel":
        kwargs.pop("mode", None)          # only the channel has a mode variant
    kwargs.update(case_specific_kwargs(args))

    if args.resume:
        from .physics.incompressible import Simulation
        sim = Simulation.from_checkpoint(args.resume)
    elif args.geometry is not None:
        if args.case != "cylinder":
            raise ValueError(
                f"--geometry places a body in an external flow and applies to "
                f"--case cylinder, not '{args.case}'"
            )
        sim = module.build_from_geometry(
            args.geometry, scale=args.geometry_scale,
            rotate_deg=args.geometry_rotate, **kwargs,
        )
    else:
        sim = module.build(**kwargs)

    viewer = LiveViewer(sim, mode=args.display, plot_every=args.plot_every,
                        t_end=args.t_end, quiver=args.quiver)
    log.info("live view: space pauses/resumes, q closes the window")
    viewer.start(save_path=args.save_animation,
                 max_frames=args.frames if args.save_animation else None,
                 show=args.save_animation is None)

    tag = f"{args.case}_live"
    for path in write_outputs(sim, args, tag):
        log.info("wrote %s", path)
    print()
    for key, value in sim.diagnostics().items():
        print(f"  {key:<20s} {value:.6g}" if isinstance(value, float)
              else f"  {key:<20s} {value}")
    return EXIT_OK


#: Exit codes.  0 = ran and validated, 1 = ran but a validation check failed,
#: 2 = could not run (bad configuration, divergence, unimplemented option).
EXIT_OK, EXIT_VALIDATION_FAILED, EXIT_ERROR = 0, 1, 2


def run_resume(args: argparse.Namespace) -> int:
    """Continue a checkpointed run headlessly.

    The checkpoint carries its own configuration, so the only thing normally
    worth overriding is ``--t-end``: without it the run would already be at the
    stored end time and stop immediately.
    """
    from .physics.incompressible import Simulation
    from .visualization import static_plot as sp

    sim = Simulation.from_checkpoint(args.resume)
    t_end = args.t_end
    if t_end is None:
        t_end = sim.config.t_end
    if t_end <= sim.fields.t + 1e-12:
        raise ValueError(
            f"the checkpoint is already at t = {sim.fields.t:g} and --t-end is "
            f"{t_end:g}; pass a larger --t-end to continue the run"
        )

    result = sim.run(t_end=t_end, progress=args.progress)
    outdir = Path(args.outdir) / args.case
    written = write_outputs(sim, args, f"{sim.config.name}_resumed")

    if not args.no_plots:
        path = outdir / f"{sim.config.name}_resumed_fields.png"
        sp.four_panel_figure(
            sim.fields, solid=sim.solid_mask,
            title=f"{sim.config.name}, t = {sim.fields.t:g}", path=path,
        )
        written.append(path)

    print()
    print(f"=== resumed {sim.config.name} ===")
    print(f"  {result.summary()}")
    for key, value in sim.diagnostics().items():
        print(f"  {key:<20s} {value:.6g}" if isinstance(value, float)
              else f"  {key:<20s} {value}")
    for path in written:
        print(f"  wrote {path}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose, args.quiet)
    try:
        return _run(args)
    except (ValueError, NotImplementedError, FileNotFoundError) as exc:
        # Configuration and usage problems: report the message, not a traceback.
        # Use --verbose to see the full stack.
        log.debug("aborting", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except DivergenceError as exc:
        log.debug("aborting", exc_info=True)
        print(f"error: the simulation diverged\n  {exc}", file=sys.stderr)
        return EXIT_ERROR


def _run(args: argparse.Namespace) -> int:
    """Dispatch the parsed arguments; see :func:`main` for error handling."""

    if args.list_cases:
        print("Available cases:\n")
        for name, description in available_cases().items():
            print(f"  {name:<14s} {description}")
        return EXIT_OK

    if not args.live:
        # Static figures must not need a display.
        from .visualization.static_plot import use_headless_backend
        use_headless_backend()

    if args.convergence:
        module = load_case("taylor_green")
        result = module.run_convergence(outdir=Path(args.outdir) / "convergence",
                                        make_plots=not args.no_plots,
                                        progress=args.progress)
        print()
        print(result.report())
        tables = getattr(result, "tables", {})
        for norm, study in tables.items():
            print(f"\n  {norm} refinement table:")
            print("\n".join("    " + line for line in study.table().splitlines()))
        return EXIT_OK if result.passed else EXIT_VALIDATION_FAILED

    if args.live:
        return run_live(args)

    if args.resume:
        return run_resume(args)

    module = load_case(args.case)
    kwargs = case_kwargs(args)
    if args.case != "channel":
        kwargs.pop("mode", None)
    kwargs.update(case_specific_kwargs(args))
    if args.geometry is not None:
        if args.case != "cylinder":
            raise ValueError(
                f"--geometry places a body in an external flow and applies to "
                f"--case cylinder, not '{args.case}'"
            )
        kwargs.update(geometry=args.geometry,
                      geometry_scale=args.geometry_scale,
                      geometry_rotate=args.geometry_rotate)

    result = module.run(outdir=Path(args.outdir) / args.case,
                        make_plots=not args.no_plots,
                        progress=args.progress, **kwargs)

    # Surface the outlet anchor in the report whenever one is active, so that
    # --p-ref is visibly doing something rather than silently accepted.
    if result.simulation.solver.dirichlet_pressure:
        walls = result.simulation.solver.dirichlet_pressure
        result.metrics["outlet_p_ref"] = float(next(iter(walls.values())))
        result.metrics["outlet_p_deviation"] = \
            result.simulation.outlet_pressure_deviation()

    # The configuration's own name, so --name reaches the files on disk.  It
    # already defaults to "<case>_Re<re>", so this changes nothing when --name
    # is absent.
    tag = result.simulation.config.name
    for path in write_outputs(result.simulation, args, tag):
        result.outputs.append(path)

    print()
    print(result.report())
    return EXIT_OK if result.passed else EXIT_VALIDATION_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
