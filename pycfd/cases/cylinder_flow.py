"""Flow past a circular cylinder using a mask-based immersed boundary.

The cylinder is represented by a boolean cell mask; every staggered face it
touches is held at zero velocity and dropped from the pressure stencil.  At
Re = 20 the wake is a steady symmetric recirculation pair; at Re = 100 the
symmetry is unstable and the flow settles into the von Karman vortex street.

Symmetry breaking
-----------------
On a symmetric grid the symmetric (unstable) solution is an exact discrete
steady state, and only round-off eventually breaks it -- which can take a very
long time.  A brief transverse pulse is therefore applied just behind the
cylinder at start-up so shedding develops on a sensible time scale.  It is a
nudge out of an unstable equilibrium, not a change to the physics; the resulting
shedding frequency is set entirely by the flow.

Accuracy note
-------------
A mask-based immersed boundary on a Cartesian grid is first-order accurate at
the surface, and this domain has a finite blockage ratio.  The drag coefficient
should therefore be read as an engineering estimate -- the literature ranges are
reported alongside it rather than as a tight pass/fail bound.

Run with::

    python main.py --case cylinder --re 100 --live
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..analysis.postprocess import Probe, force_coefficients, strouhal_number
from ..analysis.timeseries import time_average
from ..config import BCKind, BCSpec, SimulationConfig
from ..geometry.obstacles import circle_mask
from ..physics.incompressible import Simulation
from ..units import INCOMPRESSIBLE_MACH_LIMIT, Scaling

log = logging.getLogger(__name__)

#: Free-stream speed and cylinder diameter: the reference scales for Re, Cd and St.
U_INF = 1.0
DIAMETER = 1.0

#: Reynolds number used when neither ``re`` nor flight conditions are given.
DEFAULT_RE = 100.0

#: Domain in diameters: enough upstream to feel uniform, enough downstream for a wake.
DOMAIN_LENGTH = 16.0
DOMAIN_HEIGHT = 8.0
CYLINDER_X = 4.0

#: Static pressure held at the downstream boundary.  Only differences matter,
#: so zero simply makes the far field the datum.
OUTLET_PRESSURE = 0.0

#: Amplitude and duration of the start-up pulse that breaks the wake symmetry.
PERTURBATION_AMPLITUDE = 0.20
PERTURBATION_TIME = 1.0

#: Published values for an unbounded cylinder, used for reporting.
REFERENCE_CD = {20: (2.0, 2.1), 100: (1.32, 1.40)}
REFERENCE_ST = {100: (0.160, 0.172)}

#: Fraction of the record discarded before averaging force coefficients.
TRANSIENT_FRACTION = 0.5

#: What a grid study on this case should watch.  Both are *measured
#: quantities*, not errors: an immersed boundary on a Cartesian grid has no
#: exact answer to be wrong against, so refinement can only be checked by
#: whether the numbers stop moving -- which is exactly what the F-22 sweep
#: found they had not, between 5, 10 and 20 cells across the body.
CONVERGENCE_METRICS = {"cd_mean": "quantity", "cl_rms": "quantity"}


def flight_scaling(wind_speed: float, altitude: float | None = None) -> Scaling:
    """The solver-to-SI exchange rate implied by a speed and an altitude.

    The geometry's length unit is taken to be the metre -- a body loaded from a
    file carries its own dimensions, and there is nothing else those numbers
    could sensibly mean once a real wind speed is attached to them.
    """
    return Scaling.at_altitude(wind_speed, length=1.0,
                               altitude=0.0 if altitude is None else altitude)


def _resolve_reynolds(re: float | None, wind_speed: float | None,
                      altitude: float | None, reference_length: float) -> float:
    """Reynolds number from either a direct value or real flight conditions.

    The two routes are mutually exclusive on purpose: a run that was handed both
    a Reynolds number and a wind speed has been given two different answers to
    the same question, and picking one silently is how a sweep ends up plotted
    against the wrong axis.
    """
    if wind_speed is None:
        if altitude is not None:
            raise ValueError(
                "altitude only selects the air properties that turn a wind speed "
                "into a Reynolds number; pass --wind-speed as well, or drop it"
            )
        return DEFAULT_RE if re is None else float(re)

    if re is not None:
        raise ValueError(
            f"re={re:g} and wind_speed={wind_speed:g} m/s both set the Reynolds "
            "number; pass one or the other. --wind-speed derives it from the "
            "reference length and the ISA viscosity at --altitude"
        )

    scaling = flight_scaling(wind_speed, altitude)
    re = scaling.reynolds(reference_length)
    log.info(
        "%g m/s at %g m altitude, L_ref = %g m, nu = %.4g m^2/s -> Re = %.4g",
        wind_speed, altitude or 0.0, reference_length,
        scaling.kinematic_viscosity, re,
    )
    if scaling.compressible:
        log.warning(
            "M = %.2f is above the incompressible limit of %g: this solver has no "
            "density equation at all, so the result approximates a different flow "
            "rather than the same one slightly less well",
            scaling.mach, INCOMPRESSIBLE_MACH_LIMIT,
        )
    return re


def build(re: float | None = None, nx: int = 256, ny: int = 128,
          t_end: float | None = None, dt: float = 0.02, cfl_max: float = 0.4,
          domain_length: float = DOMAIN_LENGTH, domain_height: float = DOMAIN_HEIGHT,
          cylinder_x: float = CYLINDER_X, obstacle=None,
          outlet_type: str | None = None, p_ref: float | None = None,
          l_ref: float | None = None, wind_speed: float | None = None,
          altitude: float | None = None, **overrides) -> Simulation:
    """Construct the cylinder simulation without running it.

    ``domain_height`` controls the blockage ratio ``D / H``, which has a strong
    effect on the drag coefficient: confining walls accelerate the flow past the
    body and raise Cd well above its unbounded value.

    Passing ``obstacle`` replaces the cylinder with any other
    :class:`~pycfd.geometry.obstacles.Obstacle`, which is how a custom 2D shape
    is put into this same uniform-inflow / outflow configuration.  The Reynolds
    number is then formed with that body's characteristic length.

    ``outlet_type`` and ``p_ref`` retype the downstream boundary; leaving them
    ``None`` keeps this case's own choice of a pressure outlet at
    :data:`OUTLET_PRESSURE`.

    Reference length and speed
    --------------------------
    ``l_ref`` overrides the length the Reynolds number and the force
    coefficients are formed with.  A body loaded from a file defaults to its
    extent across the flow, which is the cylinder-diameter convention; an
    aircraft is conventionally reported against its overall *length* instead,
    and only the caller knows which one is meant.  The body's geometric span is
    still what the blockage ratio and the cells-across-the-body warning use, so
    overriding this changes what the numbers are normalised by and nothing else.

    ``wind_speed`` (m/s) and ``altitude`` (m) derive the Reynolds number from
    real conditions instead: ``Re = V L / nu`` with ``nu`` taken from the ISA,
    treating the geometry's length unit as the metre.  The solver itself stays
    non-dimensional -- ``u_ref`` remains 1 -- so use :func:`flight_scaling` to
    put the results back into m/s, Pa and seconds.
    """
    from . import override_outlet
    from ..core.mesh import StructuredMesh

    # A benchmark has to pin the physics it validates.  ``use_les`` is a package
    # default that legitimately changes with whatever case is being worked on,
    # and an eddy-viscosity model silently switched on turns an exact laminar
    # solution into something else -- it cost this benchmark its second-order
    # convergence once already.  ``setdefault`` keeps ``--les`` working.
    overrides.setdefault("use_les", False)

    # The mesh is built before the configuration because the body has to exist
    # before its size is known, and its size is what --l-ref overrides and
    # --wind-speed turns into a Reynolds number.
    mesh = StructuredMesh(nx, ny, domain_length, domain_height)
    cy = domain_height / 2.0
    if obstacle is None:
        obstacle = circle_mask(mesh, (cylinder_x, cy), DIAMETER / 2.0, name="cylinder")
    elif obstacle.mask.shape != mesh.shape:
        raise ValueError(
            f"the supplied obstacle was built on a {obstacle.mask.shape} grid but "
            f"the simulation mesh is {mesh.shape}; build both from the same mesh"
        )

    # Reynolds number and force coefficients follow the body's own length scale
    # unless the caller names a different convention.
    if l_ref is not None and l_ref <= 0:
        raise ValueError(f"l_ref must be positive, got {l_ref}")
    reference_length = (obstacle.characteristic_length if l_ref is None
                        else float(l_ref))
    re = _resolve_reynolds(re, wind_speed, altitude, reference_length)

    span = obstacle.characteristic_length
    if mesh.dy > span / 8.0:
        log.warning(
            "only %.1f cells span the body; forces will be crude "
            "(16 or more is recommended)", span / mesh.dy,
        )

    overrides.setdefault("name", f"cylinder_Re{re:g}")

    # Steady wakes settle quickly; shedding needs many periods to become periodic.
    default_t_end = 40.0 if re < 50 else 160.0

    cfg = SimulationConfig(
        nx=nx, ny=ny, lx=domain_length, ly=domain_height,
        re=re, u_ref=U_INF, l_ref=reference_length,
        dt=dt, t_end=default_t_end if t_end is None else t_end,
        cfl_max=cfl_max,
        boundary_config=override_outlet({
            "left": BCSpec(BCKind.INLET, velocity=U_INF, profile="uniform"),
            # Pressure outlet: anchors p = 0 downstream instead of letting the
            # level float, which is the physically meaningful far-field
            # condition for external flow and makes the reported pressure -- and
            # hence the pressure part of the drag -- absolute rather than
            # relative to an arbitrary reference cell.
            "right": BCSpec(BCKind.PRESSURE_OUTLET, p_ref=OUTLET_PRESSURE),
            "bottom": BCSpec(BCKind.SYMMETRY),
            "top": BCSpec(BCKind.SYMMETRY),
        }, outlet_type, p_ref),
        **overrides,
    )

    # The perturbation is handed over as part of the initial condition rather
    # than added afterwards, so that Simulation's initial projection removes its
    # divergence.  Injecting it into an already-projected field leaves the
    # start-up divergent, and the Runge-Kutta stages decay that only by a factor
    # of three per step -- tens of steps of contaminated transient.
    v_init = _wake_perturbation(mesh, cylinder_x, cy) if re >= 50 else None
    return Simulation(cfg, obstacle=obstacle, u_init=U_INF, v_init=v_init)


def build_from_geometry(path, re: float | None = None, nx: int = 256, ny: int = 128,
                        scale: float = 1.0, rotate_deg: float = 0.0,
                        center: tuple[float, float] | None = None,
                        threshold: float = 0.5, invert: bool = False,
                        **kwargs) -> Simulation:
    """External flow past a custom 2D body loaded from a file.

    ``path`` may be a vertex list (``.csv``/``.txt``/``.dat``) or a bitmap
    silhouette (``.png``/``.jpg``).  Outlines are scaled, rotated and placed at
    ``center`` -- one quarter of the way along the domain by default, leaving
    the rest for the wake.  Bitmaps are stretched across the whole domain, so
    they carry their own placement and ignore the transform arguments.

    ``l_ref``, ``wind_speed`` and ``altitude`` pass through to :func:`build`;
    they are the flags that matter most here, since a body from a file is
    exactly the case where the default reference length is a guess.
    """
    from ..geometry.obstacles import (
        load_polygon,
        mask_from_image,
        polygon_mask,
        transform_polygon,
    )
    from ..core.mesh import StructuredMesh

    path = Path(path)
    domain_length = kwargs.get("domain_length", DOMAIN_LENGTH)
    domain_height = kwargs.get("domain_height", DOMAIN_HEIGHT)
    mesh = StructuredMesh(nx, ny, domain_length, domain_height)

    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"):
        obstacle = mask_from_image(mesh, path, threshold=threshold, invert=invert,
                                   name=path.stem)
    else:
        verts = load_polygon(path)
        if center is None:
            center = (0.25 * domain_length, 0.5 * domain_height)
        obstacle = polygon_mask(
            mesh, transform_polygon(verts, scale, center, rotate_deg), name=path.stem
        )
    log.info("loaded %s: %s", path.name, obstacle)
    return build(re=re, nx=nx, ny=ny, obstacle=obstacle, **kwargs)


def _wake_perturbation(mesh, cylinder_x: float, cy: float) -> np.ndarray:
    """Transverse pulse behind the body, shaped like the ``v`` face layout.

    Breaks the symmetry of the wake so that shedding develops on a sensible time
    scale instead of waiting for round-off to do it.
    """
    Xv, Yv = mesh.v_grid()
    # A compact blob one diameter downstream, offset above the centreline.
    blob = np.exp(
        -(((Xv - (cylinder_x + DIAMETER)) / DIAMETER) ** 2
          + ((Yv - cy) / (0.5 * DIAMETER)) ** 2)
    )
    return PERTURBATION_AMPLITUDE * U_INF * blob


def run(re: float | None = None, nx: int = 256, ny: int = 128,
        t_end: float | None = None,
        dt: float = 0.02, outdir: str | Path = "results/cylinder",
        make_plots: bool = True, progress: bool = False, geometry=None,
        geometry_scale: float = 1.0, geometry_rotate: float = 0.0,
        outlet_type: str | None = None, p_ref: float | None = None,
        l_ref: float | None = None, wind_speed: float | None = None,
        altitude: float | None = None, transient: float | None = None,
        **overrides):
    """Run the cylinder case, measure Cd/St and write the figures.

    Passing ``geometry`` (a path to a vertex file or bitmap) substitutes a
    custom 2D body for the circular cylinder.  The published cylinder reference
    values are then not applicable and are omitted from the report.

    ``l_ref``, ``wind_speed`` and ``altitude`` are described in :func:`build`.
    When flight conditions are given, the report also carries the dimensional
    facts they imply -- speed, altitude, Mach number and the free-stream dynamic
    pressure -- so the run records what it was actually a simulation *of*.

    ``transient`` is the fraction of the record discarded before the force
    coefficients are averaged, defaulting to :data:`TRANSIENT_FRACTION`.  The
    mean that comes back carries a confidence band built from the number of
    *independent* samples rather than the number of writes, and a check on
    whether the retained window was actually stationary -- see
    :mod:`pycfd.analysis.timeseries`.  Raise it when a run started far from its
    eventual state.
    """
    from . import CaseResult

    scales = dict(l_ref=l_ref, wind_speed=wind_speed, altitude=altitude)
    if geometry is not None:
        sim = build_from_geometry(geometry, re=re, nx=nx, ny=ny,
                                  scale=geometry_scale, rotate_deg=geometry_rotate,
                                  t_end=t_end, dt=dt, outlet_type=outlet_type,
                                  p_ref=p_ref, **scales, **overrides)
    else:
        sim = build(re=re, nx=nx, ny=ny, t_end=t_end, dt=dt,
                    outlet_type=outlet_type, p_ref=p_ref, **scales, **overrides)

    # Everything normalised by a length uses the one the configuration settled
    # on, which is the body's own span unless --l-ref named another convention.
    # The Reynolds number likewise comes back from the configuration, since
    # --wind-speed derives it rather than being handed it.
    reference_length = sim.config.l_ref
    re = sim.config.re

    # Record the force history and a wake probe so Cd and St can be extracted.
    history: dict[str, list[float]] = {"t": [], "cd": [], "cl": []}
    probe = Probe(sim.mesh.lx * 0.5, sim.mesh.ly * 0.5 + 0.25 * DIAMETER, "wake")
    sim.probes.append(probe)

    def record(fields, info):
        cd, cl = force_coefficients(sim.solver.body_force_reaction, U_INF,
                                    reference_length)
        history["t"].append(fields.t)
        history["cd"].append(cd)
        history["cl"].append(cl)

    # Turn off the start-up pulse after its short window by simply letting it
    # advect away -- it is an initial condition, not a forcing term.
    result = sim.run(callback=record, callback_every=10, progress=progress)

    t = np.asarray(history["t"])
    cd = np.asarray(history["cd"])
    cl = np.asarray(history["cl"])
    transient_fraction = TRANSIENT_FRACTION if transient is None else float(transient)
    settled = (t > transient_fraction * t[-1] if t.size
               else np.array([], dtype=bool))

    # The mean of a shedding wake is a mean of correlated samples, so its
    # uncertainty follows the number of independent observations rather than
    # the number of times the callback fired.
    cd_average = time_average(t, cd, transient_fraction)

    metrics: dict[str, float] = {
        "cd_mean": cd_average.mean,
        "cd_uncertainty": cd_average.uncertainty,
        "cd_final": float(cd[-1]) if cd.size else float("nan"),
        "cl_rms": float(np.sqrt(np.mean(cl[settled] ** 2))) if settled.any() else float("nan"),
        # The body's own extent across the flow: what the grid has to resolve
        # and what blocks the channel, whatever the coefficients are divided by.
        "characteristic_length": sim.obstacle.characteristic_length,
        "cells_across_body": sim.obstacle.characteristic_length / sim.mesh.dy,
        "blockage_ratio": sim.obstacle.characteristic_length / sim.mesh.ly,
        # The length Re, Cd and St were actually formed with -- the same number
        # unless --l-ref asked for a different convention.
        "reference_length": reference_length,
        "reynolds": re,
        # What the average is actually built on: how much of the record was
        # kept, and how many independent observations that amounted to.
        "transient_fraction": transient_fraction,
        "averaging_samples": cd_average.n_samples,
        "effective_samples": cd_average.effective_samples,
        "autocorrelation_time": cd_average.autocorrelation_time,
        "cd_drift_z": cd_average.drift_z,
        "steps": result.steps,
        "final_time": result.time,
        "wall_time_s": result.wall_time,
        "max_divergence": sim.solver.max_divergence(sim.fields),
    }

    if wind_speed is not None:
        # What the run is a simulation *of*, recorded next to what it measured.
        scaling = flight_scaling(wind_speed, altitude)
        metrics["wind_speed_m_s"] = float(wind_speed)
        metrics["altitude_m"] = float(altitude or 0.0)
        metrics["kinematic_viscosity"] = scaling.kinematic_viscosity
        metrics["mach"] = scaling.mach
        metrics["dynamic_pressure_pa"] = scaling.dynamic_pressure

    checks: list[tuple[str, bool, str]] = []
    key = int(round(re))
    if geometry is not None:
        # A custom body has no published reference; report the measurement and
        # check only that the run produced a usable force signal.
        checks.append((
            "force measured on the custom body", bool(np.isfinite(metrics["cd_mean"])),
            f"Cd = {metrics['cd_mean']:.3f}, Cl_rms = {metrics['cl_rms']:.3f} "
            f"(no published reference for this geometry)",
        ))
    elif key in REFERENCE_CD:
        lo, hi = REFERENCE_CD[key]
        cd_mean = metrics["cd_mean"]
        # Blockage and a first-order immersed boundary both bias Cd upward, so a
        # generous band is used and the literature range is always reported.
        checks.append((
            f"Cd at Re={key}", bool(lo * 0.7 <= cd_mean <= hi * 1.4),
            # The band is formatted with a significant-figure spec rather than a
            # fixed one: a well-converged run's uncertainty is small enough that
            # three decimals round it to "0.000", which reads as no band at all.
            f"Cd = {cd_mean:.3f} ± {cd_average.uncertainty:.3g}; "
            f"unbounded literature range {lo}-{hi}",
        ))

    # A mean taken over a window that had not settled is a number with no
    # meaning, however tight its error bar looks -- so it is checked, not just
    # reported. Only worth asking once there is a record to ask it of.
    if cd_average.n_samples >= 4:
        checks.append((
            "force average is stationary", cd_average.stationary,
            f"the two halves of the averaging window differ by "
            f"{cd_average.drift_z:.1f} standard errors"
            + ("" if cd_average.stationary else
               f"; raise --transient above {transient_fraction:g} or run longer"),
        ))

    if re >= 50 and geometry is None:
        series = probe.as_arrays()
        st = strouhal_number(series["t"], series["v"], reference_length, U_INF)
        metrics["strouhal"] = st
        metrics["cl_peak_to_peak"] = (
            float(cl[settled].max() - cl[settled].min()) if settled.any() else float("nan")
        )
        shedding = bool(metrics["cl_peak_to_peak"] > 0.05)
        checks.append((
            "vortex shedding developed", shedding,
            f"lift oscillation peak-to-peak = {metrics['cl_peak_to_peak']:.3f}, St = {st:.4f}"
            + (f"; literature {REFERENCE_ST[key][0]}-{REFERENCE_ST[key][1]}"
               if key in REFERENCE_ST else ""),
        ))

    outputs: list[Path] = []
    if make_plots:
        from ..visualization import static_plot as sp

        outdir = Path(outdir)
        body = sim.obstacle.name
        tag = f"{body}_Re{re:g}_{nx}x{ny}"
        p1 = outdir / f"{tag}_fields.png"
        sp.four_panel_figure(
            sim.fields, solid=sim.solid_mask,
            title=f"Flow past {body}, Re = {re:g}, {nx}x{ny}", path=p1,
        )
        p2 = outdir / f"{tag}_forces.png"
        sp.time_series_figure(
            {"t": list(t), "Cd": list(cd), "Cl": list(cl)},
            keys=("Cd", "Cl"),
            title=f"Force coefficients, Re = {re:g}", path=p2,
        )
        outputs += [p1, p2]

    label = sim.obstacle.name if geometry is not None else "Cylinder"
    return CaseResult(f"{label} (Re={re:g}, {nx}x{ny})", sim, metrics, outputs, checks)
