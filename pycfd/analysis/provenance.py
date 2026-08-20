"""Provenance: recording what produced a given output file.

A results directory accumulates figures, VTK dumps and checkpoints that look
alike and are impossible to tell apart a week later.  Every exporter in
:mod:`pycfd.analysis.export` and every figure written by
:mod:`pycfd.visualization.static_plot` therefore carries a record of the run
that produced it: the version, the exact command, the configuration, and the
commit the code was at.

Two channels are used, because no single one fits every format:

*Native embedding*, where the format has somewhere to put it -- the title line
of a legacy VTK file, a field inside an ``.npz``, PNG ``tEXt`` chunks.  This
travels with the file itself and survives being copied around.

*A sidecar* ``<name>.provenance.json`` beside the output, which holds the full
record including the complete configuration.  Native slots are small or
awkwardly typed; the sidecar is the machine-readable copy.  CSV is deliberately
left pure -- a ``#`` banner would break naive readers -- so for that format the
sidecar is the only channel.
"""

from __future__ import annotations

import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

#: Seconds to wait on ``git`` before giving up.  Provenance is a nicety; it must
#: never be the reason an export hangs.
_GIT_TIMEOUT = 2.0

#: Longest provenance string written into a legacy VTK title line, which the
#: format caps at 256 characters.
VTK_TITLE_LIMIT = 250


def git_commit(repo: Path | None = None) -> str | None:
    """Short commit hash of the working tree, or ``None`` outside a repository.

    A ``+`` is appended when the tree has uncommitted changes, because a hash
    alone would otherwise imply a reproducibility it does not have.
    """
    root = Path(repo) if repo is not None else Path(__file__).resolve().parent
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=_GIT_TIMEOUT,
        )
        if rev.returncode != 0:
            return None
        commit = rev.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, timeout=_GIT_TIMEOUT,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "+"
        return commit
    except (OSError, subprocess.SubprocessError):
        return None


def invocation(argv: list[str] | None = None) -> str:
    """The command that started this process, as something you could paste back.

    ``sys.argv[0]`` is an absolute path to ``main.py``; when that is what ran,
    it is rewritten to the ``python -m pycfd.main`` form actually used to launch
    it, so the recorded command is runnable rather than merely descriptive.
    """
    argv = list(sys.argv if argv is None else argv)
    if not argv:
        return ""
    head, rest = argv[0], argv[1:]
    if Path(head).name in ("main.py", "__main__.py"):
        return shlex.join(["python", "-m", "pycfd.main", *rest])
    return shlex.join(argv)


def provenance_record(config=None, extra: dict | None = None,
                      argv: list[str] | None = None) -> dict:
    """Assemble the record describing this run.

    Parameters
    ----------
    config:
        A :class:`~pycfd.config.SimulationConfig`; its full serialised form is
        included so a result can be reproduced from the file alone.
    extra:
        Any case-level facts worth keeping alongside it (obstacle name, measured
        coefficients).
    """
    from .. import __version__

    record = {
        "pycfd_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": invocation(argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    commit = git_commit()
    if commit is not None:
        record["git_commit"] = commit
    if config is not None:
        record["config"] = config.to_dict()
    if extra:
        record["extra"] = dict(extra)
    return record


def summary_line(record: dict, limit: int = VTK_TITLE_LIMIT) -> str:
    """One-line digest for formats with a single free-text slot.

    Truncated to ``limit`` characters; the sidecar always holds the full record.
    """
    parts = [
        f"pycfd {record.get('pycfd_version', '?')}",
        record.get("generated_utc", ""),
    ]
    if "git_commit" in record:
        parts.append(f"git {record['git_commit']}")
    if record.get("command"):
        parts.append(record["command"])
    line = " | ".join(p for p in parts if p)
    return line[:limit]


def write_sidecar(path: str | Path, record: dict) -> Path:
    """Write ``<path>.provenance.json`` beside an output file.

    The suffix is replaced rather than appended, so ``field.vtk`` yields
    ``field.provenance.json`` -- one sidecar per output, not per extension.
    """
    path = Path(path)
    sidecar = path.with_suffix(".provenance.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return sidecar


def png_metadata(record: dict) -> dict[str, str]:
    """Provenance mapped onto the standard PNG ``tEXt`` keys.

    ``Software``, ``Creation Time`` and ``Description`` are registered PNG
    keywords, so ``exiftool`` and image viewers surface them without knowing
    anything about pycfd.
    """
    meta = {
        "Software": f"pycfd {record.get('pycfd_version', '?')}",
        "Creation Time": record.get("generated_utc", ""),
        "Description": summary_line(record, limit=1024),
    }
    if record.get("command"):
        meta["Comment"] = record["command"]
    return {k: v for k, v in meta.items() if v}
