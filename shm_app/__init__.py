"""Modernized SeismicHandler Motif (SHM) & Command Interpreter (SHC) package."""

import os
import tempfile

# Ensure Matplotlib uses a writable config directory (supports headless / sandboxed runs)
if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = os.path.join(tempfile.gettempdir(), f"shm_mpl_{os.getuid()}" if hasattr(os, "getuid") else "shm_mpl")
    os.environ["MPLCONFIGDIR"] = _mpl_dir
    os.makedirs(_mpl_dir, exist_ok=True)


def resolve_default_shm_root() -> str:
    """Resolve the SHM reference root directory containing inputs/, filter/, and data-examples/."""
    env_root = os.environ.get("SHM_ROOT", "").strip()
    if env_root:
        return os.path.expanduser(env_root)

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(pkg_dir)
    workspace_root = os.path.dirname(repo_root)
    candidates = (
        os.path.join(repo_root, "sh"),
        os.path.join(workspace_root, "SHM", "sh"),
        os.path.join(workspace_root, "sh"),
        "/usr/local/sh",
    )
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return os.path.join(repo_root, "sh")


__version__ = "3.1.0"
SHM_DEFAULT_ROOT = resolve_default_shm_root()
