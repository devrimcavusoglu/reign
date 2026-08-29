"""REIGN: Refurbished Embeddings with Integrated Guidance Networks.

Also defines the filesystem layout used by the training / evaluation entry
points. Every directory is resolved lazily from environment variables so the
package works both from a source checkout and from an installed distribution:

* ``REIGN_HOME``      — project root (default: the repo root when running from
  a checkout, otherwise the current working directory)
* ``REIGN_DATA_DIR``  — locally built datasets           (default: ``$REIGN_HOME/data``)
* ``REIGN_MODEL_DIR`` — checkpoints written by training  (default: ``$REIGN_HOME/models``)
"""

import os
from pathlib import Path

from reign.modeling import ReignConfig, ReignModel

__version__ = "1.0.0"

SOURCES_ROOT = Path(__file__).resolve().parent


def _default_project_root() -> Path:
    """Repo root when imported from a source checkout, else the working dir."""
    candidate = SOURCES_ROOT.parent
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


PROJECT_ROOT = Path(os.environ.get("REIGN_HOME") or _default_project_root())
DATA_DIR = Path(os.environ.get("REIGN_DATA_DIR") or PROJECT_ROOT / "data")
MODEL_DIR = Path(os.environ.get("REIGN_MODEL_DIR") or PROJECT_ROOT / "models")

__all__ = [
    "ReignConfig",
    "ReignModel",
    "SOURCES_ROOT",
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODEL_DIR",
    "__version__",
]
