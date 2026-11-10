"""Where RegLens keeps its data, configs and eval files.

Before this module every file used Path(__file__).parents[2], which only works
for an editable install from a git checkout. After a normal `pip install .`
that path points inside site-packages and nothing is found.

Resolution order:
1. REGLENS_HOME environment variable, if set
2. the source checkout, if we are running from one (pyproject.toml next to src/)
3. the current working directory
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.getenv("REGLENS_HOME")
    if env:
        return Path(env).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").exists():
        return checkout
    return Path.cwd()


ROOT = project_root()
