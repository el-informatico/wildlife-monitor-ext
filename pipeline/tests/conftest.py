"""Offline test bootstrap.

The pyproject pythonpath already puts the repo root on sys.path; this is
belt and braces for runs that bypass pytest configuration (e.g. a bare
``python -m pytest pipeline/tests`` from an unrelated rootdir).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
