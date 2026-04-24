"""Repository-local bootstrap for src-layout imports.

This lets `python3 -m unittest discover -s tests` work without requiring
callers to manually export PYTHONPATH=src.
"""

from __future__ import annotations

import os
import sys


ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

