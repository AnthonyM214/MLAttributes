"""Test package bootstrap.

Make the src-layout package importable during unittest discovery without
requiring callers to set PYTHONPATH manually.
"""

from __future__ import annotations

import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

