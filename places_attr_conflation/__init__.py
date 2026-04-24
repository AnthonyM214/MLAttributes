"""Repository-local import shim for the src-layout package.

This lets local commands import `places_attr_conflation.*` from the repo root
without requiring installation or manual PYTHONPATH changes.
"""

from __future__ import annotations

import os


__path__ = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "places_attr_conflation"))
]
