from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
import unittest

import build_backend


class PackagingTests(unittest.TestCase):
    def test_build_wheel_includes_package_data_and_entry_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wheel_dir = Path(tmpdir) / "wheel"
            wheel_dir.mkdir()
            filename = build_backend.build_wheel(str(wheel_dir))
            wheel_path = wheel_dir / filename

            self.assertTrue(wheel_path.exists())
            with zipfile.ZipFile(wheel_path) as zf:
                names = set(zf.namelist())
                self.assertIn("places_attr_conflation/static/collector_index.html", names)
                self.assertIn("places_attribute_conflation-0.1.0.dist-info/entry_points.txt", names)
                self.assertIn("places_attribute_conflation-0.1.0.dist-info/METADATA", names)
                entry_points = zf.read("places_attribute_conflation-0.1.0.dist-info/entry_points.txt").decode("utf-8")
                self.assertIn("mlattributes-eval", entry_points)
                self.assertIn("pac-benchmark-v2", entry_points)

    def test_build_editable_creates_pth_for_src_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wheel_dir = Path(tmpdir) / "editable"
            wheel_dir.mkdir()
            filename = build_backend.build_editable(str(wheel_dir))
            wheel_path = wheel_dir / filename

            with zipfile.ZipFile(wheel_path) as zf:
                self.assertIn("places_attr_conflation.pth", zf.namelist())
                pth_text = zf.read("places_attr_conflation.pth").decode("utf-8")
                self.assertIn("/src", pth_text)


if __name__ == "__main__":
    unittest.main()
