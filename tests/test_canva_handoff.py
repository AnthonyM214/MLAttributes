import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase


class CanvaHandoffTest(TestCase):
    def test_canva_handoff_bundle_is_generated(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "build_canva_handoff.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            result = subprocess.run(
                [sys.executable, str(script), "--output-dir", str(outdir)],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Wrote Canva handoff bundle", result.stdout)

            manifest_path = outdir / "manifest.json"
            outline_path = outdir / "speaker_notes_outline.md"
            guide_path = outdir / "CANVA_IMPORT_FLOW.md"
            pptx_path = outdir / "Places Attribute Conflation.pptx"

            self.assertTrue(manifest_path.exists())
            self.assertTrue(outline_path.exists())
            self.assertTrue(guide_path.exists())
            self.assertTrue(pptx_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["slides"], 14)
            self.assertEqual(manifest["artifacts"], [
                "Places Attribute Conflation.pptx",
                "speaker_notes_outline.md",
                "CANVA_IMPORT_FLOW.md",
            ])
