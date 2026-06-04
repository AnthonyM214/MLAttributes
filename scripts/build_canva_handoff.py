#!/usr/bin/env python3
"""Build a Canva-friendly handoff bundle for the PAC presentation deck.

The bundle keeps the editable PPTX as the primary artifact and emits a
separate markdown outline with slide text and speaker notes, because Canva's
documented PPTX import path does not guarantee round-tripping PowerPoint notes.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from zipfile import ZIP_DEFLATED, ZipFile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PPTX = REPO_ROOT.parent / "Places Attribute Conflation.pptx"
DEFAULT_SOURCE_MD = REPO_ROOT / "docs" / "presentations" / "MLAttributes_ProjectTerra_PAC.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "release" / "canva_handoff"


@dataclass(frozen=True)
class SlideBlock:
    index: int
    body_lines: list[str]
    notes_lines: list[str]


def strip_frontmatter(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1 :])
    return text


def split_slides(markdown: str) -> list[str]:
    markdown = strip_frontmatter(markdown).strip()
    return [chunk.strip() for chunk in re.split(r"(?m)^---\s*$", markdown) if chunk.strip()]


def extract_comment_blocks(slide_text: str) -> list[str]:
    return [re.sub(r"\n{3,}", "\n\n", block.strip()) for block in re.findall(r"<!--(.*?)-->", slide_text, flags=re.S)]


def body_lines(slide_text: str) -> list[str]:
    cleaned = re.sub(r"<!--.*?-->", "", slide_text, flags=re.S)
    lines: list[str] = []
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line:
            continue
        lines.append(line)
    return lines


def build_slide_blocks(markdown: str) -> list[SlideBlock]:
    blocks: list[SlideBlock] = []
    for index, slide_text in enumerate(split_slides(markdown), start=1):
        blocks.append(
            SlideBlock(
                index=index,
                body_lines=body_lines(slide_text),
                notes_lines=[line.strip() for block in extract_comment_blocks(slide_text) for line in block.splitlines() if line.strip()],
            )
        )
    return blocks


def outline_title(lines: Iterable[str]) -> str:
    headings = [line.lstrip("# ").strip() for line in lines if line.startswith("#")]
    if headings:
        return " / ".join(headings[:3])
    if lines:
        return lines[0]
    return "(untitled slide)"


def write_markdown_outline(blocks: list[SlideBlock], out_path: Path) -> None:
    lines: list[str] = [
        "# Canva handoff outline",
        "",
        "This file pairs the PPTX with a concise slide-by-slide outline and speaker notes.",
        "Use the PPTX as the editable source of truth and this outline as the presenter handoff.",
        "",
    ]
    for block in blocks:
        lines.extend(
            [
                f"## Slide {block.index}: {outline_title(block.body_lines)}",
                "",
                "### Slide text",
            ]
        )
        for line in block.body_lines:
            if line.startswith("<!--") or line.startswith("-->"):
                continue
            lines.append(f"- {line}")
        lines.extend(["", "### Speaker notes"])
        if block.notes_lines:
            for line in block.notes_lines:
                lines.append(f"- {line}")
        else:
            lines.append("- (none)")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_import_guide(out_path: Path, pptx_name: str, outline_name: str) -> None:
    out_path.write_text(
        "\n".join(
            [
                "# Canva import flow",
                "",
                "1. Import the PPTX into Canva: upload or drag in `"
                + pptx_name
                + "`.",
                "2. Keep the markdown outline beside it: `"
                + outline_name
                + "`.",
                "3. Review the slide text in Canva and use the outline for the speaker notes / talk track.",
                "4. After edits, export from Canva as PPTX for round-trip editing or PDF for distribution.",
                "",
                "Notes:",
                "- Canva's documented import path supports PPTX and PDF.",
                "- PowerPoint speaker notes are not part of the documented Canva import contract, so keep notes in the companion markdown file.",
            ]
        ),
        encoding="utf-8",
    )


def write_placeholder_pptx(out_path: Path) -> None:
    """Write a minimal PPTX so CI can build the handoff without local slide files."""

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
</Types>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>
"""
    presentation = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
"""
    presentation_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>
"""
    slide = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld>
</p:sld>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("ppt/presentation.xml", presentation)
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels)
        zf.writestr("ppt/slides/slide1.xml", slide)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pptx", type=Path, default=DEFAULT_SOURCE_PPTX)
    parser.add_argument("--source-markdown", type=Path, default=DEFAULT_SOURCE_MD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.source_markdown.exists():
        raise SystemExit(f"missing markdown source: {args.source_markdown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pptx_dest = args.output_dir / args.source_pptx.name
    outline_dest = args.output_dir / "speaker_notes_outline.md"
    guide_dest = args.output_dir / "CANVA_IMPORT_FLOW.md"
    manifest_dest = args.output_dir / "manifest.json"

    if args.source_pptx.exists():
        shutil.copy2(args.source_pptx, pptx_dest)
    else:
        write_placeholder_pptx(pptx_dest)
    markdown = args.source_markdown.read_text(encoding="utf-8")
    blocks = build_slide_blocks(markdown)
    write_markdown_outline(blocks, outline_dest)
    write_import_guide(guide_dest, pptx_dest.name, outline_dest.name)

    manifest = {
        "source_pptx": str(args.source_pptx),
        "source_markdown": str(args.source_markdown),
        "output_dir": str(args.output_dir),
        "artifacts": [pptx_dest.name, outline_dest.name, guide_dest.name],
        "slides": len(blocks),
    }
    manifest_dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Canva handoff bundle to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
