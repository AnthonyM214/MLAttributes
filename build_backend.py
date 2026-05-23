"""Minimal PEP 517 backend for offline wheels and editable installs."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
import tarfile
import zipfile


NAME = "places-attribute-conflation"
DIST_INFO = "places_attribute_conflation-0.1.0.dist-info"
WHEEL_NAME = "places_attribute_conflation-0.1.0-py3-none-any.whl"
SDIST_NAME = "places-attribute-conflation-0.1.0.tar.gz"
VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src" / "places_attr_conflation"
SRC_PATH = ROOT / "src"


def _metadata_text() -> str:
    return "\n".join(
        [
            "Metadata-Version: 2.1",
            f"Name: {NAME}",
            f"Version: {VERSION}",
            "Summary: Evidence-backed resolver scaffold for Overture Places attribute conflation.",
            "Requires-Python: >=3.11",
            "Requires-Dist: duckdb>=1.0",
            "Requires-Dist: numpy>=1.26",
            "Requires-Dist: pandas>=2.0",
            "Requires-Dist: pyarrow>=15.0",
            "Requires-Dist: scikit-learn>=1.4",
            "",
        ]
    )


def _wheel_text() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: build_backend",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _entry_points_text() -> str:
    return "\n".join(
        [
            "[console_scripts]",
            "mlattributes-eval = places_attr_conflation.cli:main",
            "pac-benchmark-v2 = places_attr_conflation.benchmark_v2:main",
            "pac-resolvepoi-selective = places_attr_conflation.resolvepoi_selective:main",
            "",
        ]
    )


def _record_line(path: str, data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{path},sha256={encoded},{len(data)}"


def _metadata_files() -> list[tuple[str, bytes]]:
    return [
        (f"{DIST_INFO}/METADATA", _metadata_text().encode("utf-8")),
        (f"{DIST_INFO}/WHEEL", _wheel_text().encode("utf-8")),
        (f"{DIST_INFO}/entry_points.txt", _entry_points_text().encode("utf-8")),
        (f"{DIST_INFO}/top_level.txt", b"places_attr_conflation\n"),
    ]


def _package_files() -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    for path in sorted(SRC_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".pyc" or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(SRC_PATH).as_posix()
        files.append((rel, path.read_bytes()))
    return files


def _write_wheel(archive_path: Path, *, editable: bool = False) -> None:
    records: list[str] = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if editable:
            pth_name = "places_attr_conflation.pth"
            pth_data = f"{SRC_PATH.as_posix()}\n".encode("utf-8")
            zf.writestr(pth_name, pth_data)
            records.append(_record_line(pth_name, pth_data))
        else:
            for rel, data in _package_files():
                zf.writestr(rel, data)
                records.append(_record_line(rel, data))

        for rel, data in _metadata_files():
            zf.writestr(rel, data)
            records.append(_record_line(rel, data))

        record_name = f"{DIST_INFO}/RECORD"
        record_buf = io.StringIO()
        writer = csv.writer(record_buf, lineterminator="\n")
        for line in records:
            writer.writerow(line.split(",", 2))
        writer.writerow([record_name, "", ""])
        record_data = record_buf.getvalue().encode("utf-8")
        zf.writestr(record_name, record_data)


def _build_archive(directory: str, filename: str, *, editable: bool = False) -> str:
    out = Path(directory) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_wheel(out, editable=editable)
    return filename


def get_requires_for_build_wheel(config_settings=None):  # noqa: D401
    return []


def get_requires_for_build_editable(config_settings=None):  # noqa: D401
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):  # noqa: D401
    dist_info = Path(metadata_directory) / DIST_INFO
    dist_info.mkdir(parents=True, exist_ok=True)
    for rel, data in _metadata_files():
        path = dist_info / Path(rel).name
        path.write_bytes(data)
    return DIST_INFO


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):  # noqa: D401
    return prepare_metadata_for_build_wheel(metadata_directory, config_settings=config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):  # noqa: D401
    return _build_archive(wheel_directory, WHEEL_NAME)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):  # noqa: D401
    return _build_archive(wheel_directory, WHEEL_NAME, editable=True)


def build_sdist(sdist_directory, config_settings=None):  # noqa: D401
    out = Path(sdist_directory) / SDIST_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tf:
        for rel in ["pyproject.toml", "build_backend.py", "README.md", "LICENSE"]:
            path = ROOT / rel
            if path.exists():
                tf.add(path, arcname=f"places-attribute-conflation-0.1.0/{rel}")
        for path in sorted((ROOT / "src").rglob("*")):
            if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            tf.add(path, arcname=f"places-attribute-conflation-0.1.0/{path.relative_to(ROOT).as_posix()}")
    return SDIST_NAME
