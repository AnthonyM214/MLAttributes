"""Bridge ProjectTerra seed data into an attribute-conflation corpus.

This module connects the existing ProjectTerra data-filtering work to the
MLAttributes dorking/evidence pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .dorking import build_multi_layer_plan

DEFAULT_ATTRIBUTES = ("name", "address", "phone", "website", "category")


@dataclass(frozen=True)
class SeedPlace:
    place_id: str
    source_dataset: str
    source_record_id: str
    name: str = ""
    canonical_name: str = ""
    address: str = ""
    city: str = ""
    region: str = ""
    postal_code: str = ""
    country: str = ""
    phone: str = ""
    website: str = ""
    category: str = ""


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def load_name_mapping(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping_path = Path(path)
    if not mapping_path.exists():
        return {}
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for entry in payload:
        chain = _clean(entry.get("chain"))
        for variation in entry.get("variations", []):
            if isinstance(variation, dict):
                name = _clean(variation.get("name"))
            else:
                name = _clean(variation)
            if chain and name:
                mapping[name] = chain
    return mapping


def build_evidence_manifest(seed: SeedPlace) -> list[dict[str, object]]:
    place = {
        "name": seed.canonical_name or seed.name,
        "city": seed.city,
        "region": seed.region,
        "address": seed.address,
        "phone": seed.phone,
        "website": seed.website,
    }
    rows: list[dict[str, object]] = []
    for attribute in DEFAULT_ATTRIBUTES:
        value = _clean(getattr(seed, attribute, ""))
        if not value:
            continue
        plan = build_multi_layer_plan(place, attribute)
        for layer in plan.layers:
            for query in layer.queries:
                rows.append(
                    {
                        "place_id": seed.place_id,
                        "attribute": attribute,
                        "candidate_value": value,
                        "query_layer": layer.name,
                        "query": query,
                        "preferred_sources": layer.preferred_sources,
                    }
                )
    return rows


def write_jsonl(rows: list[dict[str, object]], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build corpus manifests from ProjectTerra seed data.")
    parser.add_argument("--output-dir", default="data/corpus")
    args = parser.parse_args(argv)

    demo_seed = SeedPlace(
        place_id="demo:1",
        source_dataset="demo",
        source_record_id="1",
        name="Joe's Pizza",
        city="Santa Cruz",
        region="CA",
        website="https://joespizza.example",
        category="restaurant",
    )

    manifest = build_evidence_manifest(demo_seed)
    output_path = write_jsonl(manifest, Path(args.output_dir) / "evidence_manifest.jsonl")
    print(json.dumps({"rows": len(manifest), "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
