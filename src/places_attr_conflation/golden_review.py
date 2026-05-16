from __future__ import annotations

import csv
from pathlib import Path


HIGH_PRIORITY_LABELS = {
    'contradicts',
    'stale',
    'unclear',
}


def prioritize_review_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    prioritized = sorted(
        rows,
        key=lambda row: (
            str(row.get('weak_label', '')) not in HIGH_PRIORITY_LABELS,
            -float(row.get('confidence', 0.0)),
        ),
    )
    return prioritized


def export_review_csv(rows: list[dict[str, object]], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text('', encoding='utf-8')
        return output_path

    ordered = prioritize_review_rows(rows)

    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0].keys()))
        writer.writeheader()
        writer.writerows(ordered)

    return output_path
