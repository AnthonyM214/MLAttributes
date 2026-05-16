from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReplayEvidenceRecord:
    place_id: str
    attribute: str
    query: str
    url: str
    source_type: str
    extracted_value: str
    weak_label: str
    confidence: float


@dataclass(frozen=True)
class ReplayEpisode:
    episode_id: str
    records: list[ReplayEvidenceRecord]
    metadata: dict[str, Any]


REPLAY_DIRNAME = 'replay'


def save_episode(episode: ReplayEpisode, root: str | Path = 'data/replay') -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    output_path = root_path / f'{episode.episode_id}.json'

    payload = {
        'episode_id': episode.episode_id,
        'metadata': episode.metadata,
        'records': [asdict(record) for record in episode.records],
    }

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    return output_path


def load_episode(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))
