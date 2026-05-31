from collections import Counter
from pathlib import Path

from places_attr_conflation.claim_extraction import extract_claims_from_replay_episode
from places_attr_conflation.replay import load_replay_corpus


ROOT = Path(__file__).resolve().parents[1]


def test_raw_collected_replay_tree_is_website_heavy_and_has_no_phone_or_address_pages() -> None:
    reports_root = ROOT / "reports" / "replay_collected"
    attr_counts: Counter[str] = Counter()
    pages_with_attr: Counter[str] = Counter()
    claim_eps: Counter[str] = Counter()

    for path in reports_root.rglob("*.json"):
        try:
            corpus = load_replay_corpus(path)
        except Exception:
            continue
        for episode in corpus:
            attr_counts[episode.attribute] += 1
            pages_total = sum(len(attempt.fetched_pages) for attempt in episode.search_attempts)
            if pages_total:
                pages_with_attr[episode.attribute] += 1
            if extract_claims_from_replay_episode(episode):
                claim_eps[episode.attribute] += 1

    assert attr_counts["phone"] == 7231
    assert attr_counts["address"] == 6664
    assert attr_counts["website"] == 8367
    assert pages_with_attr["phone"] == 0
    assert pages_with_attr["address"] == 0
    assert pages_with_attr["website"] == 2670
    assert claim_eps["phone"] == 0
    assert claim_eps["address"] == 0


def test_promoted_contact_slice_is_still_the_useful_phone_address_surface() -> None:
    path = ROOT / "tests" / "fixtures" / "pac_contact_replay.json"
    corpus = load_replay_corpus(path)

    assert len(corpus) == 70
    assert sum(1 for episode in corpus if episode.attribute == "phone") == 42
    assert sum(1 for episode in corpus if episode.attribute == "address") == 28
    assert sum(1 for episode in corpus if extract_claims_from_replay_episode(episode)) == 62


def test_merged_harness_replay_is_the_low_claim_coverage_bottleneck() -> None:
    merged_full = load_replay_corpus(ROOT / "reports" / "harness" / "mlattributes_replay_merged_full.json")
    merged_unique = load_replay_corpus(ROOT / "reports" / "harness" / "mlattributes_replay_merged_unique.json")

    assert len(merged_full) == 5078
    assert len(merged_unique) == 5078
    assert sum(1 for episode in merged_full if extract_claims_from_replay_episode(episode)) == 386
    assert sum(1 for episode in merged_unique if extract_claims_from_replay_episode(episode)) == 104
    assert sum(
        sum(len(attempt.fetched_pages) for attempt in episode.search_attempts)
        for episode in merged_full
        if episode.attribute == "phone"
    ) == 0
    assert sum(
        sum(len(attempt.fetched_pages) for attempt in episode.search_attempts)
        for episode in merged_full
        if episode.attribute == "address"
    ) == 0
