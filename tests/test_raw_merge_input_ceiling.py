from collections import Counter
from pathlib import Path
import unittest

from places_attr_conflation.claim_extraction import extract_claims_from_replay_episode
from places_attr_conflation.replay import load_replay_corpus


ROOT = Path(__file__).resolve().parents[1]


class RawMergeInputCeilingTest(unittest.TestCase):
    def test_raw_merge_input_phone_address_corpora_do_not_add_promotable_signal(self) -> None:
        merge_inputs_root = ROOT / "reports" / "replay_collected"
        if not merge_inputs_root.exists():
            self.skipTest("raw replay_collected merge-input tree is not checked in")

        attr_counts: Counter[str] = Counter()
        pages_with_attr: Counter[str] = Counter()
        claims_with_attr: Counter[str] = Counter()
        phone_address_files = 0

        for path in merge_inputs_root.rglob("*.json"):
            if "merge_inputs" not in path.as_posix():
                continue
            try:
                corpus = load_replay_corpus(path)
            except Exception:
                continue

            file_has_phone_or_address = False
            for episode in corpus:
                attr_counts[episode.attribute] += 1
                if episode.attribute in {"phone", "address"}:
                    file_has_phone_or_address = True
                    pages_total = sum(len(attempt.fetched_pages) for attempt in episode.search_attempts)
                    if pages_total:
                        pages_with_attr[episode.attribute] += 1
                    if extract_claims_from_replay_episode(episode):
                        claims_with_attr[episode.attribute] += 1

            if file_has_phone_or_address:
                phone_address_files += 1

        if phone_address_files == 0:
            self.skipTest("raw merge-input phone/address corpora are not checked in for this checkout")

        self.assertGreater(attr_counts["phone"], 0)
        self.assertGreater(attr_counts["address"], 0)
        self.assertEqual(pages_with_attr["phone"], 0)
        self.assertEqual(pages_with_attr["address"], 0)
        self.assertEqual(claims_with_attr["phone"], 0)
        self.assertEqual(claims_with_attr["address"], 0)
