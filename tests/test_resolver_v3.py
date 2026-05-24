from __future__ import annotations

import unittest
from pathlib import Path

from places_attr_conflation.benchmark_v3 import evaluate_benchmark_v3
from places_attr_conflation.claim_extraction import extract_claims_from_replay_episode
from places_attr_conflation.normalization import normalize_name, normalize_phone
from places_attr_conflation.replay import load_replay_corpus
from places_attr_conflation.resolver_v3 import resolve_attribute_v3_from_claims


ROOT = Path(__file__).resolve().parents[1]


class ResolverV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episodes = load_replay_corpus(ROOT / "tests" / "fixtures" / "hard_cases_replay.json")

    def _episode(self, case_id: str):
        return next(episode for episode in self.episodes if episode.case_id == case_id)

    def test_v3_resolves_authoritative_phone_ambiguity(self) -> None:
        episode = self._episode("hard-phone-ambiguous")
        decision = resolve_attribute_v3_from_claims(
            place_id=episode.case_id,
            attribute=episode.attribute,
            candidates=[],
            claims=extract_claims_from_replay_episode(episode),
            place_context=episode.place,
        )

        self.assertFalse(decision.abstained)
        self.assertEqual(normalize_phone(decision.decision), "4155551212")

    def test_v3_resolves_mixed_authoritative_name_corroboration(self) -> None:
        episode = self._episode("hard-mixed-authoritative-name")
        decision = resolve_attribute_v3_from_claims(
            place_id=episode.case_id,
            attribute=episode.attribute,
            candidates=[],
            claims=extract_claims_from_replay_episode(episode),
            place_context=episode.place,
        )

        self.assertFalse(decision.abstained)
        self.assertEqual(normalize_name(decision.decision), normalize_name(episode.gold_value))

    def test_v3_beats_v2_on_the_hard_case_fixture(self) -> None:
        report = evaluate_benchmark_v3(self.episodes)

        self.assertGreaterEqual(report["comparison"]["accuracy_delta"], 0.1)
        self.assertEqual(report["resolver_v3"]["high_confidence_wrong_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
