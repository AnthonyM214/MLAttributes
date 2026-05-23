from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from places_attr_conflation.harness import (
    compare_arms,
    evaluate_final_decisions,
    evaluate_retrieval_proof,
    evaluate_website_authority_replay,
)
from places_attr_conflation.replay import load_replay_corpus


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "santa_cruz_replay_corpus.json"


class SantaCruzCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episodes = load_replay_corpus(FIXTURE)

    def test_santa_cruz_corpus_has_expected_structure(self) -> None:
        episodes = self.episodes

        self.assertEqual(len(episodes), 12)
        self.assertEqual(Counter(episode.attribute for episode in episodes), {"website": 4, "phone": 4, "address": 4})
        self.assertEqual(
            Counter(episode.place["name"] for episode in episodes),
            {
                "City Clerk": 3,
                "Santa Cruz Public Libraries Downtown Branch": 3,
                "Santa Cruz Museum of Natural History": 3,
                "Santa Cruz Institute for Particle Physics": 3,
            },
        )
        self.assertTrue(all(episode.place["city"] == "Santa Cruz" for episode in episodes))
        self.assertTrue(all(episode.place["region"] == "CA" for episode in episodes))
        self.assertTrue(all(episode.identity_label == "SAME_ENTITY" for episode in episodes))
        self.assertTrue(all(episode.label_origin == "authoritative_santa_cruz_v1" for episode in episodes))
        self.assertTrue(all(episode.truth_source_type in {"official_site", "government"} for episode in episodes))
        self.assertTrue(all(episode.final_decision is not None for episode in episodes))
        self.assertTrue(all(not episode.final_decision.abstained for episode in episodes if episode.final_decision is not None))

    def test_santa_cruz_corpus_separates_targeted_from_fallback(self) -> None:
        arms = compare_arms(self.episodes)
        targeted = arms["targeted"]
        fallback = arms["fallback"]
        proof = evaluate_retrieval_proof(self.episodes)

        self.assertEqual(targeted["authoritative_found_rate"], 1.0)
        self.assertEqual(fallback["authoritative_found_rate"], 0.0)
        self.assertEqual(targeted["citation_precision"], 1.0)
        self.assertEqual(targeted["top1_authoritative_rate"], 1.0)
        self.assertEqual(proof["deltas"]["authoritative_found_rate"], 1.0)

    def test_santa_cruz_corpus_supports_clean_final_decisions(self) -> None:
        website = evaluate_website_authority_replay(self.episodes)
        final = evaluate_final_decisions(self.episodes)

        self.assertEqual(website["official_pages_found_rate"], 1.0)
        self.assertEqual(website["selected_official_rate"], 1.0)
        self.assertEqual(website["false_official_rate"], 0.0)
        self.assertEqual(final["accuracy"], 1.0)
        self.assertEqual(final["abstention_rate"], 0.0)
        self.assertEqual(final["high_confidence_wrong_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
