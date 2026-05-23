from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from places_attr_conflation.harness import (
    compare_arms,
    evaluate_final_decisions,
    evaluate_resolver_v2_on_replay,
    evaluate_retrieval_proof,
    evaluate_website_authority_replay,
)
from places_attr_conflation.claim_extraction import extract_claims_from_replay_episode
from places_attr_conflation.evidence_graph import build_evidence_graph
from places_attr_conflation.replay import load_replay_corpus


ROOT = Path(__file__).resolve().parent
SMALL_FIXTURE = ROOT / "fixtures" / "santa_cruz_replay_corpus.json"
LARGE_FIXTURE = ROOT / "fixtures" / "santa_cruz_replay_corpus_expanded.json"


class SantaCruzExpandedCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.small_episodes = load_replay_corpus(SMALL_FIXTURE)
        cls.large_episodes = load_replay_corpus(LARGE_FIXTURE)

    def test_expanded_corpus_preserves_starter_timeline(self) -> None:
        small_case_ids = {episode.case_id for episode in self.small_episodes}
        large_case_ids = {episode.case_id for episode in self.large_episodes}

        self.assertTrue(small_case_ids.issubset(large_case_ids))
        self.assertEqual(len(self.small_episodes), 12)
        self.assertEqual(len(self.large_episodes), 24)
        self.assertEqual(Counter(episode.label_origin for episode in self.large_episodes), {
            "authoritative_santa_cruz_v1": 12,
            "authoritative_santa_cruz_v2": 12,
        })

    def test_expanded_corpus_has_expected_entity_distribution(self) -> None:
        episodes = self.large_episodes

        self.assertEqual(Counter(episode.attribute for episode in episodes), {"website": 8, "phone": 8, "address": 8})
        self.assertEqual(
            Counter(episode.place["name"] for episode in episodes),
            {
                "City Clerk": 3,
                "Santa Cruz Public Libraries Downtown Branch": 3,
                "Santa Cruz Museum of Natural History": 3,
                "Santa Cruz Institute for Particle Physics": 3,
                "UC Santa Cruz History Department": 3,
                "UC Santa Cruz Office of the Registrar": 3,
                "UC Santa Cruz Learning Support Services": 3,
                "UC Santa Cruz Norris Center for Natural History": 3,
            },
        )
        self.assertTrue(all(episode.place["city"] == "Santa Cruz" for episode in episodes))
        self.assertTrue(all(episode.place["region"] == "CA" for episode in episodes))
        self.assertTrue(all(episode.identity_label == "SAME_ENTITY" for episode in episodes))

    def test_expanded_corpus_supports_strict_authority_and_final_decisions(self) -> None:
        arms = compare_arms(self.large_episodes)
        proof = evaluate_retrieval_proof(self.large_episodes)
        website = evaluate_website_authority_replay(self.large_episodes)
        final = evaluate_final_decisions(self.large_episodes)

        self.assertEqual(arms["targeted"]["authoritative_found_rate"], 1.0)
        self.assertEqual(arms["fallback"]["authoritative_found_rate"], 0.0)
        self.assertEqual(arms["targeted"]["citation_precision"], 1.0)
        self.assertEqual(proof["deltas"]["authoritative_found_rate"], 1.0)
        self.assertEqual(website["official_pages_found_rate"], 1.0)
        self.assertEqual(website["selected_official_rate"], 1.0)
        self.assertEqual(website["false_official_rate"], 0.0)
        self.assertEqual(final["accuracy"], 1.0)
        self.assertEqual(final["abstention_rate"], 0.0)
        self.assertEqual(final["high_confidence_wrong_rate"], 0.0)

    def test_expanded_address_cases_do_not_create_fake_prose_contradictions(self) -> None:
        address_episodes = [episode for episode in self.large_episodes if episode.attribute == "address"]
        report = evaluate_resolver_v2_on_replay(address_episodes)

        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["abstention_rate"], 0.0)
        self.assertEqual(report["high_confidence_wrong_rate"], 0.0)

        history = next(episode for episode in address_episodes if episode.case_id == "sc-history-address")
        graph = build_evidence_graph(
            place_id=history.case_id,
            attribute=history.attribute,
            candidates=[],
            claims=extract_claims_from_replay_episode(history),
        )

        self.assertEqual([group.normalized_value for group in graph.groups], ["1156 high st santa cruz ca 95064"])
        self.assertEqual(graph.contradictions, [])


if __name__ == "__main__":
    unittest.main()
