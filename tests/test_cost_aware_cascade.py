import unittest

from places_attr_conflation.budgeted_dorking import build_budgeted_dork_plan
from places_attr_conflation.cost_policy import CostPolicy, should_escalate_to_retrieval
from places_attr_conflation.resolve_cascade import resolve_attribute_cascade
from places_attr_conflation.resolvepoi_selective import SelectiveRouterPrediction


class DummyRouter:
    def __init__(self, source="current", confidence=0.97, abstained=False):
        self.source = source
        self.confidence = confidence
        self.abstained = abstained

    def predict(self, **kwargs):
        return SelectiveRouterPrediction(
            source=self.source,
            confidence=self.confidence,
            abstained=self.abstained,
            reason="dummy router vote",
        )


class CostAwareCascadeTests(unittest.TestCase):
    def test_budgeted_dorking_reduces_full_query_plan(self):
        place = {
            "name": "Santa Cruz Museum of Art & History",
            "city": "Santa Cruz",
            "region": "CA",
            "address": "705 Front St",
            "phone": "8314291964",
            "website": "https://www.santacruzmah.org",
        }
        policy = CostPolicy(max_queries_by_attribute={"website": 3})
        budgeted = build_budgeted_dork_plan(place, "website", policy=policy)
        self.assertLessEqual(budgeted.selected_query_count, 3)
        self.assertGreater(budgeted.original_query_count, budgeted.selected_query_count)
        self.assertGreater(budgeted.query_reduction_rate, 0.0)
        self.assertTrue(all(query.layer != "fallback" for query in budgeted.queries))

    def test_normalized_equal_short_circuits_before_router_or_retrieval(self):
        result = resolve_attribute_cascade(
            place_id="case-1",
            attribute="phone",
            candidates=["(831) 555-1212", "831-555-1212"],
            learned_router=DummyRouter(source="base", confidence=0.99),
            policy=CostPolicy(allow_live_retrieval=True),
        )
        self.assertFalse(result.pending_retrieval)
        self.assertFalse(result.decision.abstained)
        self.assertEqual(result.trace.cost_tier, "normalization")
        self.assertTrue(result.trace.normalized_equal)

    def test_high_confidence_router_vote_short_circuits(self):
        result = resolve_attribute_cascade(
            place_id="case-2",
            attribute="website",
            candidates=["https://official.example", "https://directory.example"],
            learned_router=DummyRouter(source="current", confidence=0.97),
            policy=CostPolicy(allow_live_retrieval=True, auto_accept_threshold=0.92),
        )
        self.assertFalse(result.pending_retrieval)
        self.assertFalse(result.decision.abstained)
        self.assertEqual(result.decision.decision, "https://official.example")
        self.assertEqual(result.trace.cost_tier, "selective_router")

    def test_uncertain_low_value_case_abstains_without_retrieval_by_default(self):
        result = resolve_attribute_cascade(
            place_id="case-3",
            attribute="address",
            candidates=["10 Main St", "12 Main St"],
            learned_router=DummyRouter(source="current", confidence=0.55, abstained=True),
            policy=CostPolicy(allow_live_retrieval=True, high_value_place=False),
        )
        self.assertFalse(result.pending_retrieval)
        self.assertTrue(result.decision.abstained)
        self.assertEqual(result.trace.cost_tier, "abstain_without_retrieval")

    def test_uncertain_high_value_case_recommends_budgeted_retrieval(self):
        decision = should_escalate_to_retrieval(
            router_confidence=0.51,
            evidence_confidence=0.0,
            policy=CostPolicy(allow_live_retrieval=True, high_value_place=True),
        )
        self.assertEqual(decision.action, "retrieve")
        self.assertEqual(decision.cost_tier, "budgeted_retrieval")


if __name__ == "__main__":
    unittest.main()
