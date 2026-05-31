import unittest

from places_attr_conflation.benchmark_common import (
    accumulate_attribute_stats,
    new_attribute_stats,
    summarize_attribute_stats,
    summarize_benchmark_counts,
)


class BenchmarkCommonTests(unittest.TestCase):
    def test_correct_abstention_rate_counts_only_expected_abstains(self) -> None:
        stats = new_attribute_stats()
        accumulate_attribute_stats(
            stats,
            has_gold=True,
            answerable=False,
            expected_abstain=True,
            answerable_correct=False,
            expected_correct=True,
            abstained=True,
            unsafe_prediction=False,
            high_confidence_wrong=False,
            high_confidence_unsafe=False,
        )
        accumulate_attribute_stats(
            stats,
            has_gold=True,
            answerable=False,
            expected_abstain=True,
            answerable_correct=False,
            expected_correct=False,
            abstained=False,
            unsafe_prediction=True,
            high_confidence_wrong=False,
            high_confidence_unsafe=False,
        )
        accumulate_attribute_stats(
            stats,
            has_gold=True,
            answerable=True,
            expected_abstain=False,
            answerable_correct=True,
            expected_correct=True,
            abstained=False,
            unsafe_prediction=False,
            high_confidence_wrong=False,
            high_confidence_unsafe=False,
        )
        accumulate_attribute_stats(
            stats,
            has_gold=True,
            answerable=True,
            expected_abstain=False,
            answerable_correct=False,
            expected_correct=False,
            abstained=True,
            unsafe_prediction=False,
            high_confidence_wrong=False,
            high_confidence_unsafe=False,
        )

        summary = summarize_attribute_stats(stats)
        self.assertAlmostEqual(summary["correct_abstention_rate"], 0.5)
        self.assertAlmostEqual(summary["abstention_accuracy"], 0.5)
        self.assertLessEqual(summary["correct_abstention_rate"], 1.0)
        self.assertLessEqual(summary["abstention_accuracy"], 1.0)

    def test_benchmark_summary_exposes_correct_abstention_rate(self) -> None:
        stats = new_attribute_stats()
        accumulate_attribute_stats(
            stats,
            has_gold=True,
            answerable=False,
            expected_abstain=True,
            answerable_correct=False,
            expected_correct=True,
            abstained=True,
            unsafe_prediction=False,
            high_confidence_wrong=False,
            high_confidence_unsafe=False,
        )
        summary = summarize_benchmark_counts({"phone": stats}, episodes_total=1, resolver_name="demo")
        self.assertAlmostEqual(summary["correct_abstention_rate"], 1.0)
        self.assertAlmostEqual(summary["abstention_accuracy"], 1.0)
        self.assertAlmostEqual(summary["per_attribute"]["phone"]["correct_abstention_rate"], 1.0)
        self.assertAlmostEqual(summary["per_attribute"]["phone"]["abstention_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
