import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class PositiveClaimAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(
            (ROOT / "records" / "positive_claim_analysis.json").read_text(
                encoding="utf-8"
            )
        )

    def test_frozen_reanalysis(self):
        self.assertEqual(
            self.payload["analysis_status"],
            "frozen_record_reanalysis_no_retuning",
        )

    def test_all_paired_baseline_gains_are_positive(self):
        for row in self.payload["paired_baseline_tests"]:
            self.assertGreater(row["absolute_gain"], 0)
            self.assertLess(row["mcnemar_exact_log10_p"], -10)

    def test_human_delivery_retains_supported_cases(self):
        human = self.payload["human_matched_coverage"]
        self.assertEqual(human["human_supported_delivered"], human["human_supported_n"])
        self.assertLess(human["matched_coverage_hypergeom_upper_p"], 0.001)
        self.assertTrue(
            all(
                row["absolute_precision_gain"] > 0
                for row in human["leave_one_domain_out"].values()
            )
        )

    def test_score_and_semantic_extensions(self):
        score = self.payload["official_score_matched_coverage"]
        self.assertGreater(score["absolute_score_gain"], 0.1)
        self.assertLess(score["randomization_upper_p"], 1e-4)
        semantic = self.payload["semantic_validation_stack"]
        self.assertEqual(
            semantic["normalized_sql_detected"], semantic["controlled_perturbations"]
        )
        self.assertEqual(semantic["same_output_cases_recovered_by_structure"], 466)


if __name__ == "__main__":
    unittest.main()
