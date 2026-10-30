import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    with (ROOT / "records" / name).open() as handle:
        return json.load(handle)


class RevisionRecordTests(unittest.TestCase):
    def test_frozen_numeric_point(self):
        conv = load("numeric_support_baselines_convfinqa.json")["results"]
        selected = next(
            row for row in conv if row["method"] == "operand_constrained_closure"
        )
        self.assertEqual(selected["support_recall_counts"], "360/386")
        self.assertEqual(selected["unsupported_detection_counts"], "371/386")

    def test_no_anomaly_feature_leakage(self):
        payload = load("anomaly_baseline_results.json")
        self.assertEqual(payload["forbidden_features_used"], [])
        forbidden = {
            "grounding_label", "checker_label", "certificate_type",
            "restricted_replay_match", "recomputed_value_minus_stated_answer",
            "gold", "gold_program_identity",
        }
        self.assertTrue(forbidden.isdisjoint(payload["features"]))

    def test_ablation_uses_true_seed(self):
        payload = load("closure_ablation_grid_rev2.json")
        selected = next(
            row for row in payload["grid"]
            if row["operand_source"] == "reached_operands"
            and row["restriction"] == "citation_restricted"
            and row["max_depth"] == 2
            and row["tol_rel"] == 1e-4
            and row["operations"] == ["sum", "diff", "ratio", "pctchg", "prod"]
        )
        self.assertEqual(selected["support_recall_counts"], "55/60")
        self.assertEqual(selected["unsupported_detection_counts"], "59/60")

    def test_delivery_safety_trend(self):
        rows = load("delivery_policy_results.json")["results"]
        human = {
            row["policy"]: row
            for row in rows if row["dataset"] == "kramabench_human_clear"
        }
        self.assertLess(
            human["deliver_if_replay_certificate"]["human_unsupported_delivered"],
            human["deliver_all"]["human_unsupported_delivered"],
        )

    def test_wikisql_semantic_boundary(self):
        payload = load("wikisql_semantic_validation_summary.json")
        self.assertGreaterEqual(payload["executable_base_examples"], 500)
        same = {
            row["validator"]: row
            for row in payload["results"]
            if row["perturbation_class"] == "same_original_output_counterfactual"
        }
        self.assertEqual(same["original_output_equality"]["detected"], 0)
        self.assertEqual(same["normalized_sql_exact_match"]["detection_rate"], 1.0)
        self.assertEqual(same["ledger_replay_certificate_only"]["detected"], 0)

    def test_runtime_determinism(self):
        payload = load("audit_runtime_summary.json")
        self.assertTrue(payload["deterministic_repeat_agreement"]["all_exact"])


if __name__ == "__main__":
    unittest.main()
