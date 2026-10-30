#!/usr/bin/env python3
"""Fail closed on revision-record drift and missing generated artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from common import ARTIFACTS, RECORDS, write_json


def main():
    checks = {}
    assertions = json.load((RECORDS / "submission_assertions_rev2.json").open())
    checks["canonical_assertions_pass"] = assertions.get("pass") is True

    conv = json.load((RECORDS / "numeric_support_baselines_convfinqa.json").open())
    selected = next(
        row for row in conv["results"]
        if row["method"] == "operand_constrained_closure"
    )
    checks["convfinqa_frozen_counts"] = (
        selected["support_recall_counts"] == "360/386"
        and selected["unsupported_detection_counts"] == "371/386"
    )

    wiki = json.load((RECORDS / "wikisql_semantic_validation_summary.json").open())
    checks["wikisql_500_executable"] = wiki["executable_base_examples"] >= 500

    anomaly = json.load((RECORDS / "anomaly_baseline_results.json").open())
    checks["anomaly_no_declared_leakage"] = not anomaly["forbidden_features_used"]

    runtime = json.load((RECORDS / "audit_runtime_summary.json").open())
    checks["deterministic_repeats"] = runtime["deterministic_repeat_agreement"]["all_exact"]

    positive = json.load((RECORDS / "positive_claim_analysis.json").open())
    checks["positive_analysis_frozen"] = (
        positive["analysis_status"] == "frozen_record_reanalysis_no_retuning"
    )
    checks["positive_human_matched_coverage"] = (
        positive["human_matched_coverage"]["supported_retention"] == 1.0
        and positive["human_matched_coverage"]["matched_coverage_hypergeom_upper_p"] < 0.001
    )
    checks["positive_official_score_randomization"] = (
        positive["official_score_matched_coverage"]["absolute_score_gain"] > 0.1
        and positive["official_score_matched_coverage"]["randomization_upper_p"] < 1e-4
    )

    expected = [
        *(ARTIFACTS / f"table_{index}_{name}.tex" for index, name in (
            (1, "datasets"),
            (2, "baselines"),
            (3, "live_agent_audit"),
            (4, "delivery_policies"),
            (5, "wikisql_semantic_validation"),
            (6, "ablation_runtime"),
        )),
        *(ARTIFACTS / f"fig_{index}_{name}.pdf" for index, name in (
            (1, "three_panel_teaser"),
            (2, "three_phase_architecture"),
            (3, "baseline_results"),
            (4, "live_governance"),
            (5, "semantic_ablation"),
        )),
    ]
    expected.append(ARTIFACTS / "table_positive_claim_statistics.tex")
    checks["all_expected_artifacts"] = all(path.exists() and path.stat().st_size for path in expected)
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected_artifacts": [str(path.relative_to(ARTIFACTS.parent)) for path in expected],
    }
    write_json(RECORDS / "ipmc_rev2_consistency.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
