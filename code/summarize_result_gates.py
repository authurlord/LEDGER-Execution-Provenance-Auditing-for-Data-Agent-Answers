#!/usr/bin/env python3
"""Evaluate the predeclared IP&MC result gates from generated records."""

from __future__ import annotations

from common import DOCS, RECORDS, load_json, write_json


def result(rows, policy):
    return next(
        row for row in rows
        if row["dataset"] == "kramabench_human_clear" and row["policy"] == policy
    )


def numeric(rows, method):
    return next(row for row in rows["results"] if row["method"] == method)


def wiki_result(summary, perturbation, validator):
    return next(
        row for row in summary["results"]
        if row["perturbation_class"] == perturbation and row["validator"] == validator
    )


def main():
    assertions = load_json(RECORDS / "submission_assertions_rev2.json")
    conv = load_json(RECORDS / "numeric_support_baselines_convfinqa.json")
    delivery = load_json(RECORDS / "delivery_policy_results.json")
    wiki = load_json(RECORDS / "wikisql_semantic_validation_summary.json")
    ablation = load_json(RECORDS / "closure_ablation_grid_rev2.json")
    live = load_json(RECORDS / "live_six_domain_summary.json")
    runtime = load_json(RECORDS / "audit_runtime_summary.json")

    constrained = numeric(conv, "operand_constrained_closure")
    free = numeric(conv, "all_context_numbers")
    replay = result(delivery["results"], "deliver_if_replay_certificate")
    random = result(delivery["results"], "random_matched_replay_coverage")
    same_output = wiki_result(
        wiki, "same_original_output_counterfactual", "original_output_equality"
    )
    structural = wiki_result(
        wiki, "same_original_output_counterfactual", "normalized_sql_exact_match"
    )
    differential = wiki_result(
        wiki, "same_original_output_counterfactual", "differential_execution"
    )
    frozen_rows = [
        row for row in ablation["grid"]
        if row["operand_source"] == "reached_operands"
        and row["tol_rel"] == 1e-4
        and row["restriction"] == "citation_restricted"
        and len(row["operations"]) == 5
    ]
    statuses = {
        "A": (
            constrained["support_recall"] > free["support_recall"]
            and constrained["unsupported_detection"] > free["unsupported_detection"]
        ),
        "B": (
            replay["coverage"] == random["coverage"]
            and replay["human_unsupported_delivered"]
            < random["human_unsupported_delivered"]
            and replay["official_score_on_delivered"]
            >= random["official_score_on_delivered"]
        ),
        "C": (
            same_output["detected"] == 0
            and structural["detected"] > 0
            and differential["detected"] > 0
        ),
        "D": (
            len(live["rows"]) == 6
            and wiki["executable_base_examples"] >= 500
            and assertions["observed"]["kramabench"]["total"] > 0
            and assertions["observed"]["dabstep"]["total"] > 0
            and assertions["observed"]["multihiertt"]["total"] > 0
        ),
        "E": (
            len(ablation["grid"]) == 216
            and sorted(row["max_depth"] for row in frozen_rows) == [1, 2, 3]
            and ablation["frozen_selection"]["max_depth"] == 2
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(statuses.values()) else "FAIL",
        "canonical_assertions_pass": assertions["pass"],
        "gates": {
            "A": {
                "pass": statuses["A"],
                "evidence": {
                    "operand_constrained_support_recall": constrained["support_recall_counts"],
                    "operand_constrained_unsupported_detection": constrained[
                        "unsupported_detection_counts"
                    ],
                    "free_support_recall": free["support_recall_counts"],
                    "free_unsupported_detection": free["unsupported_detection_counts"],
                },
                "claim": "Operand-constrained closure dominates free closure on the fixed ConvFinQA remainder.",
            },
            "B": {
                "pass": statuses["B"],
                "evidence": {
                    "matched_coverage": replay["coverage"],
                    "replay_human_supported_delivered": replay[
                        "human_supported_delivered"
                    ],
                    "replay_human_unsupported_delivered": replay[
                        "human_unsupported_delivered"
                    ],
                    "random_human_supported_delivered": random[
                        "human_supported_delivered"
                    ],
                    "random_human_unsupported_delivered": random[
                        "human_unsupported_delivered"
                    ],
                    "replay_official_score": replay["official_score_on_delivered"],
                    "random_official_score": random["official_score_on_delivered"],
                    "bootstrap_95_interval": delivery["bootstrap_audit"][
                        "percentile_95_interval"
                    ],
                },
                "claim": "Conservative selective delivery improves matched-coverage case evidence; official score remains deployment utility, not support truth.",
            },
            "C": {
                "pass": statuses["C"],
                "evidence": {
                    "same_output_n": same_output["n"],
                    "original_output_detected": same_output["detected"],
                    "normalized_sql_detected": structural["detected"],
                    "differential_execution_detected": differential["detected"],
                    "false_alarms_on_unmodified_gold": wiki[
                        "false_alarms_on_unmodified_gold"
                    ],
                },
                "claim": "Structural and differential SQL checks detect semantic changes missed by output equality; replay remains a separate audit witness.",
            },
            "D": {
                "pass": statuses["D"],
                "evidence": {
                    "families": [
                        "ConvFinQA",
                        "MultiHiertt",
                        "KramaBench",
                        "DABstep",
                        "WikiSQL",
                    ],
                    "python_pandas_records": True,
                    "sql_executable_examples": wiki["executable_base_examples"],
                    "live_domains_reported": len(live["rows"]),
                },
                "claim": "The record/replay evaluation covers Python/pandas and SQL across five testbed families; unsupported live formats remain explicitly out of scope.",
            },
            "E": {
                "pass": statuses["E"],
                "evidence": {
                    "grid_cells": len(ablation["grid"]),
                    "frozen_selection": ablation["frozen_selection"],
                    "depth_tradeoff": [
                        {
                            "depth": row["max_depth"],
                            "support_recall": row["support_recall_counts"],
                            "unsupported_detection": row[
                                "unsupported_detection_counts"
                            ],
                            "wrong_operation_false_support": row[
                                "wrong_operation_false_support_counts"
                            ],
                        }
                        for row in sorted(frozen_rows, key=lambda row: row["max_depth"])
                    ],
                },
                "claim": "Depth exposes an interpretable recall/collision tradeoff; the depth-2 point stayed frozen.",
            },
        },
        "determinism": runtime["deterministic_repeat_agreement"],
    }
    write_json(RECORDS / "experiment_gate_summary.json", payload)

    lines = [
        "# IP&MC revision result gates",
        "",
        f"Overall status: **{payload['status']}**. Canonical assertions: "
        f"**{'PASS' if assertions['pass'] else 'FAIL'}**.",
        "",
        "| gate | status | evidence-backed conclusion |",
        "|---|---|---|",
    ]
    for gate, row in payload["gates"].items():
        lines.append(
            f"| {gate} | {'PASS' if row['pass'] else 'FAIL'} | {row['claim']} |"
        )
    lines.extend([
        "",
        "These gates support audit-layer recomputability and conservative governance, "
        "not an improvement in the host agent's reasoning accuracy. The human taxonomy "
        "is still a review draft, and the live astronomy/biomedical formats remain out "
        "of scope for the current hermetic replay implementation.",
        "",
    ])
    (DOCS / "EXPERIMENT_GATE_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
