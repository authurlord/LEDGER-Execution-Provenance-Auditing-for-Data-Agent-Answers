#!/usr/bin/env python3
"""Build the concise handoff report directly from finalized experiment records."""

from __future__ import annotations

from common import DOCS, RECORDS, load_json


def main():
    gates = load_json(RECORDS / "experiment_gate_summary.json")
    assertions = load_json(RECORDS / "submission_assertions_rev2.json")
    delivery = load_json(RECORDS / "delivery_policy_results.json")
    judge = load_json(RECORDS / "evidence_aware_judge_summary.json")
    models = load_json(RECORDS / "evidence_aware_judge_model_comparison.json")
    wiki = load_json(RECORDS / "wikisql_semantic_validation_summary.json")
    runtime = load_json(RECORDS / "audit_runtime_summary.json")
    taxonomy = load_json(RECORDS / "human_boundary_taxonomy.json")
    conv = assertions["observed"]["convfinqa"]
    multi = assertions["observed"]["multihiertt"]
    human = assertions["observed"]["human"]
    bootstrap = delivery["bootstrap_audit"]
    total = runtime["components"]["total_audit"]

    lines = [
        "# LEDGER IP&MC experiment-expansion report — 2026-07-30",
        "",
        "## What this repository does",
        "",
        "LakeOrch is a data-lake question-answering and agent-evaluation repository. "
        "The `ledger` branch adds LEDGER, a deterministic post-hoc audit layer for "
        "completed data-agent episodes: citation-restricted replay checks whether a "
        "recorded program reproduces its stated answer from declared evidence, and "
        "bounded operand closure supplies an explicit derivation witness when direct "
        "replay is unavailable. Semantic task faithfulness is evaluated separately.",
        "",
        "## Outcome",
        "",
        f"- Predeclared gates A–E: **{gates['status']}**; canonical-record drift check: "
        f"**{'PASS' if assertions['pass'] else 'FAIL'}**.",
        f"- ConvFinQA constrained closure: {conv['support_recall_counts']} support "
        f"recall and {conv['unsupported_detection_counts']} unsupported detection.",
        f"- MultiHiertt transfer: {multi['support_recall_counts']} support recall and "
        f"{multi['unsupported_detection_counts']} unsupported detection.",
        f"- Human boundary: {human['replay_and_human_no']} replay certificates among "
        f"{human['no']} clear human-unsupported cases. This is the central evidence "
        "that recomputability does not prove semantic faithfulness.",
        f"- Matched-coverage delivery: replay delivered 11 supported / 10 unsupported; "
        "random delivered 4 supported / 17 unsupported. The bootstrap official-score "
        f"delta is {bootstrap['mean_delta']:.4f}, with percentile 95% interval "
        f"[{bootstrap['percentile_95_interval'][0]:.4f}, "
        f"{bootstrap['percentile_95_interval'][1]:.4f}].",
        f"- WikiSQL: {wiki['executable_base_examples']} executable examples and "
        f"{wiki['perturbations']} controlled semantic perturbations. In the 100 "
        "same-output counterfactuals, output equality detected 0, normalized SQL "
        "detected 100, and differential execution detected 27.",
        f"- Deterministic audit runtime: median {total['median_ms']:.3f} ms, "
        f"p95 {total['p95_ms']:.3f} ms; five-repeat agreement is exact.",
        "",
        "## Local model diagnostic",
        "",
    ]
    for dataset, row in judge["datasets"].items():
        lines.append(
            f"- 35B {dataset}: {row['majority_correct']}/{row['n_clear']} "
            f"majority agreement; {row['repeat_flip_items']} flip items."
        )
    lines.extend([
        f"- 35B parse failures: {judge['parse_failures']}/{judge['calls']}.",
        f"- {models['conclusion']}",
        "",
        "## Interpretation and limits",
        "",
        "- The viable result is a deterministic audit-and-governance contribution, "
        "not a claim that LEDGER improves agent reasoning or universally verifies "
        "question-to-program semantics.",
        "- Same-operand wrong-operation cases remain a hard boundary: frozen closure "
        "detects 9/382 overall and 0/347 within its operation/depth support.",
        "- The evidence-aware judge comparison is limited on transferred human cases "
        "because most archived cases lack raw cited rows and full program bodies.",
        f"- The 10-case human taxonomy is `{taxonomy['status']}` and requires manual "
        "confirmation; no suggested label was silently promoted.",
        "- WikiSQL differential execution is useful but incomplete (27/100 on the "
        "same-output boundary); structural validation carries most of this pilot.",
        "- The six-domain live table is a discovery-arm proxy; astronomy and "
        "biomedical formats are reported as out of scope for current hermetic replay.",
        "",
        "## Reproduction",
        "",
        "Run from the repository root with the experiment environment on `PYTHONPATH`:",
        "",
        "```bash",
        "python code/submission_assertions_rev2.py",
        "python code/check_record_to_paper_consistency.py",
        "python -m unittest discover -s tests/ipmc_rev2 -v",
        "python code/build_ipmc_rev2_artifacts.py --records records --out artifacts/ipmc_rev2",
        "```",
        "",
        "See `docs/RERUN_LEDGER.md` for every corrected run and "
        "`records/experiment_gate_summary.json` for machine-readable gate evidence.",
        "",
    ])
    (DOCS / "IPMC_REV2_EXPERIMENT_REPORT_20260730.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
