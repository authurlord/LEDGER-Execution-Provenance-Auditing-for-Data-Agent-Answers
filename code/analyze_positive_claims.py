#!/usr/bin/env python3
"""Strengthen frozen LEDGER results with paired and matched-coverage inference.

This script does not tune a setting or alter any canonical record.  It adds
exact paired tests, pair-level bootstrap intervals, and a fixed-coverage
random-policy reference to the already frozen July 30 results.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from common import ARTIFACTS, RECORDS, load_json, load_jsonl, write_json
from eval_delivery_and_escalation_policies import all_kramabench_cases, human_cases


SEED = 20260730
BOOTSTRAP_REPEATS = 50_000
RANDOMIZATION_REPEATS = 200_000


def _paired_rows(path):
    grouped = defaultdict(dict)
    for row in load_jsonl(path):
        if row["taskid"].endswith("::positive"):
            grouped[row["taskid"].removesuffix("::positive")]["positive"] = row
        elif row["taskid"].endswith("::negative"):
            grouped[row["taskid"].removesuffix("::negative")]["negative"] = row
    return [
        rows for _, rows in sorted(grouped.items())
        if set(rows) == {"positive", "negative"}
    ]


def _method_accept(row, method):
    depth_rank = {"depth0": 0, "depth1": 1, "depth2": 2}
    matched_depth = depth_rank.get(row.get("matched_depth"), 99)
    if method == "single_operation":
        return matched_depth <= 1
    if method == "operand_constrained_closure":
        return matched_depth <= 2
    if method == "all_numbers_in_cited_file":
        if row["taskid"].endswith("::positive"):
            return bool(row.get("free_positive_hit"))
        return not bool(row.get("free_negative_flagged"))
    raise ValueError(method)


def _mcnemar_exact_two_sided(left_wins, right_wins):
    discordant = left_wins + right_wins
    if discordant == 0:
        return 1.0, 0.0
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2.0)
        for k in range(min(left_wins, right_wins) + 1)
    ]
    maximum = max(log_terms)
    log_tail = maximum + math.log(sum(math.exp(value - maximum) for value in log_terms))
    log_p = min(0.0, math.log(2.0) + log_tail)
    return (math.exp(log_p) if log_p > -745 else 0.0), log_p / math.log(10.0)


def _paired_comparison(dataset, pairs, baseline, rng):
    per_pair = []
    ledger_wins = 0
    baseline_wins = 0
    for pair in pairs:
        baseline_correct = 0
        ledger_correct = 0
        for polarity, label in (("positive", True), ("negative", False)):
            row = pair[polarity]
            b_correct = _method_accept(row, baseline) == label
            l_correct = _method_accept(row, "operand_constrained_closure") == label
            baseline_correct += int(b_correct)
            ledger_correct += int(l_correct)
            ledger_wins += int(l_correct and not b_correct)
            baseline_wins += int(b_correct and not l_correct)
        per_pair.append((ledger_correct - baseline_correct) / 2.0)

    deltas = np.asarray(per_pair, dtype=float)
    draws = rng.integers(0, len(deltas), size=(BOOTSTRAP_REPEATS, len(deltas)))
    boot = deltas[draws].mean(axis=1)
    baseline_correct_total = 0
    ledger_correct_total = 0
    for pair in pairs:
        for polarity, label in (("positive", True), ("negative", False)):
            row = pair[polarity]
            baseline_correct_total += int(_method_accept(row, baseline) == label)
            ledger_correct_total += int(
                _method_accept(row, "operand_constrained_closure") == label
            )
    denominator = 2 * len(pairs)
    exact_p, log10_p = _mcnemar_exact_two_sided(ledger_wins, baseline_wins)
    return {
        "dataset": dataset,
        "baseline": baseline,
        "n_pairs": len(pairs),
        "n_binary_decisions": denominator,
        "baseline_balanced_accuracy": baseline_correct_total / denominator,
        "ledger_balanced_accuracy": ledger_correct_total / denominator,
        "absolute_gain": (ledger_correct_total - baseline_correct_total) / denominator,
        "ledger_only_correct": ledger_wins,
        "baseline_only_correct": baseline_wins,
        "mcnemar_exact_two_sided_p": exact_p,
        "mcnemar_exact_log10_p": log10_p,
        "pair_bootstrap_95_interval": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
    }


def _hypergeom_upper_tail(population, positives, draws, observed):
    denominator = math.comb(population, draws)
    maximum = min(positives, draws)
    minimum = max(0, draws - (population - positives))
    numerator = sum(
        math.comb(positives, hits) * math.comb(population - positives, draws - hits)
        for hits in range(max(observed, minimum), maximum + 1)
    )
    return numerator / denominator


def _human_delivery():
    cases = human_cases()
    delivered = [case for case in cases if case["replay_certificate"]]
    positives = sum(case["human_label"] == 1 for case in cases)
    delivered_positives = sum(case["human_label"] == 1 for case in delivered)
    unsupported = sum(case["human_label"] == 0 for case in cases)
    delivered_unsupported = sum(case["human_label"] == 0 for case in delivered)
    domains = {}
    for domain in sorted({case["case_id"].split("-", 1)[0] for case in cases}):
        subset = [case for case in cases if case["case_id"].startswith(f"{domain}-")]
        kept = [case for case in subset if case["replay_certificate"]]
        support_n = sum(case["human_label"] == 1 for case in subset)
        kept_support = sum(case["human_label"] == 1 for case in kept)
        domains[domain] = {
            "n": len(subset),
            "delivered_n": len(kept),
            "supported_n": support_n,
            "supported_delivered": kept_support,
            "deliver_all_support_precision": support_n / len(subset),
            "replay_support_precision": kept_support / len(kept),
            "absolute_precision_gain": (
                kept_support / len(kept) - support_n / len(subset)
            ),
        }
    leave_one_domain_out = {}
    for domain in domains:
        subset = [
            case for case in cases
            if not case["case_id"].startswith(f"{domain}-")
        ]
        kept = [case for case in subset if case["replay_certificate"]]
        support_n = sum(case["human_label"] == 1 for case in subset)
        kept_support = sum(case["human_label"] == 1 for case in kept)
        leave_one_domain_out[domain] = {
            "removed_domain": domain,
            "n": len(subset),
            "delivered_n": len(kept),
            "absolute_precision_gain": (
                kept_support / len(kept) - support_n / len(subset)
            ),
        }
    return {
        "n": len(cases),
        "delivered_n": len(delivered),
        "human_supported_n": positives,
        "human_supported_delivered": delivered_positives,
        "support_precision_deliver_all": positives / len(cases),
        "support_precision_replay": delivered_positives / len(delivered),
        "absolute_precision_gain": (
            delivered_positives / len(delivered) - positives / len(cases)
        ),
        "supported_retention": delivered_positives / positives,
        "human_unsupported_n": unsupported,
        "human_unsupported_delivered": delivered_unsupported,
        "unsupported_escalation_rate": (unsupported - delivered_unsupported) / unsupported,
        "matched_coverage_hypergeom_upper_p": _hypergeom_upper_tail(
            len(cases), positives, len(delivered), delivered_positives
        ),
        "per_domain": domains,
        "leave_one_domain_out": leave_one_domain_out,
        "null": "uniformly random delivery of the same number of cases",
    }


def _official_score_randomization(rng):
    cases = [case for case in all_kramabench_cases() if case["official_score"] is not None]
    scores = np.asarray([case["official_score"] for case in cases], dtype=float)
    selected = np.asarray([case["replay_certificate"] for case in cases], dtype=bool)
    selected_n = int(selected.sum())
    observed = float(scores[selected].mean())
    null_means = []
    batch_size = 2_000
    remaining = RANDOMIZATION_REPEATS
    while remaining:
        current = min(batch_size, remaining)
        random_keys = rng.random((current, len(scores)))
        indices = np.argpartition(random_keys, selected_n - 1, axis=1)[:, :selected_n]
        null_means.append(scores[indices].mean(axis=1))
        remaining -= current
    null = np.concatenate(null_means)
    return {
        "scored_population_n": len(scores),
        "replay_delivered_scored_n": selected_n,
        "coverage_on_scored": selected_n / len(scores),
        "deliver_all_official_score": float(scores.mean()),
        "replay_official_score": observed,
        "absolute_score_gain": observed - float(scores.mean()),
        "relative_score_gain": observed / float(scores.mean()) - 1.0,
        "random_matched_mean": float(null.mean()),
        "random_matched_95_interval": [
            float(np.quantile(null, 0.025)),
            float(np.quantile(null, 0.975)),
        ],
        "randomization_upper_p": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "randomization_repeats": len(null),
        "random_seed": SEED,
        "null": "uniformly random delivery of the same number of scored episodes",
    }


def _semantic_stack():
    summary = load_json(RECORDS / "wikisql_semantic_validation_summary.json")
    rows = summary["results"]
    output = [row for row in rows if row["validator"] == "original_output_equality"]
    structural = [row for row in rows if row["validator"] == "normalized_sql_exact_match"]
    output_detected = sum(row["detected"] for row in output)
    structural_detected = sum(row["detected"] for row in structural)
    total = sum(row["n"] for row in structural)
    return {
        "controlled_perturbations": total,
        "output_equality_detected": output_detected,
        "output_equality_detection_rate": output_detected / total,
        "normalized_sql_detected": structural_detected,
        "normalized_sql_detection_rate": structural_detected / total,
        "same_output_cases_recovered_by_structure": structural_detected - output_detected,
        "unmodified_gold_n": summary["executable_base_examples"],
        "normalized_sql_false_alarms": summary["false_alarms_on_unmodified_gold"][
            "normalized_sql_exact_match"
        ],
        "scope": "controlled SQL semantic perturbations; not natural-error prevalence",
    }


def _render_tex(payload):
    rows = payload["paired_baseline_tests"]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Baseline & Baseline BA & LEDGER BA & Gain & Exact $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {row['baseline'].replace('_', ' ')} & "
            f"{row['baseline_balanced_accuracy']:.3f} & "
            f"{row['ledger_balanced_accuracy']:.3f} & "
            f"{row['absolute_gain']:+.3f} & "
            f"{('<10^{-300}' if row['mcnemar_exact_two_sided_p'] == 0 else format(row['mcnemar_exact_two_sided_p'], '.2e'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ARTIFACTS / "table_positive_claim_statistics.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    rng = np.random.default_rng(SEED)
    conv = _paired_rows(RECORDS / "records_ledger_checker_convfinqa_full.jsonl")
    conv = [
        pair for pair in conv
        if pair["positive"].get("constrained_positive_hit") is not None
    ]
    multi = _paired_rows(RECORDS / "records_ledger_checker_multihiertt.jsonl")
    payload = {
        "schema_version": 1,
        "analysis_status": "frozen_record_reanalysis_no_retuning",
        "seed": SEED,
        "paired_baseline_tests": [
            _paired_comparison("convfinqa", conv, "single_operation", rng),
            _paired_comparison("convfinqa", conv, "all_numbers_in_cited_file", rng),
            _paired_comparison("multihiertt", multi, "single_operation", rng),
        ],
        "human_matched_coverage": _human_delivery(),
        "official_score_matched_coverage": _official_score_randomization(rng),
        "semantic_validation_stack": _semantic_stack(),
    }
    write_json(RECORDS / "positive_claim_analysis.json", payload)
    _render_tex(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
