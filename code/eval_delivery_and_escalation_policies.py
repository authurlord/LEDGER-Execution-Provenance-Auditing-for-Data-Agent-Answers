#!/usr/bin/env python3
"""Evaluate selective-delivery policies over completed agent episodes."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict

import numpy as np

from baselines_trace_rules import _canonical_kb_rows
from common import ARTIFACTS, DOCS, RECORDS, load_jsonl, write_json


SEED = 20260730


def _judge_majorities():
    path = RECORDS / "evidence_aware_judge_records.jsonl"
    if not path.exists():
        return {}
    grouped = defaultdict(list)
    for row in load_jsonl(path):
        if row["dataset"] == "kramabench_human":
            grouped[row["item_id"]].append(row["judge_label"])
    out = {}
    for item_id, labels in grouped.items():
        counts = Counter(labels)
        out[item_id] = max(
            ("unsupported", "unknown", "supported"),
            key=lambda label: counts[label],
        )
    return out


def human_cases():
    trace = {
        row["taskid"]: row
        for row in load_jsonl(RECORDS / "trace_rule_baselines.jsonl")
        if row["dataset"] == "kramabench_human"
    }
    with (DOCS / "annotation" / "KB_SUPPORT_ANNOTATION.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        annotations = list(csv.DictReader(handle))
    judge = _judge_majorities()
    cases = []
    for row in annotations:
        label = {"yes": 1, "no": 0}.get(row["human_support_label"].strip().lower())
        if label is None:
            continue
        decisions = trace[row["taskid"]]["decisions"]
        replay = row["checker_label"].strip().lower() == "supported"
        cases.append({
            "case_id": row["taskid"],
            "official_score": float(row["official_score"]),
            "human_label": label,
            "audit_label": (
                "supported" if replay else
                "unsupported" if row["checker_label"].strip().lower() == "unsupported"
                else "unknown"
            ),
            "execution_success": decisions["execution_success"],
            "citations_resolve": decisions["citations_resolve"],
            "unrestricted_replay_match": decisions["unrestricted_replay_match"],
            "replay_certificate": replay,
            "replay_or_closure_certificate": replay,
            "judge_supported": judge.get(row["taskid"]) == "supported",
            "judge_available": row["taskid"] in judge,
        })
    return cases


def all_kramabench_cases():
    trace = {
        row["case_id"]: row
        for row in load_jsonl(RECORDS / "trace_rule_baselines.jsonl")
        if row["dataset"] == "kramabench_all"
    }
    cases = []
    for row in _canonical_kb_rows():
        case_id = f"{row['taskid']}::{row['_setting']}"
        decisions = trace[case_id]["decisions"]
        audit_label = row.get("grounding_label") or "unknown"
        cases.append({
            "case_id": case_id,
            "official_score": (
                float(row["score"])
                if row.get("score") is not None and float(row["score"]) >= 0
                else None
            ),
            "human_label": None,
            "audit_label": audit_label,
            "execution_success": decisions["execution_success"],
            "citations_resolve": decisions["citations_resolve"],
            "unrestricted_replay_match": decisions["unrestricted_replay_match"],
            "replay_certificate": audit_label == "supported",
            "replay_or_closure_certificate": audit_label == "supported",
            "judge_supported": False,
            "judge_available": False,
        })
    return cases


def _stable_order(case):
    digest = hashlib.sha256(f"{SEED}:{case['case_id']}".encode()).hexdigest()
    return digest


def summarize(dataset, policy, cases, delivered):
    kept = [case for case, decision in zip(cases, delivered) if decision]
    escalated = [case for case, decision in zip(cases, delivered) if not decision]
    scored = [
        case["official_score"]
        for case in kept
        if case["official_score"] is not None and math.isfinite(case["official_score"])
    ]
    failures = [case for case in escalated if case["audit_label"] == "unsupported"]
    human_supported = [
        case for case in kept if case.get("human_label") == 1
    ]
    human_unsupported = [
        case for case in kept if case.get("human_label") == 0
    ]
    return {
        "dataset": dataset,
        "policy": policy,
        "n_total": len(cases),
        "n_delivered": len(kept),
        "n_escalated": len(escalated),
        "coverage": len(kept) / len(cases),
        "official_score_on_delivered": sum(scored) / len(scored) if scored else None,
        "official_score_n": len(scored),
        "human_supported_delivered": len(human_supported),
        "human_unsupported_delivered": len(human_unsupported),
        "explicit_failure_precision": len(failures) / len(escalated) if escalated else None,
        "unknown_rate": (
            sum(case["audit_label"] == "unknown" for case in kept) / len(kept)
            if kept else None
        ),
    }


def evaluate(dataset, cases):
    policies = {
        "deliver_all": [True] * len(cases),
        "deliver_if_execution_success": [case["execution_success"] for case in cases],
        "deliver_if_citations_resolve": [case["citations_resolve"] for case in cases],
        "deliver_if_unrestricted_replay_match": [
            case["unrestricted_replay_match"] for case in cases
        ],
        "deliver_if_replay_certificate": [
            case["replay_certificate"] for case in cases
        ],
        "deliver_if_replay_or_closure_certificate": [
            case["replay_or_closure_certificate"] for case in cases
        ],
    }
    if all(case["judge_available"] for case in cases):
        policies["deliver_if_evidence_aware_judge_supported"] = [
            case["judge_supported"] for case in cases
        ]
    target_n = sum(policies["deliver_if_replay_certificate"])
    chosen = {
        case["case_id"]
        for case in sorted(cases, key=_stable_order)[:target_n]
    }
    policies["random_matched_replay_coverage"] = [
        case["case_id"] in chosen for case in cases
    ]
    return [
        summarize(dataset, policy, cases, decisions)
        for policy, decisions in policies.items()
    ], policies


def bootstrap_official(cases, policies):
    audit = np.asarray(policies["deliver_if_replay_certificate"], dtype=bool)
    random = np.asarray(policies["random_matched_replay_coverage"], dtype=bool)
    scores = np.asarray([
        np.nan if case["official_score"] is None else case["official_score"]
        for case in cases
    ])
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(10_000):
        indices = rng.integers(0, len(cases), len(cases))
        a = scores[indices][audit[indices]]
        b = scores[indices][random[indices]]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) and len(b):
            deltas.append(float(a.mean() - b.mean()))
    return {
        "comparison": "replay_certificate_minus_random_matched_coverage",
        "bootstrap_seed": SEED,
        "bootstrap_repeats": len(deltas),
        "mean_delta": float(np.mean(deltas)),
        "percentile_95_interval": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def main():
    human = human_cases()
    all_kb = all_kramabench_cases()
    human_results, human_policies = evaluate("kramabench_human_clear", human)
    all_results, all_policies = evaluate("kramabench_all", all_kb)
    payload = {
        "schema_version": 1,
        "random_seed": SEED,
        "results": human_results + all_results,
        "official_score_note": (
            "Official score is deployment utility only, not support ground truth."
        ),
        "bootstrap_audit": bootstrap_official(all_kb, all_policies),
    }
    write_json(RECORDS / "delivery_policy_results.json", payload)
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Policy & Delivered & Coverage & Human unsupported & Delivered score \\",
        r"\midrule",
    ]
    for row in payload["results"]:
        score = (
            "--"
            if row["official_score_on_delivered"] is None
            else f"{row['official_score_on_delivered']:.3f}"
        )
        lines.append(
            f"{row['dataset']} & {row['policy'].replace('_', ' ')} & "
            f"{row['n_delivered']}/{row['n_total']} & {row['coverage']:.3f} & "
            f"{row['human_unsupported_delivered']} & {score} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ARTIFACTS / "table_delivery_policies.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
