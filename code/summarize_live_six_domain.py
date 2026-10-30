#!/usr/bin/env python3
"""Summarize the existing six-domain API traces without relabeling audit scope."""

from __future__ import annotations

import json
from collections import Counter

from common import ARTIFACTS, RECORDS, load_jsonl, write_json


DOMAINS = ("archeology", "astronomy", "biomedical", "environment", "legal", "wildfire")


def records(domain, arm):
    path = RECORDS / f"records_ledger_api_kramabench_{domain}_{arm}.jsonl"
    return load_jsonl(path) if path.exists() else []


def score(rows):
    values = [
        float(row["score"])
        for row in rows
        if row.get("score") is not None and float(row["score"]) >= 0
    ]
    return sum(values) / len(values) if values else None


def main():
    canonical = {}
    for path in RECORDS.glob("records_ledger_n2_kb_*_*.jsonl"):
        parts = path.stem.split("_")
        domain, setting = parts[-2], parts[-1]
        canonical[(domain, setting)] = load_jsonl(path)
    rows = []
    for domain in DOMAINS:
        oracle = records(domain, "oracle")
        discovery = records(domain, "B3")
        audit_rows = canonical.get((domain, "discovery"))
        if audit_rows is None:
            distribution = {"supported": 0, "unsupported": 0, "unknown": 0}
            applicable = 0
            scope = "out_of_scope_for_current_hermetic_replay"
        else:
            distribution = Counter(row.get("grounding_label", "unknown") for row in audit_rows)
            applicable = len(audit_rows)
            scope = "applicable"
        rows.append({
            "domain": domain,
            "n_oracle": len(oracle),
            "n_discovery_proxy": len(discovery),
            "oracle_score": score(oracle),
            "discovery_proxy_score": score(discovery),
            "selection_gap": (
                score(oracle) - score(discovery)
                if score(oracle) is not None and score(discovery) is not None
                else None
            ),
            "replay_applicable_n": applicable,
            "replay_scope": scope,
            "certificate": distribution.get("supported", 0),
            "explicit_failure": distribution.get("unsupported", 0),
            "unknown": distribution.get("unknown", 0),
            "tokens": sum(int(row.get("tokens_in", 0)) + int(row.get("tokens_out", 0)) for row in discovery),
            "tool_calls": sum(int(row.get("tool_calls", 0)) for row in discovery),
            "latency_s": sum(float(row.get("latency_s", 0.0)) for row in discovery),
        })
    payload = {
        "schema_version": 1,
        "rows": rows,
        "arm_note": (
            "The existing API B3 trace is reported as a discovery-arm proxy; it is "
            "not relabeled as the canonical N2 discovery audit record."
        ),
    }
    write_json(RECORDS / "live_six_domain_summary.json", payload)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Domain & N & Oracle & Discovery proxy & Replay N & Cert./fail/unknown & Tokens \\",
        r"\midrule",
    ]
    for row in rows:
        oracle = "--" if row["oracle_score"] is None else f"{row['oracle_score']:.3f}"
        discovery = (
            "--" if row["discovery_proxy_score"] is None
            else f"{row['discovery_proxy_score']:.3f}"
        )
        lines.append(
            f"{row['domain']} & {row['n_discovery_proxy']} & {oracle} & {discovery} & "
            f"{row['replay_applicable_n']} & "
            f"{row['certificate']}/{row['explicit_failure']}/{row['unknown']} & "
            f"{row['tokens']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ARTIFACTS / "table_live_six_domain.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
