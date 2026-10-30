"""LEDGER E5 — paired statistics for the ConvFinQA checker.

Reads records/records_ledger_checker_convfinqa_full.jsonl, uses one positive row per example
for paired constrained/free fields, and writes:
  records/convfinqa_paired_stats.json
  docs/STATS_CONVFINQA.md

Run:
  python3 -m kramabench_integration.stats_convfinqa_checker
"""
from __future__ import annotations

import json
import os

from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_RECORDS = os.path.join(REPO, "records", "records_ledger_checker_convfinqa_full.jsonl")
OUT_JSON = os.path.join(REPO, "records", "convfinqa_paired_stats.json")
OUT_MD = os.path.join(REPO, "docs", "STATS_CONVFINQA.md")


def _ci(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="beta")
    return [round(float(lo), 3), round(float(hi), 3)]


def _rate(k, n):
    return round(k / n, 4) if n else None


def _load_examples(path=IN_RECORDS):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run python3 -m kramabench_integration.convfinqa_checker_fulln first")
    examples = []
    seen = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("taskid", "").endswith("::positive"):
                continue
            key = r["taskid"].rsplit("::", 1)[0]
            if key in seen:
                raise ValueError(f"duplicate positive row for {key}")
            seen.add(key)
            examples.append(r)
    examples.sort(key=lambda r: r.get("full_remainder_index", 10**9))
    return examples


def _metric_block(name, k, n):
    return {"metric": name, "count": k, "n": n, "rate": _rate(k, n), "ci95_beta": _ci(k, n)}


def _mcnemar_detection(rows):
    use = [r for r in rows if r.get("constrained_neg_flagged") is not None]
    both = sum(1 for r in use if r["constrained_neg_flagged"] and r["free_neg_flagged"])
    constrained_only = sum(1 for r in use if r["constrained_neg_flagged"] and not r["free_neg_flagged"])
    free_only = sum(1 for r in use if (not r["constrained_neg_flagged"]) and r["free_neg_flagged"])
    neither = sum(1 for r in use if (not r["constrained_neg_flagged"]) and (not r["free_neg_flagged"]))
    table = [[both, constrained_only], [free_only, neither]]
    res = mcnemar(table, exact=True)
    c_rate = sum(1 for r in use if r["constrained_neg_flagged"]) / len(use) if use else 0.0
    f_rate = sum(1 for r in use if r["free_neg_flagged"]) / len(use) if use else 0.0
    return {
        "n_paired_evaluable": len(use),
        "discordant_table": {
            "both_flagged": both,
            "constrained_only_flagged": constrained_only,
            "free_only_flagged": free_only,
            "neither_flagged": neither,
            "matrix_constrained_rows_free_cols": table,
        },
        "mcnemar_exact_p": float(res.pvalue),
        "paired_effect_delta": round(c_rate - f_rate, 4),
        "constrained_detection_rate_on_paired": round(c_rate, 4),
        "free_detection_rate_on_paired": round(f_rate, 4),
    }


def _summarize(rows, label):
    constrained_eval = [r for r in rows if r.get("constrained_pos_hit") is not None]
    n_eval = len(constrained_eval)
    n_all = len(rows)
    out = {
        "label": label,
        "n_all": n_all,
        "n_constrained_evaluable": n_eval,
        "unknown": n_all - n_eval,
        "metrics": {
            "constrained_support_recall": _metric_block(
                "constrained_support_recall",
                sum(1 for r in constrained_eval if r["constrained_pos_hit"]),
                n_eval,
            ),
            "constrained_unsupported_detection": _metric_block(
                "constrained_unsupported_detection",
                sum(1 for r in constrained_eval if r["constrained_neg_flagged"]),
                n_eval,
            ),
            "free_support_recall": _metric_block(
                "free_support_recall",
                sum(1 for r in rows if r["free_pos_hit"]),
                n_all,
            ),
            "free_unsupported_detection": _metric_block(
                "free_unsupported_detection",
                sum(1 for r in rows if r["free_neg_flagged"]),
                n_all,
            ),
        },
        "paired_detection": _mcnemar_detection(rows),
    }
    return out


def _md_for(stats):
    lines = [
        "# ConvFinQA Checker Paired Statistics",
        "",
        "Exact intervals use Clopper-Pearson beta 95% CIs. McNemar uses the exact paired test on unsupported-detection verdicts; constrained unknown rows are excluded from the paired test.",
        "",
    ]
    for key in ("n300_continuity", "full_n_headline"):
        block = stats[key]
        lines += [f"## {block['label']}", ""]
        lines += ["| metric | count | n | rate | 95% CI |",
                  "|---|---:|---:|---:|---:|"]
        for m in block["metrics"].values():
            ci = m["ci95_beta"]
            lines.append(f"| {m['metric']} | {m['count']} | {m['n']} | {m['rate']:.4f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
        pd = block["paired_detection"]
        tab = pd["discordant_table"]
        lines += [
            "",
            f"McNemar exact p: `{pd['mcnemar_exact_p']:.6g}`",
            f"Paired detection effect Δ: `{pd['paired_effect_delta']:.4f}`",
            "",
            "| paired detection cell | count |",
            "|---|---:|",
            f"| both flagged | {tab['both_flagged']} |",
            f"| constrained-only flagged | {tab['constrained_only_flagged']} |",
            f"| free-only flagged | {tab['free_only_flagged']} |",
            f"| neither flagged | {tab['neither_flagged']} |",
            "",
        ]
    return "\n".join(lines) + "\n"


def main():
    rows = _load_examples()
    n300 = [r for r in rows if r.get("in_n300_continuity")]
    stats = {
        "input_records": os.path.relpath(IN_RECORDS, REPO),
        "n300_continuity": _summarize(n300, "n300_continuity"),
        "full_n_headline": _summarize(rows, "full_n_headline"),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    json.dump(stats, open(OUT_JSON, "w"), indent=1)
    open(OUT_MD, "w").write(_md_for(stats))

    print(json.dumps(stats, indent=1))
    p_full = stats["full_n_headline"]["paired_detection"]["mcnemar_exact_p"]
    print("G5 GATE:", "PASS" if p_full < 0.001 else "FAIL")
    print("->", OUT_JSON)
    print("->", OUT_MD)
    return stats


if __name__ == "__main__":
    main()
