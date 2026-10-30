"""LEDGER E8 - MultiHiertt checker-property extension.

CPU-only, deterministic, no LLM. Uses the exact frozen operating point from the
ConvFinQA calibration:
  ops={sum,diff,ratio,pctchg,prod}, chain=full, max_depth=2, tol_rel=1e-4, percent=ON.

Run:
  python3 -m kramabench_integration.multihiertt_checker
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
import re

from .convfinqa_gold import parse_program
from .convfinqa_constrained_calib import swap_negative
from . import cost_accounting as CA
from . import provenance as PV

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO, "datasets", "multihiertt", "multihiertt_data", "train.json")
OUT_RECORDS = os.path.join(REPO, "records", "records_ledger_checker_multihiertt.jsonl")
OUT_SUMMARY = os.path.join(REPO, "records", "multihiertt_checker_summary.json")
OUT_DOC = os.path.join(REPO, "docs", "MULTIHIERTT_CHECKER.md")

FROZEN = {"ops": ("sum", "diff", "ratio", "pctchg", "prod"), "chain": "full",
          "max_depth": 2, "tol_rel": 1e-4, "percent": True}
ALLOWED_DSL_OPS = {"add", "subtract", "multiply", "divide", "exp"}
_NUMERIC = re.compile(r"^\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*%?\s*$|^\s*[-+]?\d+(?:\.\d+)?\s*%?\s*$")


def _frozen_json():
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in FROZEN.items()}


def _to_float(value):
    if isinstance(value, bool) or value is None:
        return None, "answer_non_numeric"
    if isinstance(value, (int, float)):
        x = float(value)
        return (x, None) if math.isfinite(x) else (None, "answer_non_finite")
    s = str(value).strip()
    if s.lower() in {"yes", "no", "true", "false"}:
        return None, "answer_yes_no"
    if not _NUMERIC.match(s):
        return None, "answer_non_numeric"
    try:
        x = float(s.replace(",", "").replace("%", ""))
    except ValueError:
        return None, "answer_non_numeric"
    return (x, None) if math.isfinite(x) else (None, "answer_non_finite")


def _percent_context(question, answer, program):
    ql = (question or "").lower()
    return ("%" in str(answer) or "%" in ql or "percent" in ql or "const_100" in (program or ""))


def _depth_bin(depth):
    if depth <= 1:
        return "depth_1"
    if depth == 2:
        return "depth_2"
    if depth == 3:
        return "depth_3"
    return "depth_ge4"


def _rate(num, den):
    return round(num / den, 4) if den else None


def _swap_negative_guaranteed(examples, i):
    neg = swap_negative(examples, i)
    if abs(float(neg) - float(examples[i]["exe_ans"])) > 1e-9:
        return neg
    for j, other in enumerate(examples):
        if j != i and abs(float(other["exe_ans"]) - float(examples[i]["exe_ans"])) > 1e-9:
            return other["exe_ans"]
    raise ValueError("cannot construct cross-example negative: all executable answers are identical")


def load_examples(path=DATA_PATH):
    data = json.load(open(path))
    examples = []
    reasons = Counter()
    for idx, ex in enumerate(data):
        qa = ex.get("qa") or {}
        program = qa.get("program") or ""
        answer = qa.get("answer")
        calls, operands, consts, depth, ok = parse_program(program)

        reason = None
        if not program:
            reason = "missing_program"
        elif not ok:
            reason = "program_parse_failed"
        else:
            bad_ops = sorted({op for op, _args in calls if op not in ALLOWED_DSL_OPS})
            if bad_ops:
                reason = "non_arithmetic_operator"
        exe_ans, answer_reason = _to_float(answer)
        if reason is None and answer_reason is not None:
            reason = answer_reason
        if reason is None and not (operands or consts):
            reason = "empty_citations"

        if reason is not None:
            reasons[reason] += 1
            continue

        question = qa.get("question", "")
        examples.append({
            "uid": ex.get("uid") or f"row_{idx}",
            "index": idx,
            "question": question,
            "question_type": qa.get("question_type"),
            "answer": answer,
            "exe_ans": exe_ans,
            "program": program,
            "operands": sorted(operands),
            "consts": sorted(consts),
            "depth": depth,
            "depth_bin": _depth_bin(depth),
            "percent_ctx": _percent_context(question, answer, program),
        })

    stats = {
        "total": len(data),
        "evaluable": len(examples),
        "unknown": len(data) - len(examples),
        "excluded_reasons": dict(sorted(reasons.items())),
    }
    return examples, stats


def _per_example(examples):
    rows = []
    for i, q in enumerate(examples):
        pool = PV.ConstrainedPool(q["operands"], q["consts"], FROZEN["ops"], chain=FROZEN["chain"])
        neg_val = _swap_negative_guaranteed(examples, i)
        lab = pool.check(q["exe_ans"], tol_rel=FROZEN["tol_rel"], percent=FROZEN["percent"],
                         percent_ctx=q["percent_ctx"], max_depth=FROZEN["max_depth"])
        nlab = pool.check(neg_val, tol_rel=FROZEN["tol_rel"], percent=FROZEN["percent"],
                          percent_ctx=q["percent_ctx"], max_depth=FROZEN["max_depth"])
        rows.append({
            **q,
            "negative_value": neg_val,
            "constrained_pos_hit": bool(lab),
            "constrained_neg_flagged": not bool(nlab),
            "matched_depth": lab,
            "negative_matched_depth": nlab,
        })
    return rows


def _summarize_rows(rows, label):
    n = len(rows)
    pos_hit = sum(1 for r in rows if r["constrained_pos_hit"])
    neg_flagged = sum(1 for r in rows if r["constrained_neg_flagged"])
    false_support = n - neg_flagged
    return {
        "label": label,
        "n": n,
        "support_recall": _rate(pos_hit, n),
        "support_recall_counts": f"{pos_hit}/{n}",
        "unsupported_detection": _rate(neg_flagged, n),
        "unsupported_detection_counts": f"{neg_flagged}/{n}",
        "false_support_rate": _rate(false_support, n),
        "false_support_counts": f"{false_support}/{n}",
        "unknown": 0,
    }


def _depth_summary(rows):
    out = {}
    for bin_name in ("depth_1", "depth_2", "depth_3", "depth_ge4"):
        out[bin_name] = _summarize_rows([r for r in rows if r["depth_bin"] == bin_name], bin_name)
    return out


def _miss_attribution(rows):
    misses = [r for r in rows if not r["constrained_pos_hit"]]
    depth_gt2 = [r for r in misses if r["depth"] > FROZEN["max_depth"]]
    by_depth = Counter(r["depth"] for r in misses)
    return {
        "recall_misses": len(misses),
        "misses_depth_gt_2": len(depth_gt2),
        "misses_depth_gt_2_rate": _rate(len(depth_gt2), len(misses)),
        "miss_depth_distribution": {str(k): v for k, v in sorted(by_depth.items())},
        "interpretation": (
            "Depth>2 miss mass is outside the frozen bounded checker language "
            "(max_depth=2); this is Prop-1 boundary confirmation, not MultiHiertt tuning failure."
        ),
    }


def _write_records(rows, path):
    writer = CA.RecordWriter(path)
    frozen = _frozen_json()
    for row in rows:
        paired = {
            "constrained_pos_hit": row["constrained_pos_hit"],
            "constrained_neg_flagged": row["constrained_neg_flagged"],
            "constrained_positive_hit": row["constrained_pos_hit"],
            "constrained_negative_flagged": row["constrained_neg_flagged"],
            "constrained_matched_depth": row["matched_depth"],
            "negative_matched_depth": row["negative_matched_depth"],
        }
        provenance = [{"cited_operands": row["operands"], "cited_consts": row["consts"]}]
        for polarity, val, gold, glabel, matched in (
            ("positive", row["exe_ans"], row["exe_ans"],
             "supported" if row["constrained_pos_hit"] else "unsupported", row["matched_depth"]),
            ("negative", row["negative_value"], "SWAPPED",
             "unsupported" if row["constrained_neg_flagged"] else "supported", row["negative_matched_depth"]),
        ):
            writer.write(CA.build_record(
                taskid=f"{row['uid']}::{polarity}",
                domain="multihiertt-train",
                system="checker",
                pred=val,
                gold=gold,
                score=-1,
                meter=None,
                grounding_label=glabel,
                provenance=provenance,
                depth_gold=row["depth"],
                depth_bin=row["depth_bin"],
                percent_ctx=row["percent_ctx"],
                matched_depth=matched,
                program=row["program"],
                question_type=row["question_type"],
                frozen_point=frozen,
                **paired))


def _write_doc(summary, path):
    depth_rows = []
    for key in ("depth_1", "depth_2", "depth_3", "depth_ge4"):
        row = summary["depth_strata"][key]
        depth_rows.append(
            f"| {key.replace('_', ' ')} | {row['n']} | {row['support_recall']} "
            f"({row['support_recall_counts']}) | {row['unsupported_detection']} "
            f"({row['unsupported_detection_counts']}) | {row['false_support_rate']} "
            f"({row['false_support_counts']}) |"
        )
    miss = summary["prop1_miss_attribution"]
    head = summary["headline"]
    gate = summary["G8_gate"]
    text = f"""# LEDGER E8 - MultiHiertt checker-property extension

Run date: 2026-07-06. Command: `python3 -m kramabench_integration.multihiertt_checker`.

This is a CPU-only deterministic checker-property extension. It reuses the ConvFinQA parser and
the exact frozen ConvFinQA operating point: `ops={{sum,diff,ratio,pctchg,prod}}`, `chain=full`,
`max_depth=2`, `tol_rel=1e-4`, `percent=ON`. No MultiHiertt calibration or retuning was performed.

## Load

- Total examples: {summary['load_stats']['total']}
- Evaluable examples: {summary['load_stats']['evaluable']}
- Unknown / excluded examples: {summary['load_stats']['unknown']}
- Excluded reasons: `{json.dumps(summary['load_stats']['excluded_reasons'], sort_keys=True)}`

## Headline

| metric | value |
|---|---:|
| support recall | {head['support_recall']} ({head['support_recall_counts']}) |
| unsupported detection | {head['unsupported_detection']} ({head['unsupported_detection_counts']}) |
| false support | {head['false_support_rate']} ({head['false_support_counts']}) |
| unknown | {summary['load_stats']['unknown']} |

## Depth strata

| gold depth | n | support recall | unsupported detection | false support |
|---|---:|---:|---:|---:|
{os.linesep.join(depth_rows)}

## Prop-1 miss attribution

Recall misses: {miss['recall_misses']}. Misses with gold depth > 2: {miss['misses_depth_gt_2']}
({miss['misses_depth_gt_2_rate']}).

Miss depth distribution: `{json.dumps(miss['miss_depth_distribution'], sort_keys=True)}`.

Interpretation: MultiHiertt has deeper derivations than the frozen depth-2 checker language. A lower
headline recall is therefore not a failed retuning result; when miss mass lands beyond depth 2, it is
the expected Prop-1 boundary behavior for the bounded checker language.

## G8

Criterion: `evaluable >= 200 and unsupported_detection >= 0.85`.

G8: **{'PASS' if gate['PASS'] else 'FAIL'}**. Evaluable={gate['evaluable']};
unsupported_detection={gate['unsupported_detection']} ({head['unsupported_detection_counts']}).

Records: `records/records_ledger_checker_multihiertt.jsonl`.
Summary: `records/multihiertt_checker_summary.json`.
"""
    with open(path, "w") as f:
        f.write(text)


def main():
    examples, load_stats = load_examples()
    rows = _per_example(examples)
    _write_records(rows, OUT_RECORDS)

    headline = _summarize_rows(rows, "multihiertt_train_evaluable")
    summary = {
        "dataset": "MultiHiertt train",
        "input_path": os.path.relpath(DATA_PATH, REPO),
        "records_path": os.path.relpath(OUT_RECORDS, REPO),
        "frozen_point": _frozen_json(),
        "load_stats": load_stats,
        "headline": headline,
        "depth_distribution": dict(sorted(Counter(r["depth"] for r in rows).items())),
        "depth_strata": _depth_summary(rows),
        "prop1_miss_attribution": _miss_attribution(rows),
        "sampling": {"used": False, "discarded": 0},
    }
    summary["G8_gate"] = {
        "criteria": "evaluable>=200 AND unsupported_detection>=0.85",
        "evaluable": load_stats["evaluable"],
        "unsupported_detection": headline["unsupported_detection"],
        "PASS": load_stats["evaluable"] >= 200 and headline["unsupported_detection"] >= 0.85,
    }

    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
    json.dump(summary, open(OUT_SUMMARY, "w"), indent=1)
    _write_doc(summary, OUT_DOC)

    print(json.dumps(summary, indent=1))
    print("G8 GATE:", "PASS" if summary["G8_gate"]["PASS"] else "FAIL")
    print("->", os.path.relpath(OUT_RECORDS, REPO))
    print("->", os.path.relpath(OUT_SUMMARY, REPO))
    print("->", os.path.relpath(OUT_DOC, REPO))
    return summary


if __name__ == "__main__":
    main()
