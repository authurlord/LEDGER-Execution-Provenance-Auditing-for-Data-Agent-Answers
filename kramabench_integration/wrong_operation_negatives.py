"""LEDGER P0-4: same-operand wrong-operation negatives for ConvFinQA.

CPU-only, deterministic, no LLM.  This is the hard negative distribution called
out by review: the cited operands are correct, but the operation is wrong.  It
uses the exact E13 frozen point and split (seed=60, rest=388).

Run:
  python3 -m kramabench_integration.wrong_operation_negatives
"""
from __future__ import annotations

from collections import Counter
import json
import math
import os
import re
from typing import Any

from . import convfinqa_gold as G
from . import cost_accounting as CA
from . import provenance as PV
from .convfinqa_constrained_calib import swap_negative

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_RECORDS = os.path.join(REPO, "records", "convfinqa_wrong_operation_negatives.jsonl")
OUT_SUMMARY = os.path.join(REPO, "records", "convfinqa_wrong_operation_negatives_summary.json")
OUT_DOC = os.path.join(REPO, "docs", "WRONG_OPERATION_NEGATIVES.md")

FROZEN = {
    "ops": ("sum", "diff", "ratio", "pctchg", "prod"),
    "chain": "full",
    "max_depth": 2,
    "tol_rel": 1e-4,
    "percent": True,
}

OP_TO_FROZEN = {
    "add": "sum",
    "subtract": "diff",
    "multiply": "prod",
    "divide": "ratio",
}
SWAPS = {
    "add": ("subtract",),
    "subtract": ("add",),
    "multiply": ("divide",),
    "divide": ("multiply",),
}
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _frozen_json() -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in FROZEN.items()}


def _tol_ne(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) > max(FROZEN["tol_rel"] * abs(float(b)), 1e-9)


def _const_value(arg: str) -> float | None:
    m = NUM_RE.search(arg.replace("m1", "-1"))
    return float(m.group()) if m else None


def _arg_value(arg: str, values: list[float]) -> float | None:
    arg = arg.strip()
    if arg.startswith("#"):
        try:
            idx = int(arg[1:])
        except ValueError:
            return None
        return values[idx] if 0 <= idx < len(values) else None
    if arg.startswith("const_"):
        return _const_value(arg)
    if re.fullmatch(r"-?\d+(?:\.\d+)?%?", arg):
        v = float(arg.rstrip("%"))
        return v / 100.0 if arg.endswith("%") else v
    return None


def _apply(op: str, args: list[float]) -> float | None:
    if len(args) != 2:
        return None
    a, b = args
    try:
        if op == "add":
            v = a + b
        elif op == "subtract":
            v = a - b
        elif op == "multiply":
            v = a * b
        elif op == "divide":
            if b == 0:
                return None
            v = a / b
        elif op == "exp":
            v = a ** b
        else:
            return None
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return v if math.isfinite(v) and abs(v) < 1e15 else None


def _eval_calls(calls: list[tuple[str, list[str]]], replace: tuple[int, str] | None = None) -> tuple[float | None, list[float]]:
    values: list[float] = []
    for i, (op, raw_args) in enumerate(calls):
        op2 = replace[1] if replace and replace[0] == i else op
        args = [_arg_value(a, values) for a in raw_args]
        if any(a is None for a in args):
            return None, values
        v = _apply(op2, [float(a) for a in args if a is not None])
        if v is None:
            return None, values
        values.append(v)
    return (values[-1] if values else None), values


def _is_const_two(arg: str) -> bool:
    val = _arg_value(arg, [])
    return val is not None and abs(val - 2.0) <= 1e-12


def _mean_forget_divide_candidate(q: dict[str, Any], calls: list[tuple[str, list[str]]]) -> dict[str, Any] | None:
    """Detect divide(add(a,b), const_2) in ConvFinQA's sequential program form."""
    if len(calls) < 2:
        return None
    op0, args0 = calls[0]
    op1, args1 = calls[1]
    if op0 != "add" or op1 != "divide" or len(args0) != 2 or len(args1) != 2:
        return None
    if not (args1[0] == "#0" and _is_const_two(args1[1])):
        return None
    _, values = _eval_calls(calls)
    if not values:
        return None
    neg = values[0]
    if not _tol_ne(neg, q["exe_ans"]):
        return None
    return {
        "negative_value": neg,
        "construction": "mean_forget_divide",
        "construction_detail": "program begins divide(add(a,b), const_2); negative is add(a,b)",
        "gold_first_op": op0,
        "wrong_op": "add",
        "wrong_frozen_op": "sum",
        "wrong_op_in_frozen_ops": True,
        "mutated_depth": 1,
        "mutated_program": f"{calls[0][0]}({','.join(calls[0][1])})",
    }


def wrong_operation_negative(q: dict[str, Any]) -> dict[str, Any] | None:
    calls, _, _, _, ok = G.parse_program(q.get("program", ""))
    if not ok or not calls or q.get("exe_ans") is None:
        return None

    special = _mean_forget_divide_candidate(q, calls)
    if special:
        return special

    first_op, _ = calls[0]
    for wrong_op in SWAPS.get(first_op, ()):
        neg, _ = _eval_calls(calls, replace=(0, wrong_op))
        if neg is None or not _tol_ne(neg, q["exe_ans"]):
            continue
        frozen_op = OP_TO_FROZEN.get(wrong_op)
        mutated = list(calls)
        mutated[0] = (wrong_op, mutated[0][1])
        return {
            "negative_value": neg,
            "construction": f"first_op_{first_op}_to_{wrong_op}",
            "construction_detail": "replace the first gold program operation with a same-arity wrong operation",
            "gold_first_op": first_op,
            "wrong_op": wrong_op,
            "wrong_frozen_op": frozen_op,
            "wrong_op_in_frozen_ops": bool(frozen_op in FROZEN["ops"]),
            "mutated_depth": len(calls),
            "mutated_program": ", ".join(f"{op}({','.join(args)})" for op, args in mutated),
        }
    return None


def _check(pool: PV.ConstrainedPool, value: float, q: dict[str, Any]) -> str | None:
    return pool.check(
        value,
        tol_rel=FROZEN["tol_rel"],
        percent=FROZEN["percent"],
        percent_ctx=q["percent_ctx"],
        max_depth=FROZEN["max_depth"],
    )


def _pct(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _summarize_detection(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    n = len(rows)
    flagged = sum(1 for r in rows if r["negative_flagged"])
    accepted = n - flagged
    return {
        "label": label,
        "n": n,
        "unsupported_detection": _pct(flagged, n),
        "unsupported_detection_counts": f"{flagged}/{n}",
        "false_support_rate": _pct(accepted, n),
        "false_support_counts": f"{accepted}/{n}",
    }


def _stratify_wrong_ops(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for key in (True, False):
        part = [r for r in rows if r["wrong_op_in_frozen_ops"] is key]
        s = _summarize_detection(part, f"wrong_op_in_frozen_ops={key}")
        s["accepted_counts"] = s["false_support_counts"]
        out.append(s)
    return out


def _stratify_frozen_depth(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strata = [
        ("in_frozen_ops_and_within_frozen_depth",
         lambda r: r["wrong_op_in_frozen_ops"] and r["mutated_depth"] <= FROZEN["max_depth"]),
        ("in_frozen_ops_but_over_frozen_depth",
         lambda r: r["wrong_op_in_frozen_ops"] and r["mutated_depth"] > FROZEN["max_depth"]),
        ("outside_frozen_ops", lambda r: not r["wrong_op_in_frozen_ops"]),
    ]
    out = []
    for label, pred in strata:
        part = [r for r in rows if pred(r)]
        s = _summarize_detection(part, label)
        s["accepted_counts"] = s["false_support_counts"]
        out.append(s)
    return out


def _construction_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(r["construction"] for r in rows).items()))


def _build_rows(rest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    wrong_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    continuity = rest[:300]

    for i, q in enumerate(rest):
        evaluable = q["exe_ans"] is not None and bool(q["operands"])
        if not evaluable:
            skipped["not_evaluable"] += 1
            continue

        pool = PV.ConstrainedPool(q["operands"], q["consts"], FROZEN["ops"], chain=FROZEN["chain"])
        pos_depth = _check(pool, q["exe_ans"], q)

        swap_val = swap_negative(continuity, i) if i < len(continuity) else swap_negative(rest, i)
        swap_depth = _check(pool, swap_val, q)
        swap_rows.append({
            "uid": q["uid"],
            "qhash": q["qhash"],
            "full_remainder_index": i,
            "negative_distribution": "cross_example_swap",
            "negative_value": swap_val,
            "negative_flagged": not bool(swap_depth),
            "negative_matched_depth": swap_depth,
            "positive_matched_depth": pos_depth,
        })

        neg = wrong_operation_negative(q)
        if not neg:
            skipped["not_constructible"] += 1
            continue
        neg_depth = _check(pool, neg["negative_value"], q)
        wrong_rows.append({
            "uid": q["uid"],
            "qhash": q["qhash"],
            "full_remainder_index": i,
            "negative_distribution": "same_operand_wrong_operation",
            "negative_value": neg["negative_value"],
            "negative_flagged": not bool(neg_depth),
            "negative_matched_depth": neg_depth,
            "positive_matched_depth": pos_depth,
            "exe_ans": q["exe_ans"],
            "program": q["program"],
            "depth_gold": q["depth"],
            "percent_ctx": q["percent_ctx"],
            "operands": q["operands"],
            "consts": q["consts"],
            **neg,
        })
    return wrong_rows, swap_rows, skipped


def _write_records(rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(OUT_RECORDS), exist_ok=True)
    writer = CA.RecordWriter(OUT_RECORDS)
    for r in rows:
        glabel = "unsupported" if r["negative_flagged"] else "supported"
        writer.write(CA.build_record(
            taskid=f"{r['uid']}::wrong_operation_negative",
            domain="convfinqa-dev",
            system="checker_wrong_operation",
            sources_selected=[],
            gold_sources=[],
            pred=r["negative_value"],
            gold="WRONG_OPERATION",
            score=-1,
            meter=None,
            grounding_label=glabel,
            provenance=[{"cited_operands": r["operands"], "cited_consts": r["consts"]}],
            depth_gold=r["depth_gold"],
            percent_ctx=r["percent_ctx"],
            matched_depth=r["negative_matched_depth"],
            full_remainder_index=r["full_remainder_index"],
            frozen_point=_frozen_json(),
            negative_distribution=r["negative_distribution"],
            negative_label_source="construction:same operands with wrong operation; no checker-in-the-loop",
            exe_ans=r["exe_ans"],
            program=r["program"],
            mutated_program=r["mutated_program"],
            construction=r["construction"],
            construction_detail=r["construction_detail"],
            gold_first_op=r["gold_first_op"],
            wrong_op=r["wrong_op"],
            wrong_frozen_op=r["wrong_frozen_op"],
            wrong_op_in_frozen_ops=r["wrong_op_in_frozen_ops"],
            mutated_depth=r["mutated_depth"],
            constrained_negative_flagged=r["negative_flagged"],
            negative_matched_depth=r["negative_matched_depth"],
            constrained_positive_hit=bool(r["positive_matched_depth"]),
            constrained_matched_depth=r["positive_matched_depth"],
        ))


def _markdown_table(rows: list[list[Any]]) -> str:
    header = rows[0]
    body = rows[1:]
    lines = ["| " + " | ".join(map(str, header)) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in body)
    return "\n".join(lines)


def _write_doc(summary: dict[str, Any]) -> None:
    comp = summary["comparison"]
    strata = summary["wrong_operation_strata"]
    depth_strata = summary["wrong_operation_depth_strata"]
    table1 = _markdown_table([
        ["negative distribution", "n", "unsupported detection", "false support"],
        [comp["cross_example_swap"]["label"], comp["cross_example_swap"]["n"],
         comp["cross_example_swap"]["unsupported_detection_counts"], comp["cross_example_swap"]["false_support_counts"]],
        [comp["same_operand_wrong_operation"]["label"], comp["same_operand_wrong_operation"]["n"],
         comp["same_operand_wrong_operation"]["unsupported_detection_counts"],
         comp["same_operand_wrong_operation"]["false_support_counts"]],
    ])
    table2 = _markdown_table(
        [["stratum", "n", "unsupported detection", "accepted / false support"]]
        + [[s["label"], s["n"], s["unsupported_detection_counts"], s["false_support_counts"]] for s in strata]
    )
    table3 = _markdown_table(
        [["stratum", "n", "unsupported detection", "accepted / false support"]]
        + [[s["label"], s["n"], s["unsupported_detection_counts"], s["false_support_counts"]] for s in depth_strata]
    )
    text = f"""# Wrong-Operation Negatives

This CPU-only audit reruns the E13 frozen ConvFinQA closure point on a harder
negative distribution.  E13's unsupported negatives were cross-example answer
swaps.  Here, the operands remain the gold cited operands, but the operation is
wrong by construction: mean-as-sum (`divide(add(a,b), const_2)` -> `add(a,b)`)
or a same-arity first-op replacement (`add`/`subtract`, `multiply`/`divide`).

Labels are assigned by the deterministic construction, not by the checker.  A
wrong-operation negative is unsupported because it is not the gold derivation,
even when its value is derivable from the same operands under the closure.

## Setup

- Data: `convfinqa_gold.load_qas()` with `split_seed_rest`, same seed as E13.
- Split: `{json.dumps(summary["split_stats"], sort_keys=True)}`.
- Frozen point: `{json.dumps(summary["frozen_point"], sort_keys=True)}`.
- Constructible wrong-operation negatives: {summary["wrong_operation_constructible"]}/{summary["evaluable_rest"]}.
- Skips: `{json.dumps(summary["skipped"], sort_keys=True)}`.

## Detection Comparison

{table1}

## Frozen-Op Stratification

{table2}

The raw frozen-op stratum is not sufficient by itself because E13 freezes both
operator vocabulary and `max_depth=2`.  The depth-aware split is:

{table3}

Construction counts: `{json.dumps(summary["construction_counts"], sort_keys=True)}`.

## Interpretation

This is the expected blind spot.  Closure acceptance verifies derivability of a
stated value from cited operands under the bounded operator set; it does not
verify that the stated value came from the agent's actual derivation.  Same
operands plus a wrong in-vocabulary operation can therefore be accepted.  In
this run, every wrong operation that was both in the frozen operator set and
within the frozen depth budget was accepted (347/347).  The 9 detected
wrong-operation negatives are all in-vocabulary but over the frozen depth budget
(depth-3 programs under a depth-2 checker), so they are not a contradiction of
the closure prediction.

In live LEDGER mode, this class is handled by hermetic re-execution rather than
closure acceptance.  E12/E9 exercise the mechanism: the final program is rerun
under the citation-restricted file view, and the recomputed value is compared
directly to the stated answer.  If the agent computed a sum while stating or
requiring a mean, re-exec exposes the actual output mismatch as class (a);
closure is only the approximation used when no executable trace is available.
"""
    os.makedirs(os.path.dirname(OUT_DOC), exist_ok=True)
    with open(OUT_DOC, "w") as f:
        f.write(text)


def main() -> dict[str, Any]:
    qas = G.load_qas()
    seed, rest, stats = G.split_seed_rest(qas)
    old_seed, old_scale, old_stats = G.split_seed_scale(qas)
    assert [q["qhash"] for q in seed] == [q["qhash"] for q in old_seed], "seed drift vs split_seed_scale"
    assert [q["qhash"] for q in rest[:300]] == [q["qhash"] for q in old_scale], "n=300 continuity split drift"
    assert not ({q["qhash"] for q in seed} & {q["qhash"] for q in rest}), "seed/rest overlap"

    wrong_rows, swap_rows, skipped = _build_rows(rest)
    _write_records(wrong_rows)

    summary = {
        "split_stats": stats,
        "n300_continuity_split_stats": old_stats,
        "frozen_point": _frozen_json(),
        "evaluable_rest": len(swap_rows),
        "wrong_operation_constructible": len(wrong_rows),
        "skipped": dict(sorted(skipped.items())),
        "comparison": {
            "cross_example_swap": _summarize_detection(swap_rows, "cross_example_swap"),
            "same_operand_wrong_operation": _summarize_detection(wrong_rows, "same_operand_wrong_operation"),
        },
        "wrong_operation_strata": _stratify_wrong_ops(wrong_rows),
        "wrong_operation_depth_strata": _stratify_frozen_depth(wrong_rows),
        "construction_counts": _construction_counts(wrong_rows),
        "note": (
            "Closure acceptance checks derivability from cited operands, not derivation identity. "
            "Live traces use hermetic re-execution; the recomputed value is authoritative."
        ),
    }
    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=1)
    _write_doc(summary)

    print(json.dumps(summary, indent=1))
    print("->", OUT_RECORDS)
    print("->", OUT_SUMMARY)
    print("->", OUT_DOC)
    return summary


if __name__ == "__main__":
    main()
