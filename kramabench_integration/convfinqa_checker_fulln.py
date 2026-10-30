"""LEDGER E13 — ConvFinQA checker on the full disjoint remainder.

CPU-only, deterministic, no LLM. Uses the frozen operating point from G1:
  ops={sum,diff,ratio,pctchg,prod}, chain=full, max_depth=2, tol_rel=1e-4, percent=ON.

Run:
  python3 -m kramabench_integration.convfinqa_checker_fulln
"""
from __future__ import annotations

from collections import Counter
import json
import os

from . import convfinqa_gold as G
from . import cost_accounting as CA
from . import provenance as PV
from .convfinqa_constrained_calib import swap_negative
from .convfinqa_support_calib import derivable as free_closure_derivable

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = {"ops": ("sum", "diff", "ratio", "pctchg", "prod"), "chain": "full",
          "max_depth": 2, "tol_rel": 1e-4, "percent": True}


def _frozen_json():
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in FROZEN.items()}


def _empty_counts():
    return {"pos_hit": 0, "pos_miss": 0, "neg_flagged": 0, "neg_false_support": 0,
            "unknown": 0, "fc_pos_hit": 0, "fc_neg_flagged": 0}


def _add_counts(cnt, row):
    if row["constrained_pos_hit"] is None:
        cnt["unknown"] += 1
    else:
        cnt["pos_hit" if row["constrained_pos_hit"] else "pos_miss"] += 1
        cnt["neg_flagged" if row["constrained_neg_flagged"] else "neg_false_support"] += 1
    cnt["fc_pos_hit"] += 1 if row["free_pos_hit"] else 0
    cnt["fc_neg_flagged"] += 1 if row["free_neg_flagged"] else 0


def _summarize(rows, label):
    cnt = _empty_counts()
    for row in rows:
        _add_counts(cnt, row)
    n = len(rows)
    n_eval = n - cnt["unknown"]
    constrained = {
        "support_recall": round(cnt["pos_hit"] / n_eval, 4) if n_eval else None,
        "support_recall_counts": f"{cnt['pos_hit']}/{n_eval}",
        "unsupported_detection": round(cnt["neg_flagged"] / n_eval, 4) if n_eval else None,
        "unsupported_detection_counts": f"{cnt['neg_flagged']}/{n_eval}",
        "false_support_rate": round(cnt["neg_false_support"] / n_eval, 4) if n_eval else None,
        "false_support_counts": f"{cnt['neg_false_support']}/{n_eval}",
    }
    free = {
        "support_recall": round(cnt["fc_pos_hit"] / n, 4) if n else None,
        "support_recall_counts": f"{cnt['fc_pos_hit']}/{n}",
        "unsupported_detection": round(cnt["fc_neg_flagged"] / n, 4) if n else None,
        "unsupported_detection_counts": f"{cnt['fc_neg_flagged']}/{n}",
    }
    return {
        "label": label,
        "n": n,
        "n_evaluable": n_eval,
        "unknown": cnt["unknown"],
        "unknown_rate": round(cnt["unknown"] / n, 4) if n else None,
        "constrained": constrained,
        "free_closure_baseline": free,
    }


def _per_example(rest):
    rows = []
    continuity = rest[:300]
    for i, q in enumerate(rest):
        pool = PV.ConstrainedPool(q["operands"], q["consts"], FROZEN["ops"], chain=FROZEN["chain"])
        neg_val = swap_negative(continuity, i) if i < len(continuity) else swap_negative(rest, i)
        evaluable = q["exe_ans"] is not None and bool(q["operands"])
        if evaluable:
            lab = pool.check(q["exe_ans"], tol_rel=FROZEN["tol_rel"], percent=FROZEN["percent"],
                             percent_ctx=q["percent_ctx"], max_depth=FROZEN["max_depth"])
            nlab = pool.check(neg_val, tol_rel=FROZEN["tol_rel"], percent=FROZEN["percent"],
                              percent_ctx=q["percent_ctx"], max_depth=FROZEN["max_depth"])
        else:
            lab = nlab = None
        free_pos_depth = free_closure_derivable(q["exe_ans"], q["ctx_numbers"], 1e-3, "pairwise")
        free_neg_depth = free_closure_derivable(neg_val, q["ctx_numbers"], 1e-3, "pairwise")
        rows.append({
            "uid": q["uid"],
            "qhash": q["qhash"],
            "full_remainder_index": i,
            "in_n300_continuity": i < 300,
            "negative_value": neg_val,
            "depth_gold": q["depth"],
            "percent_ctx": q["percent_ctx"],
            "operands": q["operands"],
            "consts": q["consts"],
            "exe_ans": q["exe_ans"],
            "constrained_pos_hit": bool(lab) if evaluable else None,
            "constrained_neg_flagged": (not bool(nlab)) if evaluable else None,
            "free_pos_hit": bool(free_pos_depth),
            "free_neg_flagged": not bool(free_neg_depth),
            "matched_depth": lab,
            "constrained_matched_depth": lab,
            "free_matched_depth": free_pos_depth,
            "negative_matched_depth": nlab,
            "free_negative_matched_depth": free_neg_depth,
        })
    return rows


def _write_records(rows, path):
    W = CA.RecordWriter(path)
    for row in rows:
        paired = {
            "constrained_pos_hit": row["constrained_pos_hit"],
            "constrained_neg_flagged": row["constrained_neg_flagged"],
            "free_pos_hit": row["free_pos_hit"],
            "free_neg_flagged": row["free_neg_flagged"],
            "constrained_positive_hit": row["constrained_pos_hit"],
            "constrained_negative_flagged": row["constrained_neg_flagged"],
            "free_positive_hit": row["free_pos_hit"],
            "free_negative_flagged": row["free_neg_flagged"],
            "constrained_matched_depth": row["constrained_matched_depth"],
            "free_matched_depth": row["free_matched_depth"],
        }
        for polarity, val, gold, glabel, depth in (
            ("positive", row["exe_ans"], row["exe_ans"],
             "unknown" if row["constrained_pos_hit"] is None else
             ("supported" if row["constrained_pos_hit"] else "unsupported"),
             row["matched_depth"]),
            ("negative", row["negative_value"], "SWAPPED",
             "unknown" if row["constrained_neg_flagged"] is None else
             ("unsupported" if row["constrained_neg_flagged"] else "supported"),
             row["negative_matched_depth"]),
        ):
            W.write(CA.build_record(
                taskid=f"{row['uid']}::{polarity}", domain="convfinqa-dev", system="checker",
                pred=val, gold=gold, score=-1, meter=None, grounding_label=glabel,
                provenance=[{"cited_operands": row["operands"], "cited_consts": row["consts"]}],
                depth_gold=row["depth_gold"], percent_ctx=row["percent_ctx"],
                matched_depth=depth,
                full_remainder_index=row["full_remainder_index"],
                in_n300_continuity=row["in_n300_continuity"],
                frozen_point=_frozen_json(),
                **paired))


def main():
    qas = G.load_qas()
    seed, rest, stats = G.split_seed_rest(qas)
    old_seed, old_scale, old_stats = G.split_seed_scale(qas)
    assert [q["qhash"] for q in seed] == [q["qhash"] for q in old_seed], "seed drift vs split_seed_scale"
    assert [q["qhash"] for q in rest[:300]] == [q["qhash"] for q in old_scale], "n=300 continuity split drift"
    assert not ({q["qhash"] for q in seed} & {q["qhash"] for q in rest}), "seed/rest overlap"
    print("split:", json.dumps(stats))

    rows = _per_example(rest)
    out_records = os.path.join(REPO, "records", "records_ledger_checker_convfinqa_full.jsonl")
    _write_records(rows, out_records)

    summary = {
        "split_stats": stats,
        "n300_continuity_split_stats": old_stats,
        "frozen_point": _frozen_json(),
        "n300_continuity": _summarize(rows[:300], "n300_continuity"),
        "full_n_headline": _summarize(rows, "full_n_headline"),
        "note": "full-n is the entire unique disjoint remainder after the same 60-seed calibration split.",
    }
    out_summary = os.path.join(REPO, "records", "convfinqa_checker_fulln_summary.json")
    json.dump(summary, open(out_summary, "w"), indent=1)
    print(json.dumps(summary, indent=1))

    full = summary["full_n_headline"]["constrained"]
    passed = full["support_recall"] >= 0.92 and full["unsupported_detection"] >= 0.93
    print("G13 GATE:", "PASS" if passed else "FAIL")
    if not passed:
        added = rows[300:]
        print("added_88_depth_dist:", json.dumps(dict(sorted(Counter(r["depth_gold"] for r in added).items()))))
    print("->", out_records)
    print("->", out_summary)
    return summary


if __name__ == "__main__":
    main()
