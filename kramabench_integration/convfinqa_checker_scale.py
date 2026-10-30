"""LEDGER v2 STEP 2 — C2 main table: checker at scale (ConvFinQA disjoint n=300, CPU, no LLM).

Evaluates ONLY the FROZEN operating point from STEP 1 (G1) + the free-closure baseline row for
contrast. NO tuning here (test-set hygiene). Emits one JSONL record per (example, polarity) with
tokens=0, system="checker" (deterministic steps carry cost records too, per the v2 brief), plus
counts-not-just-rates summary and unknown_rate reported separately.

FROZEN POINT (from records/convfinqa_constrained_grid.json / LEDGER_G1_VERDICT.md):
  ops={sum,diff,ratio,pctchg,prod}, chain=full, max_depth=2, tol_rel=1e-4, percent=ON.

Run: conda run -n kramabench python -m kramabench_integration.convfinqa_checker_scale
"""
from __future__ import annotations
import os, json
from . import convfinqa_gold as G
from . import provenance as PV
from . import cost_accounting as CA
from .convfinqa_support_calib import derivable as free_closure_derivable
from .convfinqa_constrained_calib import swap_negative

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = {"ops": ("sum", "diff", "ratio", "pctchg", "prod"), "chain": "full",
          "max_depth": 2, "tol_rel": 1e-4, "percent": True}


def main():
    qas = G.load_qas()
    seed, scale, stats = G.split_seed_scale(qas)
    assert not ({q["qhash"] for q in seed} & {q["qhash"] for q in scale}), "seed/scale overlap"
    print("split:", json.dumps(stats))
    out_records = os.path.join(REPO, "records", "records_ledger_checker_convfinqa.jsonl")
    W = CA.RecordWriter(out_records)

    cnt = {"pos_hit": 0, "pos_miss": 0, "neg_flagged": 0, "neg_false_support": 0,
           "unknown": 0, "fc_pos_hit": 0, "fc_neg_flagged": 0}
    for i, q in enumerate(scale):
        pool = PV.ConstrainedPool(q["operands"], q["consts"], FROZEN["ops"], chain=FROZEN["chain"])
        neg_val = swap_negative(scale, i)
        evaluable = q["exe_ans"] is not None and bool(q["operands"])
        if not evaluable:
            cnt["unknown"] += 1
            lab = nlab = None
            polarity_rows = [("positive", q["exe_ans"], "unknown"), ("negative", neg_val, "unknown")]
        else:
            lab = pool.check(q["exe_ans"], tol_rel=FROZEN["tol_rel"], percent=FROZEN["percent"],
                             percent_ctx=q["percent_ctx"], max_depth=FROZEN["max_depth"])
            nlab = pool.check(neg_val, tol_rel=FROZEN["tol_rel"], percent=FROZEN["percent"],
                              percent_ctx=q["percent_ctx"], max_depth=FROZEN["max_depth"])
            cnt["pos_hit" if lab else "pos_miss"] += 1
            cnt["neg_false_support" if nlab else "neg_flagged"] += 1
            polarity_rows = [("positive", q["exe_ans"], "supported" if lab else "unsupported"),
                             ("negative", neg_val, "supported" if nlab else "unsupported")]
        # free-closure baseline (same example, context numbers)
        free_pos_depth = free_closure_derivable(q["exe_ans"], q["ctx_numbers"], 1e-3, "pairwise")
        free_neg_depth = free_closure_derivable(neg_val, q["ctx_numbers"], 1e-3, "pairwise")
        free_pos_hit = bool(free_pos_depth)
        free_neg_flagged = not bool(free_neg_depth)
        cnt["fc_pos_hit"] += 1 if free_pos_hit else 0
        cnt["fc_neg_flagged"] += 1 if free_neg_flagged else 0
        paired = {
            "constrained_pos_hit": bool(lab) if evaluable else None,
            "constrained_neg_flagged": (not bool(nlab)) if evaluable else None,
            "free_pos_hit": free_pos_hit,
            "free_neg_flagged": free_neg_flagged,
            "constrained_positive_hit": bool(lab) if evaluable else None,
            "constrained_negative_flagged": (not bool(nlab)) if evaluable else None,
            "free_positive_hit": free_pos_hit,
            "free_negative_flagged": free_neg_flagged,
            "constrained_matched_depth": lab,
            "free_matched_depth": free_pos_depth,
        }
        for polarity, val, glabel in polarity_rows:
            W.write(CA.build_record(
                taskid=f"{q['uid']}::{polarity}", domain="convfinqa-dev", system="checker",
                pred=val, gold=(q["exe_ans"] if polarity == "positive" else "SWAPPED"),
                score=-1, meter=None, grounding_label=glabel,
                provenance=[{"cited_operands": q["operands"], "cited_consts": q["consts"]}],
                depth_gold=q["depth"], percent_ctx=q["percent_ctx"],
                matched_depth=(lab if polarity == "positive" else nlab),
                **paired,
                frozen_point={k: (list(v) if isinstance(v, tuple) else v) for k, v in FROZEN.items()}))

    n_eval = len(scale) - cnt["unknown"]
    summary = {
        "n_scale": len(scale), "n_evaluable": n_eval,
        "unknown_rate": round(cnt["unknown"] / len(scale), 4),
        "frozen_point": {k: (list(v) if isinstance(v, tuple) else v) for k, v in FROZEN.items()},
        "constrained": {
            "support_recall": round(cnt["pos_hit"] / n_eval, 4),
            "support_recall_counts": f"{cnt['pos_hit']}/{n_eval}",
            "unsupported_detection": round(cnt["neg_flagged"] / n_eval, 4),
            "unsupported_detection_counts": f"{cnt['neg_flagged']}/{n_eval}",
            "false_support_rate": round(cnt["neg_false_support"] / n_eval, 4),
            "false_support_counts": f"{cnt['neg_false_support']}/{n_eval}",
        },
        "free_closure_baseline": {
            "support_recall": round(cnt["fc_pos_hit"] / len(scale), 4),
            "support_recall_counts": f"{cnt['fc_pos_hit']}/{len(scale)}",
            "unsupported_detection": round(cnt["fc_neg_flagged"] / len(scale), 4),
            "unsupported_detection_counts": f"{cnt['fc_neg_flagged']}/{len(scale)}",
        },
        "note": "frozen point chosen on the 60-seed ONLY; this 300 set is disjoint (question-hash).",
    }
    json.dump(summary, open(os.path.join(REPO, "records", "convfinqa_checker_scale_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("->", out_records)
    return summary


if __name__ == "__main__":
    main()
