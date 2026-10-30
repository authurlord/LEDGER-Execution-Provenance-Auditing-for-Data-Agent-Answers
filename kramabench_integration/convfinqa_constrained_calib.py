"""LEDGER v2 STEP 1 — G1 gate: operand-constrained checker calibration on original ConvFinQA (60 seed).

Positives  = gold exe_ans with citations = gold-program operands (supported BY CONSTRUCTION).
Negatives  = cross-example SWAPPED exe_ans with the SAME citations (unsupported by construction;
             unbiased — no checker in the loop; accidental derivability counts AGAINST the checker).
Baseline   = the old FREE-CLOSURE checker over the SAME examples' context numbers (same-data contrast).

Grid axes: ops {4ops, 4ops+prod} × tol {1e-3, 1e-4, 1e-5} × percent {on, off}  → freeze ONE point here.
G1 GATE: unsupported-detection ≥ 0.75 AND support-recall ≥ 0.85 at the frozen point.
Deterministic, CPU-only, no LLM anywhere.

Run: conda run -n kramabench python -m kramabench_integration.convfinqa_constrained_calib
"""
from __future__ import annotations
import os, json
from . import convfinqa_gold as G
from . import provenance as PV
from .convfinqa_support_calib import derivable as free_closure_derivable

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPS4 = ("sum", "diff", "ratio", "pctchg")
OPS5 = OPS4 + ("prod",)


def swap_negative(seed, i):
    """Another example's exe_ans (nonequal), evaluated against example i's citations."""
    n = len(seed)
    for step in (7, 13, 29):
        j = (i + step) % n
        if abs(seed[j]["exe_ans"] - seed[i]["exe_ans"]) > 1e-9:
            return seed[j]["exe_ans"]
    return seed[(i + 1) % n]["exe_ans"]


def eval_point(seed, pools, tol, percent, max_depth=3):
    """pools: prebuilt {i: ConstrainedPool} for the current ops-set (build once, match per point)."""
    pos_hit = neg_flag = 0
    detail = []
    for i, q in enumerate(seed):
        pool = pools[i]
        lab = pool.check(q["exe_ans"], tol_rel=tol, percent=percent, percent_ctx=q["percent_ctx"], max_depth=max_depth)
        neg_val = swap_negative(seed, i)
        nlab = pool.check(neg_val, tol_rel=tol, percent=percent, percent_ctx=q["percent_ctx"], max_depth=max_depth)
        pos_hit += 1 if lab else 0
        neg_flag += 1 if not nlab else 0
        detail.append({"uid": q["uid"], "depth": q["depth"], "percent_ctx": q["percent_ctx"],
                       "operands": q["operands"], "consts": q["consts"],
                       "gold": q["exe_ans"], "gold_derived": lab,
                       "swap": neg_val, "swap_falsely_supported": bool(nlab),
                       "swap_matched_depth": nlab})
    n = len(seed)
    return {"support_recall": round(pos_hit / n, 4), "unsupported_detection": round(neg_flag / n, 4),
            "false_support_rate": round(1 - neg_flag / n, 4), "n": n,
            "pos_hit": pos_hit, "neg_flagged": neg_flag}, detail


def free_closure_row(seed, tol=1e-3):
    """Same-data baseline: OLD free-closure derivability over the example's context numbers."""
    pos = neg = 0
    for i, q in enumerate(seed):
        nums = q["ctx_numbers"]
        pos += 1 if free_closure_derivable(q["exe_ans"], nums, tol, "pairwise") else 0
        neg += 1 if not free_closure_derivable(swap_negative(seed, i), nums, tol, "pairwise") else 0
    n = len(seed)
    return {"support_recall": round(pos / n, 4), "unsupported_detection": round(neg / n, 4),
            "false_support_rate": round(1 - neg / n, 4), "n": n, "pos_hit": pos, "neg_flagged": neg}


def main():
    qas = G.load_qas()
    seed, scale, stats = G.split_seed_scale(qas)
    print("split:", json.dumps(stats))
    grid, best = [], None
    print(f"{'ops':10} {'chain':7} {'dep':4} {'tol':8} {'pct':4} {'supp-recall':>11} {'unsup-det':>10}")
    # round-2 axes (failure-analysis-driven, still calibrated on the SEED only): chain linear/full,
    # max_depth 2/3. Round-1 (full/depth3) FAILED: D2-full ≈135k values too dense in the ratio zone.
    for ops, opsname in ((OPS4, "4ops"), (OPS5, "4ops+prod")):
        for chain in ("linear", "full"):
            pools = {i: PV.ConstrainedPool(q["operands"], q["consts"], ops, chain=chain) for i, q in enumerate(seed)}
            for max_depth in (2, 3):
                for tol in (1e-4, 1e-5):
                    for percent in (True, False):
                        m, detail = eval_point(seed, pools, tol, percent, max_depth)
                        row = {"ops": opsname, "chain": chain, "max_depth": max_depth,
                               "tol": tol, "percent": percent, **m}
                        grid.append(row)
                        print(f"{opsname:10} {chain:7} {max_depth:<4} {tol:<8.0e} {str(percent)[0]:4} "
                              f"{m['support_recall']:>11.3f} {m['unsupported_detection']:>10.3f}")
                        if m["support_recall"] >= 0.85 and m["unsupported_detection"] >= 0.75:
                            key = (m["unsupported_detection"], m["support_recall"], ops is OPS4)
                            if best is None or key > best[0]:
                                best = (key, row, detail)
    baseline = free_closure_row(seed)
    print("free-closure baseline (same 60, ctx numbers):", json.dumps(baseline))
    out = {"split_stats": stats, "grid": grid, "free_closure_baseline_same_data": baseline,
           "frozen_operating_point": (best[1] if best else None),
           "G1_gate": {"criteria": "unsup-det>=0.75 AND supp-recall>=0.85",
                        "PASS": best is not None}}
    os.makedirs(os.path.join(REPO, "records"), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, "records", "convfinqa_constrained_grid.json"), "w"), indent=1)
    if best:
        json.dump(best[2], open(os.path.join(REPO, "records", "convfinqa_constrained_seed_detail.json"), "w"), indent=1)
        print("\nFROZEN OPERATING POINT:", json.dumps({k: best[1][k] for k in ("ops", "tol", "percent",
                                                       "support_recall", "unsupported_detection")}))
        print("G1 GATE: PASS")
    else:
        # auditable failure analysis: which negatives survive at the best-recall point
        print("G1 GATE: FAIL — no grid point clears (>=0.85 recall AND >=0.75 unsup-det)")
    print("-> records/convfinqa_constrained_grid.json")
    return out


if __name__ == "__main__":
    main()
