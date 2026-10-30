"""LEDGER STEP 4 (N2) — ConvFinQA support-checker CALIBRATION (deterministic, no LLM, no GPU).

Question: can a training-free support checker tell "this numeric answer is derivable from the cited
context" from "it is not"? ConvFinQA gives us labels FOR FREE:
  - the GOLD answer of each example is derivable from its inline context BY CONSTRUCTION -> known-SUPPORTED
  - a perturbed value (gold*1.37+jitter, re-checked to not be accidentally derivable) -> known-UNSUPPORTED
So checker sensitivity (support-recall on gold) and false-support rate (on perturbed) measure the
discrimination of the check itself — the P2-seed number, with zero human annotation.

Checker = value derivable from the context's numbers via:
  cell match | single-column aggregates | pairwise ops {a/b, a-b, a+b, (a-b)/b} | x100 / /100 pct variants.
This is the "gold-program-free" approximation (flare-convfinqa lacks the DSL; noted in LEDGER_DATA_INVENTORY).

Run: conda run -n kramabench python -m kramabench_integration.convfinqa_support_calib
"""
from __future__ import annotations
import os, re, json, random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = os.path.join(REPO, "datasets", "normalized", "convfinqa_provenance_seed.jsonl")
_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def context_numbers(query: str) -> list[float]:
    """All numbers in the inline context (table + pre/post text), de-noised."""
    ctx = query.split("Context:", 1)[-1]
    ctx = ctx.split("Conversations:", 1)[0]
    vals = []
    for m in _NUM.findall(ctx):
        try:
            v = float(m.replace(",", ""))
        except ValueError:
            continue
        if abs(v) < 1e12:                      # drop degenerate huge ids
            vals.append(v)
    # dedupe, keep bounded for pairwise ops
    out = sorted(set(vals))
    return out[:120]


def derivable(value: float, nums: list[float], tol_rel=1e-3, ops="pairwise") -> str | None:
    """Is `value` derivable from nums? Returns the op label or None.
    ops: 'cell' (cells + pct variants only) | 'pairwise' (+ {sum,diff,ratio,pctchg} over pairs)."""
    def close(a, b):
        return abs(a - b) <= max(tol_rel * abs(b), 1e-9)
    variants = [(value, "")]
    if value != 0:
        variants += [(value / 100.0, "/100"), (value * 100.0, "x100")]
    for v, tag in variants:
        for x in nums:
            if close(x, v):
                return "cell" + tag
        if ops == "cell":
            continue
        for i, a in enumerate(nums):
            for b in nums[i + 1:]:
                for r, op in ((a + b, "sum"), (a - b, "diff"), (b - a, "diff"),
                              (a / b if b else None, "ratio"), (b / a if a else None, "ratio"),
                              ((a - b) / b if b else None, "pctchg"), ((b - a) / a if a else None, "pctchg")):
                    if r is not None and close(r, v):
                        return f"{op}{tag}"
    return None


def main(n=60, seed=13):
    """codex P0 fix: the re-jitter-until-not-derivable negatives are CHECKER-FILTERED (upward-biased,
    circular) — keep them only as a sanity variant. PRIMARY negatives = gold answers SWAPPED across
    examples (a real number, plausibly scaled, from a DIFFERENT context — no checker in the loop)."""
    rng = random.Random(seed)
    rows = [json.loads(l) for l in open(SEED)][:n]
    valid = []
    for r in rows:
        try:
            gold = float(str(r["answer"]).replace("%", "").replace(",", ""))
        except ValueError:
            continue
        nums = context_numbers(r["query"])
        if nums:
            valid.append((r, gold, nums))
    n_valid = len(valid)
    golds = [g for _, g, _ in valid]
    supported_hit = swap_flagged = jitter_flagged = 0
    per = []
    for i, (r, gold, nums) in enumerate(valid):
        op = derivable(gold, nums)
        supported_hit += 1 if op else 0
        # PRIMARY negative: another example's gold answer (unbiased; may rarely be coincidentally
        # derivable -> counts AGAINST the checker, as it should)
        j = (i + 7) % n_valid
        swap = golds[j] if abs(golds[j] - gold) > 1e-9 else golds[(i + 13) % n_valid]
        swap_op = derivable(swap, nums)
        swap_flagged += 1 if not swap_op else 0
        # SANITY variant (biased, reported separately): jittered value re-drawn to avoid derivability
        pert = gold * 1.37 + rng.uniform(0.011, 0.09) * (abs(gold) + 1)
        tries = 0
        while derivable(pert, nums) and tries < 5:
            pert = pert * 1.13 + rng.uniform(0.01, 0.05) * (abs(gold) + 1); tries += 1
        jitter_flagged += 1 if not derivable(pert, nums) else 0
        per.append({"taskid": r["taskid"], "gold": gold, "gold_derivable_via": op,
                    "swap_negative": swap, "swap_falsely_supported": bool(swap_op)})
    rec = {
        "n": n_valid,
        "support_recall_on_gold": round(supported_hit / n_valid, 4),
        "unsupported_detection_SWAP (primary, unbiased)": round(swap_flagged / n_valid, 4),
        "false_support_rate_SWAP": round(1 - swap_flagged / n_valid, 4),
        "unsupported_detection_JITTER (sanity, checker-filtered -> biased up)": round(jitter_flagged / n_valid, 4),
        "note": ("labels BY-CONSTRUCTION (gold=supported; cross-example swapped gold=unsupported). "
                 "Checker = cell/agg/pairwise-op derivability, gold-program-free approximation."),
    }
    outp = os.path.join(REPO, "logs", "convfinqa_support_calibration.json")
    json.dump({"summary": rec, "per_example": per}, open(outp, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    print("->", outp)
    return rec


def grid(n=60):
    """Operating-point grid: tolerance × op-set -> (support-recall on gold, unsupported-detection on SWAP).
    Quantifies the precision/recall tradeoff of the training-free support checker (P2-seed deliverable)."""
    rows = [json.loads(l) for l in open(SEED)][:n]
    valid = []
    for r in rows:
        try:
            gold = float(str(r["answer"]).replace("%", "").replace(",", ""))
        except ValueError:
            continue
        nums = context_numbers(r["query"])
        if nums:
            valid.append((gold, nums))
    golds = [g for g, _ in valid]
    nv = len(valid)
    out = []
    print(f"{'ops':9} {'tol':7} {'supp-recall':>11} {'unsup-det(SWAP)':>16}")
    for ops in ("cell", "pairwise"):
        for tol in (1e-3, 1e-4, 1e-5):
            sr = sum(1 for g, nums in valid if derivable(g, nums, tol, ops)) / nv
            ud = sum(1 for i, (g, nums) in enumerate(valid)
                     if not derivable(golds[(i + 7) % nv] if abs(golds[(i+7) % nv]-g) > 1e-9
                                      else golds[(i + 13) % nv], nums, tol, ops)) / nv
            out.append({"ops": ops, "tol": tol, "support_recall": round(sr, 4), "unsupported_detection": round(ud, 4)})
            print(f"{ops:9} {tol:<7.0e} {sr:>11.3f} {ud:>16.3f}")
    outp = os.path.join(REPO, "logs", "convfinqa_support_grid.json")
    json.dump(out, open(outp, "w"), indent=1)
    print("->", outp)
    return out


if __name__ == "__main__":
    import sys
    if "--grid" in sys.argv:
        grid()
    else:
        main()
