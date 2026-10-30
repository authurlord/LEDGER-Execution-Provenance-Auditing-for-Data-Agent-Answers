"""LEDGER v2 STEP 3g — evaluate checker vs HUMAN support labels (G2 verdict block).

Consumes docs/annotation/KB_SUPPORT_ANNOTATION.csv AFTER Authur fills `human_support_label`
(yes/no/unclear) — the script refuses to run on empty human columns (labels are Authur's; never
auto-filled). Reports, with raw counts and exact (Clopper-Pearson) 95% CIs:
  - unsupported-detection precision/recall vs human labels (checker 'unsupported*' vs human 'no')
  - checker-human agreement (excluding 'unclear')
  - false-positive rate on human-supported answers, contrasted with the old final-answer gate (34/47)
  - unknown_rate + unclear_rate separately.
Flag<->correctness correlation is SECONDARY/EXPLORATORY (printed last, labeled as such).

Run (after labels): conda run -n kramabench python -m kramabench_integration.eval_support_labels
"""
from __future__ import annotations
import os, csv, json, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHEET = os.path.join(REPO, "docs", "annotation", "KB_SUPPORT_ANNOTATION.csv")


def _cp_ci(k, n, alpha=0.05):
    """Clopper-Pearson exact CI via beta quantiles (scipy-free approximation is not exact; use scipy
    if available, else Wilson as fallback labeled approx)."""
    try:
        from scipy.stats import beta
        lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
        hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
        return round(lo, 3), round(hi, 3), "exact"
    except Exception:
        if n == 0:
            return 0.0, 1.0, "n=0"
        p = k / n; z = 1.96
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return round(max(0, c - h), 3), round(min(1, c + h), 3), "wilson-approx"


def main():
    if not os.path.exists(SHEET):
        raise SystemExit(f"annotation sheet not found: {SHEET}")
    rows = list(csv.DictReader(open(SHEET)))
    labeled = [r for r in rows if (r.get("human_support_label") or "").strip().lower() in ("yes", "no", "unclear")]
    if not labeled:
        raise SystemExit("HUMAN LABELS EMPTY — G2 pending. This script never auto-fills labels (protocol).")
    unlabeled = len(rows) - len(labeled)
    unclear = [r for r in labeled if r["human_support_label"].strip().lower() == "unclear"]
    use = [r for r in labeled if r["human_support_label"].strip().lower() in ("yes", "no")]

    def checker_unsup(r):
        return (r.get("checker_label") or "").strip().lower().startswith("unsup")

    def checker_sup(r):
        return (r.get("checker_label") or "").strip().lower() == "supported"

    tp = sum(1 for r in use if checker_unsup(r) and r["human_support_label"].lower() == "no")
    fp = sum(1 for r in use if checker_unsup(r) and r["human_support_label"].lower() == "yes")
    fn = sum(1 for r in use if not checker_unsup(r) and r["human_support_label"].lower() == "no")
    tn = sum(1 for r in use if not checker_unsup(r) and r["human_support_label"].lower() == "yes")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    agree = sum(1 for r in use if checker_unsup(r) == (r["human_support_label"].lower() == "no")
                and (checker_unsup(r) or checker_sup(r)))
    n_agree_base = sum(1 for r in use if checker_unsup(r) or checker_sup(r))
    fp_on_supported = fp / max(1, sum(1 for r in use if r["human_support_label"].lower() == "yes"))

    p_lo, p_hi, p_kind = _cp_ci(tp, tp + fp) if tp + fp else (0, 1, "n=0")
    r_lo, r_hi, r_kind = _cp_ci(tp, tp + fn) if tp + fn else (0, 1, "n=0")

    out = {
        "n_rows": len(rows), "n_labeled": len(labeled), "n_unlabeled": unlabeled,
        "unclear_rate": f"{len(unclear)}/{len(labeled)}",
        "unsupported_detection_vs_human": {
            "precision": f"{prec:.3f} ({tp}/{tp+fp}) CI95=[{p_lo},{p_hi}] ({p_kind})",
            "recall": f"{rec:.3f} ({tp}/{tp+fn}) CI95=[{r_lo},{r_hi}] ({r_kind})",
            "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        },
        "checker_human_agreement": f"{agree}/{n_agree_base}",
        "fp_rate_on_human_supported": f"{fp_on_supported:.3f} (old final-answer gate: 34/47 = 0.723)",
        "G2_gate_guide": "precision>=0.6 AND recall>=0.5 AND fp-rate << 0.723",
        "G2_pass": bool(tp + fp and tp + fn and prec >= 0.6 and rec >= 0.5 and fp_on_supported < 0.36),
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    # SECONDARY/EXPLORATORY ONLY (protocol): flag vs correctness
    corr_rows = [r for r in use if (r.get("official_score") or "").strip() not in ("", "-1")]
    if corr_rows:
        both = sum(1 for r in corr_rows if checker_sup(r) == (float(r["official_score"]) > 0.5))
        print(f"[secondary/exploratory] flag<->correctness agreement: {both}/{len(corr_rows)}")
    return out


if __name__ == "__main__":
    main()
