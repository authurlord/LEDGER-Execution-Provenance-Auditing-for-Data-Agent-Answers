"""Metrics for support determination, clustered by parent.

Given records (with ``gold_support``, ``gold_class``, ``parent_id``) and a per-record predicted label
in {supported, unsupported, unknown}, compute confusion, precision/recall/F1 for both classes,
false-certification / false-rejection rates, unknown/coverage, per-class recall, and 95% bootstrap
confidence intervals resampled at the PARENT level (generated variants of one parent are not
independent).

Run self-test: python3 -m experiments.sqlledger.evaluation.metrics
"""
from __future__ import annotations
import math
from collections import Counter, defaultdict

SUPP, UNSUPP, UNK = "supported", "unsupported", "unknown"


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def confusion(pairs):
    """pairs: list of (gold_support, pred_label). Returns counts dict."""
    c = Counter()
    for g, p in pairs:
        c[(g, p)] += 1
    return c


def point_metrics(records, preds):
    """records aligned with preds (list of label strings). Returns a metrics dict."""
    pairs = [(r["gold_support"], p) for r, p in zip(records, preds)
             if r["gold_support"] in (SUPP, UNSUPP)]
    n = len(pairs)
    # unsupported-detection (positive class = unsupported)
    tp = sum(1 for g, p in pairs if g == UNSUPP and p == UNSUPP)
    fp = sum(1 for g, p in pairs if g == SUPP and p == UNSUPP)
    fn = sum(1 for g, p in pairs if g == UNSUPP and p != UNSUPP)
    tn = sum(1 for g, p in pairs if g == SUPP and p == SUPP)
    up, ur, uf = _prf(tp, fp, fn)
    # supported-detection (positive class = supported)
    stp = sum(1 for g, p in pairs if g == SUPP and p == SUPP)
    sfp = sum(1 for g, p in pairs if g == UNSUPP and p == SUPP)
    sfn = sum(1 for g, p in pairs if g == SUPP and p != SUPP)
    sp, sr, sf = _prf(stp, sfp, sfn)
    n_unsupp = sum(1 for g, _ in pairs if g == UNSUPP)
    n_supp = sum(1 for g, _ in pairs if g == SUPP)
    unknown = sum(1 for g, p in pairs if p == UNK)
    return {
        "n": n,
        "unsupported_detection": {"precision": up, "recall": ur, "f1": uf,
                                   "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "supported_detection": {"precision": sp, "recall": sr, "f1": sf},
        "false_certification_rate": (sum(1 for g, p in pairs if g == UNSUPP and p == SUPP) /
                                     n_unsupp if n_unsupp else 0.0),
        "false_rejection_rate": (sum(1 for g, p in pairs if g == SUPP and p == UNSUPP) /
                                 n_supp if n_supp else 0.0),
        "unknown_rate": unknown / n if n else 0.0,
    }


def per_class_recall(records, preds):
    """For supported classes (s1,s2) recall = correctly-supported; for u_* = correctly-unsupported."""
    by = defaultdict(lambda: [0, 0])
    for r, p in zip(records, preds):
        cls = r["gold_class"]; gold = r["gold_support"]
        want = SUPP if gold == SUPP else UNSUPP
        by[cls][1] += 1
        if p == want:
            by[cls][0] += 1
    return {cls: {"recall": c[0] / c[1] if c[1] else 0.0, "n": c[1]} for cls, c in sorted(by.items())}


def _percentile(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs); k = (len(xs) - 1) * q
    lo = math.floor(k); hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def cluster_key(rec):
    """Clustering unit for bootstrap/leakage: the database (Spider/BIRD) or table (WikiSQL) — both
    are ``db_id`` in our schema — namespaced by dataset so ids never collide across corpora."""
    return f"{rec['dataset']}:{rec['db_id']}"


def parent_bootstrap(records, preds, metric_fn, n_boot=1000, seed=20260711, key=None):
    """Bootstrap a scalar metric by resampling CLUSTERS (default = parent_id; pass key=cluster_key to
    resample by database/table). Deterministic LCG (fixed seed; Math.random-equivalents avoided)."""
    key = key or (lambda r: r["parent_id"])
    by_parent = defaultdict(list)
    for r, p in zip(records, preds):
        by_parent[key(r)].append((r, p))
    parents = sorted(by_parent)
    m = len(parents)
    if m == 0:
        return (float("nan"), float("nan"))
    state = seed & 0xFFFFFFFF
    def rnd():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF
    vals = []
    for _ in range(n_boot):
        recs, prs = [], []
        for _ in range(m):
            pid = parents[int(rnd() * m) % m]
            for r, p in by_parent[pid]:
                recs.append(r); prs.append(p)
        vals.append(metric_fn(recs, prs))
    return (_percentile(vals, 0.025), _percentile(vals, 0.975))


def evaluate(records, preds, n_boot=500, cluster_by_db=False):
    pm = point_metrics(records, preds)
    pcr = per_class_recall(records, preds)
    key = cluster_key if cluster_by_db else None
    ci_ur = parent_bootstrap(records, preds,
                             lambda rs, ps: point_metrics(rs, ps)["unsupported_detection"]["recall"], n_boot, key=key)
    ci_up = parent_bootstrap(records, preds,
                             lambda rs, ps: point_metrics(rs, ps)["unsupported_detection"]["precision"], n_boot, key=key)
    pm["unsupported_detection"]["recall_ci95"] = ci_ur
    pm["unsupported_detection"]["precision_ci95"] = ci_up
    pm["per_class_recall"] = pcr
    return pm


def _selftest():
    recs = [{"gold_support": "supported", "gold_class": "s1", "parent_id": "p1"},
            {"gold_support": "unsupported", "gold_class": "u_a1", "parent_id": "p1"},
            {"gold_support": "unsupported", "gold_class": "u_c1", "parent_id": "p2"},
            {"gold_support": "supported", "gold_class": "s2", "parent_id": "p2"}]
    preds = ["supported", "unsupported", "unsupported", "supported"]  # perfect
    m = evaluate(recs, preds, n_boot=50)
    assert m["unsupported_detection"]["recall"] == 1.0 and m["supported_detection"]["recall"] == 1.0
    assert m["false_certification_rate"] == 0.0 and m["false_rejection_rate"] == 0.0
    preds2 = ["supported", "supported", "unsupported", "supported"]  # miss one unsupported
    m2 = evaluate(recs, preds2, n_boot=50)
    assert abs(m2["unsupported_detection"]["recall"] - 0.5) < 1e-9, m2["unsupported_detection"]
    assert m2["per_class_recall"]["u_a1"]["recall"] == 0.0
    print("metrics selftest OK")


if __name__ == "__main__":
    _selftest()
