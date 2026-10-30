"""LEDGER v2 STEP 3e + STEP 4 — join official scores into provenance records; emit C3 decomposition,
selective-answering, and cost tables.

Official score = the newest <workload>_measures_*.csv per SUT written AFTER the STEP-3 runs
(main-task rows, metric='success'). Joined into records (score field). C3 classification per domain:
  selection-bound  : oracle - discovery >= 0.10
  reasoning-bound  : oracle <= 0.40 AND gap < 0.10   (even gold files stay low)
  mixed            : otherwise
(dump-all 0.036 stays the discovery-bound floor reference from D2.)

Run: conda run -n kramabench python -m kramabench_integration.kb_c3_aggregate
"""
from __future__ import annotations
import os, re, json, glob
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(REPO, "KramaBench")
SUTS = {"oracle": "BaselineLLMSystemLocalQwenFewShot", "discovery": "BaselineLLMSystemLocalQwenDiscovery"}
DOMAINS = ["environment", "wildfire", "legal", "archeology"]


def latest_measures(sut, workload, min_ts="20260703"):
    fs = []
    for f in glob.glob(os.path.join(KB, "results", sut, f"{workload}_measures_*.csv")):
        m = re.search(r"measures_(\d{8}_\d{6})\.csv$", f)
        if m and m.group(1) >= min_ts:
            fs.append((m.group(1), f))
    return sorted(fs)[-1][1] if fs else None


SCORE_METRICS = ["success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"]   # official set


def task_scores(sut, workload):
    """Per-main-task primary-metric value (official aggregate_results semantics: each task contributes
    its score metric; NaN = unscored, e.g. paraphrase-eval unavailable with our dummy OpenAI key)."""
    fp = latest_measures(sut, workload)
    if not fp:
        return {}, None
    df = pd.read_csv(fp)
    main = df[df["metric"].isin(SCORE_METRICS) & (df["task_id"].str.count("-") < 3)].copy()
    main["v"] = pd.to_numeric(main["value"], errors="coerce")
    return dict(zip(main["task_id"], main["v"])), fp


def main():
    rows_all = []
    c3 = {}
    for setting, sut in SUTS.items():
        for dom in DOMAINS:
            scores, mfp = task_scores(sut, dom)
            rp = os.path.join(REPO, "records", f"records_ledger_n2_kb_{dom}_{setting}.jsonl")
            if not os.path.exists(rp):
                continue
            recs = [json.loads(l) for l in open(rp)]
            for r in recs:
                r["score"] = float(scores.get(r["taskid"], float("nan"))) if scores else -1
                r["measures_file"] = os.path.basename(mfp) if mfp else ""
                rows_all.append(r)
            with open(rp, "w") as f:                     # rewrite with joined scores
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            sc = [r["score"] for r in recs if r["score"] == r["score"] and r["score"] >= 0]
            c3.setdefault(dom, {})[setting] = {
                "n": len(recs),
                "acc": round(sum(sc) / len(sc), 4) if sc else None,            # mean over SCORED tasks
                "acc_total": round(sum(sc) / len(recs), 4) if recs else None,  # unscored counted 0 (official-denominator view)
                "scored": len(sc),
                "supported": sum(1 for r in recs if r["grounding_label"] == "supported"),
                "unsupported": sum(1 for r in recs if r["grounding_label"] == "unsupported"),
                "unknown": sum(1 for r in recs if r["grounding_label"] == "unknown"),
                "classes": {c: sum(1 for r in recs if r.get("unsupported_class") == c) for c in "abcd"},
                "tok_in": round(sum(r["tokens_in"] for r in recs) / max(1, len(recs))),
                "tok_out": round(sum(r["tokens_out"] for r in recs) / max(1, len(recs))),
                "turns": round(sum(r["turns"] for r in recs) / max(1, len(recs)), 2),
                "lat": round(sum(r.get("latency_s", 0) for r in recs) / max(1, len(recs)), 1),
            }

    # C3 classification
    for dom, d in c3.items():
        o, di = d.get("oracle", {}).get("acc"), d.get("discovery", {}).get("acc")
        if o is None or di is None:
            d["class"] = "pending"
        elif o - di >= 0.10:
            d["class"] = "selection-bound"
        elif o <= 0.40:
            d["class"] = "reasoning-bound"
        else:
            d["class"] = "mixed"

    # STEP 4: selective answering on discovery records
    disc = [r for r in rows_all if r["system"].endswith("|discovery") and r["score"] >= 0]
    def sel(policy):
        if policy == "always":
            keep = disc
        elif policy == "supported":
            keep = [r for r in disc if r["grounding_label"] == "supported"]
        else:
            keep = [r for r in disc if r["grounding_label"] in ("supported", "unknown")]
        n = len(keep)
        return {"coverage": f"{n}/{len(disc)}" if disc else "0/0",
                "precision_on_answered": round(sum(r["score"] for r in keep) / n, 4) if n else None}
    selective = {p: sel(p) for p in ("always", "supported", "supported+unknown")}

    out = {"c3_decomposition": c3, "dump_all_reference": 0.036,
           "d2_anchors": {"discovery": 0.276, "oracle": 0.314},
           "selective_answering_discovery": selective}
    json.dump(out, open(os.path.join(REPO, "records", "kb_c3_selective_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
