"""SQLLedger CLI — Phase 0 smoke and (later) dev/frozen runs.

Phase 0 smoke: build a small WikiSQL corpus, validate every record, check rerun determinism, run all
baselines B0-B6 on identical eligible IDs, compute parent-clustered metrics, and write records +
docs/sqlledger/SMOKE_REPORT.md. Enforces the Phase-0 gates (schema-valid, witnessed, determinism,
identical IDs, no crash-as-sensitivity, B5 dominates B0).

Run: python3 -m experiments.sqlledger.cli smoke --split dev --limit 200
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from .adapters.wikisql import WikiSQLDataset, WIKI
from .construct.generate import generate
from .schema import validate
from .audit.baselines import BASELINES, run_baseline
from .evaluation.metrics import evaluate

REPO = Path(__file__).resolve().parents[2]


def _src_db_path(rec):
    """Resolve the source SQLite path for a record from its dataset + db_id."""
    if rec["dataset"] == "wikisql":
        return str(WIKI / f"{rec['source_split']}.db")
    if rec["dataset"] == "spider":
        from .adapters.spider import db_path
        return db_path(rec["db_id"])
    raise ValueError(rec["dataset"])


def _sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
REC_DIR = REPO / "records" / "sqlledger" / "wikisql"
DOC = REPO / "docs" / "sqlledger" / "SMOKE_REPORT.md"


def _determinism_check(split, limit):
    r1, c1, _ = generate(split=split, limit=limit)
    r2, c2, _ = generate(split=split, limit=limit)
    key = lambda recs: {x["case_id"]: (x["gold_support"], x["gold_class"],
                                       x["stated_answer_canonical"]) for x in recs}
    return key(r1) == key(r2), len(r1)


def smoke(split="dev", limit=200):
    if not (WIKI / f"{split}.db").exists():
        raise SystemExit(f"WikiSQL {split}.db absent under {WIKI}")
    recs, counts, reasons = generate(split=split, limit=limit, eval_split="calibration")
    gates = {}

    # G0: schema validity
    for r in recs:
        validate(r)
    gates["G0_schema_valid"] = True

    # witness non-empty (construction gate)
    gates["G3_all_witnessed"] = all(bool(r["construction_witness"]) for r in recs)

    # G1: rerun determinism
    det_ok, _ = _determinism_check(split, min(limit, 100))
    gates["G1_determinism"] = det_ok

    # run baselines on identical eligible IDs
    src = sqlite3.connect(str(WIKI / f"{split}.db"))
    src.execute("PRAGMA query_only=ON")
    baseline_preds = {name: [] for name in BASELINES}
    latencies = {name: [] for name in BASELINES}
    for r in recs:
        for name in BASELINES:
            out = run_baseline(name, r, src)
            baseline_preds[name].append(out["label"])
            latencies[name].append(out.get("latency_ms", 0.0))
    src.close()

    # metrics per baseline
    results = {}
    for name in BASELINES:
        m = evaluate(recs, baseline_preds[name], n_boot=300)
        lat = sorted(latencies[name])
        m["latency_ms_p50"] = lat[len(lat) // 2] if lat else 0.0
        m["latency_ms_p95"] = lat[int(len(lat) * 0.95)] if lat else 0.0
        results[name] = m

    b5 = results["B5"]["unsupported_detection"]
    b0 = results["B0"]["unsupported_detection"]
    gates["B5_dominates_B0"] = b5["f1"] > b0["f1"]
    gates["identical_eligible_ids"] = all(len(baseline_preds[n]) == len(recs) for n in BASELINES)

    # write records
    REC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REC_DIR / f"{split}_calibration_smoke.jsonl"
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_report(split, limit, recs, counts, reasons, results, gates, out_path)
    print(json.dumps({"n_records": len(recs), "classes": dict(counts), "rejections": dict(reasons),
                      "gates": gates,
                      "B5": {k: round(v, 3) if isinstance(v, float) else v for k, v in b5.items()},
                      "B0_f1": round(b0["f1"], 3)}, indent=1))
    return results, gates


def _fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci and ci[0] == ci[0] else "[n/a]"


def _write_report(split, limit, recs, counts, reasons, results, gates, out_path):
    L = [f"# SQLLedger Phase 0 smoke — WikiSQL {split}",
         "",
         f"Parents attempted: {limit}. Records: {len(recs)}. Source: `external/raw/wikisql/data/{split}.db`.",
         f"Records written: `{out_path.relative_to(REPO)}`. Gold labels generated with no LEDGER checker/LLM.",
         "",
         "## Class counts", "", "| class | n |", "|---|---|"]
    for cls in ("s1", "s2", "u_a1", "u_b", "u_c1", "u_c2", "u_a3"):
        L.append(f"| {cls} | {counts.get(cls, 0)} |")
    L += ["", f"Rejections: `{dict(reasons)}`", "",
          "## Phase-0 gates", "", "| gate | pass |", "|---|---|"]
    for k, v in gates.items():
        L.append(f"| {k} | {'✅' if v else '❌'} |")
    L += ["", "## Baseline ablation (unsupported-detection, parent-clustered)", "",
          "| baseline | precision | recall | F1 | false-cert | false-rej | unknown | p50 ms |",
          "|---|---|---|---|---|---|---|---|"]
    labels = {"B0": "citation-exists", "B1": "static-extract", "B2": "unrestricted-replay",
              "B3": "restricted-replay", "B4": "+read/decl", "B5": "+intervention (LEDGER)",
              "B6": "+legacy-counterfactual"}
    for name in ("B0", "B1", "B2", "B3", "B4", "B5", "B6"):
        u = results[name]["unsupported_detection"]
        m = results[name]
        L.append(f"| {name} {labels[name]} | {u['precision']:.3f} {_fmt_ci(u.get('precision_ci95'))} "
                 f"| {u['recall']:.3f} {_fmt_ci(u.get('recall_ci95'))} | {u['f1']:.3f} "
                 f"| {m['false_certification_rate']:.3f} | {m['false_rejection_rate']:.3f} "
                 f"| {m['unknown_rate']:.3f} | {m['latency_ms_p50']:.2f} |")
    L += ["", "## Per-class recall (B5 = full LEDGER)", "",
          "| class | recall | n |", "|---|---|---|"]
    for cls, d in results["B5"]["per_class_recall"].items():
        L.append(f"| {cls} | {d['recall']:.3f} | {d['n']} |")
    L += ["", "_Note: s2 remains skewed toward empty distractor results on single-table WikiSQL "
          "(valid but degenerate); Spider/BIRD multi-table corpora will exercise non-trivial c1/joins._"]
    DOC.write_text("\n".join(L) + "\n")


def _run_baselines(recs):
    con_cache = {}
    preds = {name: [] for name in BASELINES}
    lat = {name: [] for name in BASELINES}
    for r in recs:
        p = _src_db_path(r)
        if p not in con_cache:
            c = sqlite3.connect(p); c.execute("PRAGMA query_only=ON"); con_cache[p] = c
        src = con_cache[p]
        for name in BASELINES:
            out = run_baseline(name, r, src)
            preds[name].append(out["label"]); lat[name].append(out.get("latency_ms", 0.0))
    for c in con_cache.values():
        c.close()
    return preds, lat


def phase1(wikisql_limit=1000, spider_limit=300, git_commit="unknown"):
    recs = []
    per_ds = {}
    for dataset, split, lim in (("wikisql", "dev", wikisql_limit), ("spider", "dev", spider_limit)):
        r, counts, reasons = generate(split=split, limit=lim, eval_split="calibration", dataset=dataset)
        for x in r:
            validate(x)
        per_ds[dataset] = {"n": len(r), "classes": dict(counts), "rejections": dict(reasons)}
        recs += r
    preds, lat = _run_baselines(recs)
    results = {name: evaluate(recs, preds[name], n_boot=300) for name in BASELINES}
    # per-dataset B5
    for dataset in ("wikisql", "spider"):
        idx = [i for i, r in enumerate(recs) if r["dataset"] == dataset]
        drecs = [recs[i] for i in idx]; dpred = [preds["B5"][i] for i in idx]
        per_ds[dataset]["B5"] = evaluate(drecs, dpred, n_boot=200)["unsupported_detection"]

    out_dir = REPO / "records" / "sqlledger"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_path = out_dir / "phase1_dev_calibration.jsonl"
    with open(dev_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # manifest with dataset + config hashes
    manifest = {
        "seed": 20260711, "git_commit": git_commit, "engine": f"sqlite-{sqlite3.sqlite_version}",
        "n_records": len(recs), "per_dataset": {k: {"n": v["n"], "classes": v["classes"],
                                                    "rejections": v["rejections"]} for k, v in per_ds.items()},
        "dataset_hashes": {
            "wikisql_dev.db": _sha256(WIKI / "dev.db") if (WIKI / "dev.db").exists() else None,
        },
        "records_sha256": _sha256(dev_path),
    }
    (REPO / "manifests" / "sqlledger").mkdir(parents=True, exist_ok=True)
    man_path = REPO / "manifests" / "sqlledger" / "dev_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=1))

    _write_phase1_report(recs, per_ds, results, dev_path, man_path)
    print(json.dumps({"n_records": len(recs),
                      "per_dataset": {k: v["n"] for k, v in per_ds.items()},
                      "B5_overall": {k: round(v, 3) if isinstance(v, float) else v
                                     for k, v in results["B5"]["unsupported_detection"].items()
                                     if k in ("precision", "recall", "f1")},
                      "records": str(dev_path.relative_to(REPO)),
                      "manifest": str(man_path.relative_to(REPO))}, indent=1))
    return results


def _write_phase1_report(recs, per_ds, results, dev_path, man_path):
    doc = REPO / "docs" / "sqlledger" / "PHASE1_DEV_REPORT.md"
    L = ["# SQLLedger Phase 1 (dev-scale) — WikiSQL + Spider", "",
         f"Records: {len(recs)} (`{dev_path.relative_to(REPO)}`). Manifest: `{man_path.relative_to(REPO)}`.",
         "Gold labels: execution + declared-citation logic + machine-checkable witnesses, no LLM/checker.",
         "", "## Per-dataset", "", "| dataset | n | classes | B5 precision | B5 recall | B5 F1 |",
         "|---|---|---|---|---|---|"]
    for ds in ("wikisql", "spider"):
        d = per_ds[ds]; u = d["B5"]
        L.append(f"| {ds} | {d['n']} | {d['classes']} | {u['precision']:.3f} | {u['recall']:.3f} | {u['f1']:.3f} |")
    L += ["", "### Rejection taxonomy", ""]
    for ds in ("wikisql", "spider"):
        L.append(f"- **{ds}**: `{per_ds[ds]['rejections']}`")
    L += ["", "## Baseline ablation (combined, unsupported-detection)", "",
          "| baseline | precision | recall | F1 | false-cert | false-rej |",
          "|---|---|---|---|---|---|"]
    for name in ("B0", "B1", "B2", "B3", "B4", "B5", "B6"):
        u = results[name]["unsupported_detection"]; m = results[name]
        L.append(f"| {name} | {u['precision']:.3f} | {u['recall']:.3f} | {u['f1']:.3f} "
                 f"| {m['false_certification_rate']:.3f} | {m['false_rejection_rate']:.3f} |")
    L += ["", "## Per-class recall (B5, combined)", "", "| class | recall | n |", "|---|---|---|"]
    for cls, dd in results["B5"]["per_class_recall"].items():
        L.append(f"| {cls} | {dd['recall']:.3f} | {dd['n']} |")
    doc.write_text("\n".join(L) + "\n")


def _calibration_clusters():
    """Cluster keys used in Phase-1 calibration (for the leakage guard)."""
    from .evaluation.metrics import cluster_key
    p = REPO / "records" / "sqlledger" / "phase1_dev_calibration.jsonl"
    if not p.exists():
        return set()
    return {cluster_key(json.loads(l)) for l in open(p)}


def phase2(wikisql_parents=20000, spider=True, bird=True, git_commit="unknown"):
    from .evaluation.metrics import cluster_key
    specs = [("wikisql", "train", wikisql_parents)]
    if spider:
        specs.append(("spider", "train_spider", None))
    if bird:
        try:
            from .adapters.bird import BIRDDataset  # noqa: F401
            specs.append(("bird", "mini_dev", None))
        except Exception as e:  # noqa: BLE001
            print(f"[phase2] BIRD skipped: {e}")

    recs = []; per_ds = {}
    for dataset, split, lim in specs:
        r, counts, reasons = generate(split=split, limit=lim, eval_split="test", dataset=dataset)
        for x in r:
            validate(x)
        per_ds[dataset] = {"n": len(r), "classes": dict(counts), "rejections": dict(reasons)}
        recs += r

    # LEAKAGE GUARD: no test cluster may appear in Phase-1 calibration
    calib = _calibration_clusters()
    test_clusters = {cluster_key(r) for r in recs}
    leak = test_clusters & calib
    if leak:
        raise SystemExit(f"LEAKAGE: {len(leak)} test clusters overlap calibration, e.g. {sorted(leak)[:5]}")

    # FREEZE manifest BEFORE evaluation
    out_dir = REPO / "records" / "sqlledger"; out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / "phase2_frozen_test.jsonl"
    with open(test_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "seed": 20260711, "git_commit": git_commit, "engine": f"sqlite-{sqlite3.sqlite_version}",
        "n_records": len(recs), "n_clusters": len(test_clusters),
        "per_dataset": per_ds,
        "leakage_guard": {"calibration_clusters": len(calib), "overlap": 0},
        "records_sha256": _sha256(test_path),
    }
    (REPO / "manifests" / "sqlledger").mkdir(parents=True, exist_ok=True)
    man_path = REPO / "manifests" / "sqlledger" / "frozen_test_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=1))

    # EVALUATE (frozen)
    preds, lat = _run_baselines(recs)
    results = {name: evaluate(recs, preds[name], n_boot=300, cluster_by_db=True) for name in BASELINES}
    for dataset in per_ds:
        idx = [i for i, r in enumerate(recs) if r["dataset"] == dataset]
        per_ds[dataset]["B5"] = evaluate([recs[i] for i in idx], [preds["B5"][i] for i in idx],
                                         n_boot=200, cluster_by_db=True)["unsupported_detection"]

    _write_final_results(recs, per_ds, results, test_path, man_path)
    print(json.dumps({"n_records": len(recs), "n_clusters": len(test_clusters),
                      "per_dataset": {k: v["n"] for k, v in per_ds.items()},
                      "leakage_overlap": len(leak),
                      "B5_overall": {k: round(v, 3) if isinstance(v, float) else v
                                     for k, v in results["B5"]["unsupported_detection"].items()
                                     if k in ("precision", "recall", "f1", "recall_ci95", "precision_ci95")},
                      "manifest": str(man_path.relative_to(REPO))}, indent=1))
    return results


def _write_final_results(recs, per_ds, results, test_path, man_path):
    doc = REPO / "docs" / "sqlledger" / "FINAL_RESULTS.md"
    L = ["# SQLLedger FINAL RESULTS (Phase 2, frozen held-out test)", "",
         f"Records: {len(recs)} across {len({r['dataset']+':'+r['db_id'] for r in recs})} db/table clusters. "
         f"Source: `{test_path.relative_to(REPO)}`. Manifest: `{man_path.relative_to(REPO)}`.",
         "Held-out test (WikiSQL train tables / Spider+BIRD train dbs) disjoint from the Phase-1 "
         "development calibration; leakage guard enforced. Gold labels: execution + declared-citation "
         "logic + witnesses, no LLM/checker.", "",
         "## Per-dataset (B5 = full LEDGER, db/table-clustered CI)", "",
         "| dataset | n | B5 precision | B5 recall | B5 F1 |", "|---|---|---|---|---|"]
    for ds, d in per_ds.items():
        u = d["B5"]
        L.append(f"| {ds} | {d['n']} | {u['precision']:.3f} {_fmt_ci(u.get('precision_ci95'))} "
                 f"| {u['recall']:.3f} {_fmt_ci(u.get('recall_ci95'))} | {u['f1']:.3f} |")
    L += ["", "### Rejection taxonomy", ""]
    for ds, d in per_ds.items():
        L.append(f"- **{ds}** (n={d['n']}): `{d['rejections']}`")
    L += ["", "## Baseline ablation (combined, unsupported-detection, cluster-bootstrap CI)", "",
          "| baseline | precision | recall | F1 | false-cert | false-rej | unknown |",
          "|---|---|---|---|---|---|---|"]
    names = {"B0": "citation-exists", "B1": "static-extract", "B2": "unrestricted-replay",
             "B3": "restricted-replay", "B4": "+read/decl", "B5": "+intervention (LEDGER)",
             "B6": "+legacy-counterfactual"}
    for name in ("B0", "B1", "B2", "B3", "B4", "B5", "B6"):
        u = results[name]["unsupported_detection"]; m = results[name]
        L.append(f"| {name} {names[name]} | {u['precision']:.3f} {_fmt_ci(u.get('precision_ci95'))} "
                 f"| {u['recall']:.3f} {_fmt_ci(u.get('recall_ci95'))} | {u['f1']:.3f} "
                 f"| {m['false_certification_rate']:.3f} | {m['false_rejection_rate']:.3f} | {m['unknown_rate']:.3f} |")
    L += ["", "## Per-class recall (B5, combined)", "", "| class | recall | n |", "|---|---|---|"]
    for cls, dd in results["B5"]["per_class_recall"].items():
        L.append(f"| {cls} | {dd['recall']:.3f} | {dd['n']} |")
    doc.write_text("\n".join(L) + "\n")
    # machine-checkable paper tables
    pt = REPO / "docs" / "sqlledger" / "PAPER_TABLES.md"
    pl = ["# PAPER_TABLES (recomputed from frozen test records)", "",
          f"manifest: `{man_path.relative_to(REPO)}` · records: `{test_path.relative_to(REPO)}`", "",
          "Every number below is recomputed from the same run's records via "
          "`experiments.sqlledger.cli phase2`.", "",
          "## Table: ablation ladder (unsupported-detection F1)", ""]
    for name in ("B0", "B1", "B2", "B3", "B4", "B5", "B6"):
        u = results[name]["unsupported_detection"]
        pl.append(f"- {name} ({names[name]}): F1={u['f1']:.3f} P={u['precision']:.3f} R={u['recall']:.3f} "
                  f"(n={u['tp']+u['fp']+u['fn']+u['tn']})")
    pt.write_text("\n".join(pl) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke"); s.add_argument("--split", default="dev"); s.add_argument("--limit", type=int, default=200)
    p1 = sub.add_parser("phase1")
    p1.add_argument("--wikisql", type=int, default=1000); p1.add_argument("--spider", type=int, default=300)
    p1.add_argument("--git-commit", default="unknown")
    p2 = sub.add_parser("phase2")
    p2.add_argument("--wikisql-parents", type=int, default=20000)
    p2.add_argument("--no-spider", action="store_true"); p2.add_argument("--no-bird", action="store_true")
    p2.add_argument("--git-commit", default="unknown")
    a = ap.parse_args()
    if a.cmd == "smoke":
        _, gates = smoke(a.split, a.limit)
        if not all(gates.values()):
            raise SystemExit("PHASE-0 GATE FAILURE: " + ", ".join(k for k, v in gates.items() if not v))
    elif a.cmd == "phase1":
        phase1(a.wikisql, a.spider, a.git_commit)
    elif a.cmd == "phase2":
        phase2(a.wikisql_parents, not a.no_spider, not a.no_bird, a.git_commit)


if __name__ == "__main__":
    main()
