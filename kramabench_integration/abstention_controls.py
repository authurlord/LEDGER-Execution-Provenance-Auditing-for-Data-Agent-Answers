"""LEDGER E4 - abstention controls for selective answering.

Runs deterministic CPU controls plus a DeepSeek self-reported confidence baseline
over KramaBench discovery-setting stated-answer episodes. The confidence baseline
is a baseline only: it never writes or modifies `grounding_label`.

Smoke:
  python3 -m kramabench_integration.abstention_controls --mode smoke --limit 5

Full:
  python3 -m kramabench_integration.abstention_controls --mode full
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import math
import os
import random
import re
import time
from typing import Any

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(REPO, ".env")
RECORD_GLOB = os.path.join(REPO, "records", "records_ledger_n2_kb_*_discovery.jsonl")
TASKS_PATH = os.path.join(REPO, "datasets", "normalized", "kramabench_tasks.jsonl")
OUT_JSONL = os.path.join(REPO, "records", "kb_abstention_controls.jsonl")
OUT_DOC = os.path.join(REPO, "docs", "ABSTENTION_CONTROLS.md")

DEFAULT_MODEL = "deepseek-v4-flash"
RANDOM_SEED = 20260709
RANDOM_RESAMPLES = 100
BOOTSTRAP_RESAMPLES = 10_000

PRICE_PER_1M_INPUT_USD = 0.14
PRICE_PER_1M_OUTPUT_USD = 0.28


def read_env(path=ENV_PATH):
    vals = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY") if not vals.get(k)]
    if missing:
        raise RuntimeError(f"missing required .env keys: {', '.join(missing)}")
    return vals


def _task_questions():
    out = {}
    with open(TASKS_PATH) as f:
        for line in f:
            row = json.loads(line)
            out[row["taskid"]] = row.get("query") or ""
    return out


def _is_stated(pred: Any) -> bool:
    return pred is not None and str(pred).strip() != ""


def _finite_score(score: Any) -> bool:
    try:
        return math.isfinite(float(score))
    except Exception:
        return False


def _domain_from_path(path: str) -> str:
    m = re.search(r"records_ledger_n2_kb_([^_]+)_discovery\.jsonl$", os.path.basename(path))
    return m.group(1) if m else ""


def _cited_from_record(row: dict) -> dict:
    files, cols, ops = [], [], []
    for item in row.get("provenance") or []:
        if item.get("source_file"):
            files.append(item["source_file"])
        cols.extend(item.get("columns") or [])
        ops.extend(item.get("ops") or [])
    return {
        "files": sorted(set(map(str, files))),
        "columns": sorted(set(map(str, cols))),
        "ops": sorted(set(map(str, ops))),
        "offered_files": list(map(str, row.get("offered_files") or row.get("sources_selected") or [])),
    }


def load_episodes(limit: int | None = None):
    questions = _task_questions()
    included, excluded = [], []
    for path in sorted(glob.glob(RECORD_GLOB)):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not _is_stated(r.get("pred")):
                    continue
                base = {
                    "taskid": r.get("taskid"),
                    "domain": r.get("domain") or _domain_from_path(path),
                    "question": questions.get(r.get("taskid"), ""),
                    "stated_answer": r.get("pred"),
                    "score": r.get("score"),
                    "grounding_label": r.get("grounding_label"),
                    "unsupported_class": r.get("unsupported_class"),
                    "unsupported_subtype": r.get("unsupported_subtype"),
                    "support_detail": r.get("support_detail"),
                    "exec_failed": bool(r.get("exec_failed")),
                    "trace_path": r.get("trace_path"),
                    "cited": _cited_from_record(r),
                    "record_source": os.path.relpath(path, REPO),
                }
                if not _finite_score(r.get("score")):
                    excluded.append({**base, "exclude_reason": "non_finite_official_score"})
                    continue
                base["score"] = float(r.get("score"))
                included.append(base)
    included.sort(key=lambda x: (x["domain"], x["taskid"]))
    if limit is not None:
        included = included[:limit]
    return included, excluded


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


def build_confidence_prompt(item: dict):
    payload = {
        "question": item.get("question") or "",
        "stated_answer": item.get("stated_answer"),
        "cited_files": item["cited"].get("files") or item["cited"].get("offered_files") or [],
        "cited_columns": item["cited"].get("columns") or [],
        "cited_operations": item["cited"].get("ops") or [],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are estimating the reliability of a data-agent stated answer from its "
                "visible cited metadata. Do not use or infer any hidden gold answer. Return "
                "only JSON with keys confidence and rationale. confidence must be a number "
                "from 0 to 1, where 1 means you are highly confident the stated answer is "
                "supported by the cited evidence and 0 means not confident."
            ),
        },
        {
            "role": "user",
            "content": "Estimate confidence for this stated answer:\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def call_confidence(env: dict, model: str, item: dict, timeout_s: int = 60) -> dict:
    base = env["DEEPSEEK_BASE_URL"].rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload = {
        "model": model,
        "messages": build_confidence_prompt(item),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {env['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    latency = round(time.perf_counter() - t0, 3)
    resp.raise_for_status()
    body = resp.json()
    parsed = _extract_json(body["choices"][0]["message"]["content"])
    try:
        conf = max(0.0, min(1.0, float(parsed.get("confidence"))))
    except Exception:
        conf = 0.0
    usage = body.get("usage") or {}
    tin = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    tout = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    cost = (tin / 1_000_000) * PRICE_PER_1M_INPUT_USD + (tout / 1_000_000) * PRICE_PER_1M_OUTPUT_USD
    return {
        "confidence_score": conf,
        "confidence_rationale": str(parsed.get("rationale") or "")[:800],
        "confidence_parse_ok": True,
        "confidence_tokens_in": tin,
        "confidence_tokens_out": tout,
        "confidence_latency_s": latency,
        "confidence_cost_usd": round(cost, 8),
    }


def safe_confidence(env, model, item, timeout_s):
    try:
        return call_confidence(env, model, item, timeout_s)
    except Exception as e:
        return {
            "confidence_score": 0.0,
            "confidence_rationale": "call_or_parse_error",
            "confidence_parse_ok": False,
            "confidence_error": str(e)[:300],
            "confidence_tokens_in": 0,
            "confidence_tokens_out": 0,
            "confidence_latency_s": 0.0,
            "confidence_cost_usd": 0.0,
        }


def load_confidence_cache(path=OUT_JSONL):
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("domain"), row.get("taskid"))
        if key and row.get("confidence_score") is not None:
            out[key] = {k: row.get(k) for k in (
                "confidence_score",
                "confidence_rationale",
                "confidence_parse_ok",
                "confidence_error",
                "confidence_tokens_in",
                "confidence_tokens_out",
                "confidence_latency_s",
                "confidence_cost_usd",
            )}
    return out


def _precision(items, keep):
    kept = [x for x, k in zip(items, keep) if k]
    return sum(x["score"] for x in kept) / len(kept) if kept else None


def _policy_masks(items, target_withhold):
    n = len(items)
    unsupported = [x.get("grounding_label") == "unsupported" for x in items]
    ledger_keep = [not x for x in unsupported]

    exec_bad = [bool(x.get("exec_failed")) or "no output" in str(x.get("support_detail", "")).lower() for x in items]
    exec_keep = [not x for x in exec_bad]

    conf_order = sorted(range(n), key=lambda i: (items[i].get("confidence_score", 0.0), items[i]["taskid"]))
    conf_withheld = set(conf_order[:target_withhold])
    conf_keep = [i not in conf_withheld for i in range(n)]
    return ledger_keep, conf_keep, exec_keep


def _random_masks(n, target_withhold, resamples=RANDOM_RESAMPLES, seed=RANDOM_SEED):
    rng = random.Random(seed)
    masks = []
    for _ in range(resamples):
        withheld = set(rng.sample(range(n), target_withhold))
        masks.append([i not in withheld for i in range(n)])
    return masks


def _curve_from_order(items, order):
    n = len(items)
    rows = []
    withheld = set()
    for k in range(0, n):
        if k:
            withheld.add(order[k - 1])
        keep = [i not in withheld for i in range(n)]
        rows.append({
            "withheld": k,
            "coverage": (n - k) / n if n else 0.0,
            "kept": n - k,
            "precision": _precision(items, keep),
        })
    return rows


def _aurc(curve):
    pts = [(r["coverage"], 1.0 - r["precision"]) for r in curve if r["precision"] is not None]
    pts.sort()
    if len(pts) < 2:
        return None
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return round(area, 6)


def _mean_std(vals):
    mean = sum(vals) / len(vals) if vals else None
    if not vals:
        return None, None
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, math.sqrt(var)


def _bootstrap_ci(items, keep_a, keep_b, resamples=BOOTSTRAP_RESAMPLES, seed=RANDOM_SEED + 17):
    rng = random.Random(seed)
    n = len(items)
    diffs = []
    for _ in range(resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        a = [items[i]["score"] for i in idxs if keep_a[i]]
        b = [items[i]["score"] for i in idxs if keep_b[i]]
        if not a or not b:
            continue
        diffs.append((sum(a) / len(a)) - (sum(b) / len(b)))
    diffs.sort()
    if not diffs:
        return None
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    return [round(lo, 6), round(hi, 6)]


def _bootstrap_ci_random_mean(items, ledger_keep, random_masks, resamples=BOOTSTRAP_RESAMPLES, seed=RANDOM_SEED + 23):
    rng = random.Random(seed)
    n = len(items)
    diffs = []
    for _ in range(resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        led = [items[i]["score"] for i in idxs if ledger_keep[i]]
        if not led:
            continue
        led_p = sum(led) / len(led)
        rps = []
        for mask in random_masks:
            vals = [items[i]["score"] for i in idxs if mask[i]]
            if vals:
                rps.append(sum(vals) / len(vals))
        if rps:
            diffs.append(led_p - (sum(rps) / len(rps)))
    diffs.sort()
    if not diffs:
        return None
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    return [round(lo, 6), round(hi, 6)]


def summarize(items):
    n = len(items)
    target_withhold = sum(1 for x in items if x.get("grounding_label") == "unsupported")
    ledger_keep, conf_keep, exec_keep = _policy_masks(items, target_withhold)
    random_keep = _random_masks(n, target_withhold)

    random_precs = [_precision(items, m) for m in random_keep]
    random_mean, random_std = _mean_std(random_precs)
    random_ci = _bootstrap_ci_random_mean(items, ledger_keep, random_keep)

    matched = {
        "n": n,
        "target_withheld": target_withhold,
        "target_coverage": f"{n - target_withhold}/{n}",
        "ledger": {
            "withheld": sum(not k for k in ledger_keep),
            "coverage": f"{sum(ledger_keep)}/{n}",
            "precision": round(_precision(items, ledger_keep), 6),
        },
        "random": {
            "resamples": RANDOM_RESAMPLES,
            "withheld": target_withhold,
            "coverage": f"{n - target_withhold}/{n}",
            "precision_mean": round(random_mean, 6),
            "precision_std": round(random_std, 6),
            "ledger_minus_random_mean_bootstrap_ci95": random_ci,
        },
        "confidence": {
            "withheld": sum(not k for k in conf_keep),
            "coverage": f"{sum(conf_keep)}/{n}",
            "precision": round(_precision(items, conf_keep), 6),
            "ledger_minus_confidence_bootstrap_ci95": _bootstrap_ci(items, ledger_keep, conf_keep),
        },
        "execution_health": {
            "withheld": sum(not k for k in exec_keep),
            "coverage": f"{sum(exec_keep)}/{n}",
            "precision": round(_precision(items, exec_keep), 6),
            "matched_coverage_note": (
                "native policy cannot reach matched coverage without adding another abstention rule"
                if sum(not k for k in exec_keep) != target_withhold else "matched"
            ),
        },
    }
    gate = {
        "criterion": "LEDGER precision >= random mean and >= confidence at matched coverage",
        "pass": (
            matched["ledger"]["precision"] >= matched["random"]["precision_mean"]
            and matched["ledger"]["precision"] >= matched["confidence"]["precision"]
        ),
        "no_regret_case_evidence": True,
        "significance_note": "Report as no-regret case evidence if bootstrap CIs include 0.",
    }

    ledger_order = sorted(range(n), key=lambda i: (items[i].get("grounding_label") != "unsupported", items[i]["taskid"]))
    conf_order = sorted(range(n), key=lambda i: (items[i].get("confidence_score", 0.0), items[i]["taskid"]))
    exec_order = sorted(range(n), key=lambda i: (not (bool(items[i].get("exec_failed")) or "no output" in str(items[i].get("support_detail", "")).lower()), items[i]["taskid"]))
    curves = {
        "ledger": _curve_from_order(items, ledger_order),
        "confidence": _curve_from_order(items, conf_order),
        "execution_health": _curve_from_order(items, exec_order),
    }
    curves["random_mean"] = []
    for k in range(n):
        vals = []
        rng = random.Random(RANDOM_SEED + k)
        for _ in range(RANDOM_RESAMPLES):
            withheld = set(rng.sample(range(n), k))
            vals.append(_precision(items, [i not in withheld for i in range(n)]))
        m, s = _mean_std(vals)
        curves["random_mean"].append({
            "withheld": k,
            "coverage": (n - k) / n if n else 0.0,
            "kept": n - k,
            "precision_mean": round(m, 6),
            "precision_std": round(s, 6),
        })
    aurc = {k: _aurc(v) for k, v in curves.items() if k != "random_mean"}
    aurc["random_mean"] = _aurc([{"coverage": r["coverage"], "precision": r["precision_mean"]} for r in curves["random_mean"]])
    return matched, curves, aurc, gate


def _write_doc(summary, excluded):
    m = summary["matched"]
    lines = [
        "# LEDGER E4 - Abstention controls",
        "",
        "Discovery-setting stated-answer episodes with finite official scores are used for RQ4.",
        f"Excluded stated episodes with non-finite official score: {len(excluded)}.",
        "",
        "## Matched Coverage",
        "",
        f"Target coverage: {m['target_coverage']} (withhold {m['target_withheld']}).",
        "",
        "| strategy | coverage | withheld | delivered official-score precision | note |",
        "|---|---:|---:|---:|---|",
        f"| LEDGER | {m['ledger']['coverage']} | {m['ledger']['withheld']} | {m['ledger']['precision']} | withhold unsupported |",
        f"| random | {m['random']['coverage']} | {m['random']['withheld']} | {m['random']['precision_mean']} +/- {m['random']['precision_std']} | 100 fixed-seed resamples |",
        f"| confidence | {m['confidence']['coverage']} | {m['confidence']['withheld']} | {m['confidence']['precision']} | withhold lowest DeepSeek confidence |",
        f"| execution-health | {m['execution_health']['coverage']} | {m['execution_health']['withheld']} | {m['execution_health']['precision']} | {m['execution_health']['matched_coverage_note']} |",
        "",
        "## Bootstrap CIs",
        "",
        f"- LEDGER - confidence precision CI95: {m['confidence']['ledger_minus_confidence_bootstrap_ci95']}",
        f"- LEDGER - random-mean precision CI95: {m['random']['ledger_minus_random_mean_bootstrap_ci95']}",
        "",
        "If a CI includes 0, this is no-regret case evidence, not a significance claim.",
        "",
        "## AURC",
        "",
        "| strategy | AURC |",
        "|---|---:|",
    ]
    for k, v in summary["aurc"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "Lower AURC is better. AURC is reported as a curve diagnostic; G4 is judged at the",
        "paper's matched coverage point.",
        "",
        "## Coverage-Precision Curve Samples",
        "",
        "| withheld | coverage | LEDGER | confidence | random mean | execution-health |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    n = summary["matched"]["n"]
    sample_withheld = sorted({0, 1, 2, summary["matched"]["target_withheld"], 5, 10, n // 2})
    for k in sample_withheld:
        if k < 0 or k >= n:
            continue
        led = summary["curves"]["ledger"][k]
        conf = summary["curves"]["confidence"][k]
        rnd = summary["curves"]["random_mean"][k]
        exe = summary["curves"]["execution_health"][k]
        lines.append(
            f"| {k} | {led['kept']}/{n} | {round(led['precision'], 6)} | "
            f"{round(conf['precision'], 6)} | {rnd['precision_mean']} | "
            f"{round(exe['precision'], 6)} |"
        )
    lines += [
        "",
        "## G4",
        "",
        f"Criterion: {summary['G4_gate']['criterion']}",
        f"G4: {'PASS' if summary['G4_gate']['pass'] else 'FAIL'}",
        "",
        "Confidence baseline is read-only and does not write `grounding_label`.",
        "",
        "Records: `records/kb_abstention_controls.jsonl`.",
    ]
    with open(OUT_DOC, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout-s", type=int, default=60)
    ap.add_argument("--reuse-confidence", action="store_true",
                    help="Reuse confidence fields from existing records/kb_abstention_controls.jsonl.")
    args = ap.parse_args(argv)
    limit = args.limit if args.limit is not None else (5 if args.mode == "smoke" else None)

    items, excluded = load_episodes(limit=limit)
    env = None if args.reuse_confidence else read_env()
    cache = load_confidence_cache() if args.reuse_confidence else {}
    for item in items:
        cached = cache.get((item.get("domain"), item.get("taskid")))
        if cached:
            item.update(cached)
        else:
            if env is None:
                env = read_env()
            item.update(safe_confidence(env, args.model, item, args.timeout_s))
        assert "grounding_label" in item

    matched, curves, aurc, gate = summarize(items)
    summary = {
        "mode": args.mode,
        "model": args.model,
        "temperature": 0,
        "random_seed": RANDOM_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "items": len(items),
        "excluded_nonfinite_score_stated": len(excluded),
        "score_distribution": dict(Counter(str(x["score"]) for x in items)),
        "grounding_distribution": dict(Counter(x.get("grounding_label") for x in items)),
        "confidence_parse_errors": sum(1 for x in items if not x.get("confidence_parse_ok")),
        "confidence_tokens_in": sum(x.get("confidence_tokens_in", 0) for x in items),
        "confidence_tokens_out": sum(x.get("confidence_tokens_out", 0) for x in items),
        "confidence_cost_usd": round(sum(x.get("confidence_cost_usd", 0.0) for x in items), 8),
        "matched": matched,
        "aurc": aurc,
        "G4_gate": gate,
        "curves": curves,
    }

    if args.mode == "full":
        os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
        os.makedirs(os.path.dirname(OUT_DOC), exist_ok=True)
        with open(OUT_JSONL, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        _write_doc(summary, excluded)

    print(json.dumps({
        "mode": summary["mode"],
        "items": summary["items"],
        "excluded_nonfinite_score_stated": summary["excluded_nonfinite_score_stated"],
        "confidence_parse_errors": summary["confidence_parse_errors"],
        "matched": summary["matched"],
        "aurc": summary["aurc"],
        "G4_gate": summary["G4_gate"],
    }, indent=1, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
