"""LEDGER: counterfactual data-dependence check (A3 / DR4) over DABStep supported episodes.
Post-hoc over records_ledger_dabstep_support.jsonl + saved trace programs. Pure CPU, no LLM.
Interventions: I_num (scale numeric cells/values x1.37; CSV cells and JSON numbers) and
I_row (drop a deterministic 10% of CSV data rows / JSON top-level list elements).
Vacuous (a3) only if BOTH leave the recomputed value unchanged within tolerance.
Run: python3 -m kramabench_integration.dabstep_counterfactual
"""
from __future__ import annotations
import json, os, shutil, tempfile, subprocess, sys
from pathlib import Path

from .dabstep_support_run import (CONTEXT_DIR, ALLOWED_FILES, KENV_PY, _scrubbed_env,
                                  exec_program, _norm_num)

REPO = Path(__file__).resolve().parents[1]
REC_IN = REPO / "records" / "records_ledger_dabstep_support.jsonl"
REC_OUT = REPO / "records" / "dabstep_counterfactual.jsonl"
DOC_OUT = REPO / "docs" / "DABSTEP_COUNTERFACTUAL.md"
TOL = 1e-4
SCALE = 1.37


def _perturb_csv(src: Path, dst: Path, mode: str) -> None:
    import pandas as pd
    df = pd.read_csv(src, low_memory=False)
    if mode == "num":
        for c in df.columns:
            if df[c].dtype.kind in "if":
                df[c] = df[c] * SCALE
    else:  # row: drop deterministic 10%
        df = df[[i % 10 != 0 for i in range(len(df))]]
    df.to_csv(dst, index=False)


def _scale_json(x, factor):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x * factor
    if isinstance(x, list):
        return [_scale_json(v, factor) for v in x]
    if isinstance(x, dict):
        return {k: _scale_json(v, factor) for k, v in x.items()}
    return x


def _perturb_json(src: Path, dst: Path, mode: str) -> None:
    obj = json.load(open(src))
    if mode == "num":
        obj = _scale_json(obj, SCALE)
    else:
        if isinstance(obj, list):
            obj = [v for i, v in enumerate(obj) if i % 10 != 0]
        elif isinstance(obj, dict):
            keys = sorted(obj)
            drop = set(keys[:: 10])
            obj = {k: v for k, v in obj.items() if k not in drop}
    json.dump(obj, open(dst, "w"))


def _perturbed_reexec(code: str, files_cited: list[str], mode: str):
    tmpdir = Path(tempfile.mkdtemp(prefix=f"dabstep_cf_{mode}_"))
    try:
        mirror = tmpdir / "context"; mirror.mkdir(parents=True)
        for name in files_cited:
            if name not in ALLOWED_FILES:
                continue
            src, dst = CONTEXT_DIR / name, mirror / name
            try:
                if name.endswith(".csv"):
                    _perturb_csv(src, dst, mode)
                elif name.endswith(".json"):
                    _perturb_json(src, dst, mode)
                else:
                    shutil.copy2(src, dst)  # md/txt: copy verbatim
            except Exception:
                shutil.copy2(src, dst)
        rewritten = code.replace(str(CONTEXT_DIR), str(mirror))
        rewritten = rewritten.replace("/mnt/data/yanmy/LakeOrch/datasets/dabstep/data/context", str(mirror))
        res = exec_program(rewritten, mirror, timeout=120)
        if res["timeout"] or res["returncode"] != 0:
            return None, "exec_changed"       # failure under intervention = sensitivity
        return res["answer"], None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    rows = [json.loads(l) for l in open(REC_IN)]
    sup = [r for r in rows if r.get("grounding_label") == "supported"]
    out, n_checked, n_vac, n_sens, n_na = [], 0, 0, 0, 0
    for r in sup:
        base = _norm_num(r.get("pred"))
        tp = REPO / r["trace_path"]
        if base is None or not tp.exists():
            n_na += 1
            out.append({**{k: r[k] for k in ("taskid", "level", "pred", "grounding_label")},
                        "cf_status": "not_applicable"})
            continue
        code = open(tp).read()
        n_checked += 1
        verdicts = {}
        insens = 0
        for mode in ("num", "row"):
            v, err = _perturbed_reexec(code, r.get("files_cited") or [], mode)
            nv = _norm_num(v)
            unchanged = (err is None and nv is not None and abs(nv - base) <= TOL * max(1, abs(base)))
            verdicts[mode] = {"value": v, "err": err, "unchanged": bool(unchanged)}
            insens += int(unchanged)
        vac = insens == 2
        n_vac += int(vac); n_sens += int(not vac)
        out.append({**{k: r[k] for k in ("taskid", "level", "pred", "grounding_label")},
                    "cf_status": "vacuous" if vac else "sensitive", "cf_detail": verdicts})
        print(f"  {r['taskid']}: {'VACUOUS' if vac else 'sensitive'}", flush=True)
    with open(REC_OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False, default=str) + "\n")
    summary = {"supported": len(sup), "numeric_checked": n_checked,
               "vacuous_a3": n_vac, "sensitive": n_sens, "not_applicable_or_text": n_na}
    doc = ["# LEDGER — DABStep counterfactual data-dependence check (A3/DR4)\n",
           "Same interventions as the KramaBench check (numeric cells x1.37 incl. JSON numbers; "
           "drop 10% rows / list elements; both-unchanged => a3_vacuous). Post-hoc over the 60-task "
           "DABStep support run; deterministic, zero-LLM.\n",
           f"- supported episodes: {len(sup)}",
           f"- numeric checked: {n_checked}",
           f"- **vacuous (a3): {n_vac}**",
           f"- sensitive: {n_sens}",
           f"- not-applicable (text/list answers or missing trace): {n_na}\n",
           "Vacuous episodes:\n"]
    for o in out:
        if o.get("cf_status") == "vacuous":
            doc.append(f"- {o['taskid']} (level {o.get('level')}): pred={str(o['pred'])[:40]}")
    DOC_OUT.write_text("\n".join(doc) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
