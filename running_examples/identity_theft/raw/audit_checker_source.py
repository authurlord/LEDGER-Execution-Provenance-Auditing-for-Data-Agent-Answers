"""LEDGER v2 STEP 3b/3c — post-hoc trace capture + provenance over DS-Guru SUT artifacts.

Consumes KramaBench/system_scratch/<SUT>/<query_id>/_intermediate/ (pipeline-N.py, pipeline-N_out.json,
errors-N.txt, token_count-N.json, ledger_cost.json) — the SUT's reasoning is UNTOUCHED; everything here
is post-hoc and deterministic (no LLM anywhere).

Per task:
  TRACE    : cited files = file-path string literals actually referenced by the FINAL executed program
             (AST string scan, filtered to existing files); columns/ops summary; exec status.
  SUPPORT  : re-execute the final program in a scrubbed env with the filesystem view RESTRICTED to the
             cited files (cited files copied into a temp mirror; the data-root prefix in the code is
             rewritten to the mirror). An uncited read then fails loudly.
  CLASSES  : (a) stated != recomputed on cited sources (numeric tol = frozen point 1e-4)
             (b) exec failed but the system still stated an answer
             (c) uncited-source dependency (mirror re-exec hits FileNotFoundError)
             (d) required operands absent from cited sources (KeyError / missing column)
             else supported; 'unknown' for non-recomputable cases (counted separately).
Emits records/records_ledger_n2_kb_<domain>_<setting>.jsonl via cost_accounting.build_record
(LLM cost from ledger_cost.json + token_count files; checker steps tokens=0).

Run: conda run -n kramabench python -m kramabench_integration.kb_trace_provenance \
       --sut BaselineLLMSystemLocalQwenFewShot --setting oracle --domains environment,wildfire,legal,archeology
"""
from __future__ import annotations
import os, re, io, json, glob, shutil, argparse, subprocess, tempfile, contextlib
from . import cost_accounting as CA

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(REPO, "KramaBench")
SCRATCH = os.path.join(KB, "system_scratch")
DATA = os.path.join(KB, "data")
KENV_PY = "/home/yanmy/miniforge3/envs/kramabench/bin/python"
TOL_REL = 1e-4          # frozen operating point (LEDGER_G1_VERDICT.md)
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


# ---------------- trace capture (3b) ----------------
def _last_try(idir):
    tries = sorted(int(m.group(1)) for f in os.listdir(idir)
                   for m in [re.match(r"pipeline-(\d+)\.py$", f)] if m)
    return tries[-1] if tries else None


def _string_literals(code):
    import ast
    lits = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lits.append(node.value)
    except SyntaxError:
        lits = re.findall(r"['\"]([^'\"]{4,300})['\"]", code)
    return lits


_BASENAME_IDX = {}


def _domain_index(domain):
    """basename -> relpath index for the domain lake (built once; light domains have unique basenames)."""
    if domain not in _BASENAME_IDX:
        droot = os.path.join(DATA, domain)
        idx = {}
        for dirpath, _, files in os.walk(droot):
            for f in files:
                idx.setdefault(f, os.path.relpath(os.path.join(dirpath, f), droot))
        _BASENAME_IDX[domain] = idx
    return _BASENAME_IDX[domain]


def capture_trace(qdir, domain):
    """-> dict(trace) or None if no executed pipeline."""
    idir = os.path.join(qdir, "_intermediate")
    if not os.path.isdir(idir):
        return None
    n = _last_try(idir)
    if n is None:
        return None
    code = open(os.path.join(idir, f"pipeline-{n}.py"), errors="ignore").read()
    err_fp = os.path.join(idir, f"errors-{n}.txt")
    out_fp = os.path.join(idir, f"pipeline-{n}_out.json")
    stderr = open(err_fp, errors="ignore").read() if os.path.exists(err_fp) else ""
    stdout = open(out_fp, errors="ignore").read() if os.path.exists(out_fp) else ""
    droot = os.path.join(DATA, domain)
    idx = _domain_index(domain)
    cited, uncited_paths = [], []
    for s in _string_literals(code):
        if not re.search(r"\.(csv|xlsx|txt|json|tsv|html)$", s, re.I):
            continue
        # resolution order: absolute | relative-to-droot | basename index (codes cite bare filenames
        # and join a base_dir variable at runtime — plain join missed the input/ subdir)
        cands = ([s] if os.path.isabs(s) else [os.path.join(droot, s)])
        base = os.path.basename(s)
        if base in idx:
            cands.append(os.path.join(droot, idx[base]))
        hit = next((c for c in cands if os.path.exists(c)), None)
        if hit:
            real = os.path.realpath(hit)
            if real.startswith(os.path.realpath(droot)):
                cited.append(os.path.relpath(real, droot))
            else:
                uncited_paths.append(s)           # exists but OUTSIDE the domain lake
        else:
            uncited_paths.append(s)               # referenced but not found
    cols = sorted(set(re.findall(r"""\[['"]([A-Za-z_][A-Za-z0-9 _%/().-]{1,40})['"]\]""", code)))[:30]
    ops = sorted(set(re.findall(r"\.(groupby|merge|sum|mean|max|min|count|sort_values|query|pivot|value_counts|nunique|corr)\(", code)))
    # per-try token totals
    tin = tout = 0
    for tf in glob.glob(os.path.join(idir, "token_count-*.json")):
        try:
            t = json.load(open(tf)); tin += t.get("token_usage_input", 0); tout += t.get("token_usage_output", 0)
        except Exception:
            pass
    cost = {}
    cfp = os.path.join(idir, "ledger_cost.json")
    if os.path.exists(cfp):
        try: cost = json.load(open(cfp))
        except Exception: pass
    return {"try": n, "code": code, "stderr": stderr, "stdout": stdout,
            "cited": sorted(set(cited)), "uncited_refs": sorted(set(uncited_paths)),
            "columns": cols, "ops": ops, "turns": n + 1,
            "tokens_in": cost.get("tokens_in", tin), "tokens_out": cost.get("tokens_out", tout),
            "latency_s": cost.get("latency_s", 0.0),
            "offered": cost.get("subset_files_offered", []),
            "code_path": os.path.join(idir, f"pipeline-{n}.py")}


def _last_json_obj(text):
    """Parse the LAST JSON object in mixed stdout (debug prints + pretty-printed json)."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    dec = json.JSONDecoder()
    last = None
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                last = obj
        except Exception:
            pass
        i = text.find("{", i + 1)
    return last


def stated_answer(trace):
    """The system's stated main-task answer from the executed stdout json (None if absent/unparseable)."""
    d = _last_json_obj(trace["stdout"])
    if isinstance(d, dict):
        return d.get("main-task", d.get("main_task"))
    return None


# ---------------- restricted re-execution (3c) ----------------
def _scrubbed_env():
    safe = {}
    for k, v in os.environ.items():
        if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "OPENAI", "DEEPSEEK", "ANTHROPIC")):
            continue
        safe[k] = v
    safe["HOME"] = "/tmp"
    return safe


def reexecute_restricted(trace, domain, timeout=180):
    """Re-run the final program with the FS view restricted to cited files. -> (value, err_kind, detail)
    err_kind in {None, 'uncited_read', 'missing_operand', 'exec_error', 'timeout', 'no_output'}."""
    droot = os.path.realpath(os.path.join(DATA, domain))
    tmp = tempfile.mkdtemp(prefix="ledger_prov_")
    try:
        mirror = os.path.join(tmp, "data")
        for rel in trace["cited"]:
            src = os.path.join(droot, rel); dst = os.path.join(mirror, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try: shutil.copy(src, dst)
            except Exception: pass
        code = trace["code"].replace(droot, mirror).replace(os.path.join(DATA, domain), mirror)
        cf = os.path.join(tmp, "prog.py")
        open(cf, "w").write(code)
        r = subprocess.run([KENV_PY, cf], capture_output=True, text=True, timeout=timeout,
                           cwd=mirror, env=_scrubbed_env())
        if r.returncode != 0 or r.stderr.strip():
            s = r.stderr
            if "FileNotFoundError" in s or "No such file" in s:
                return None, "uncited_read", s[-400:]
            if "KeyError" in s or "columns" in s.lower() and "not" in s.lower():
                return None, "missing_operand", s[-400:]
            return None, "exec_error", s[-400:]
        if not r.stdout.strip():
            return None, "no_output", ""
        d = _last_json_obj(r.stdout)
        if isinstance(d, dict):
            return d.get("main-task", d.get("main_task")), None, ""
        return r.stdout.strip().splitlines()[-1], None, "non-json stdout tail"
    except subprocess.TimeoutExpired:
        return None, "timeout", ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _norm_num(x):
    try:
        s = str(x).replace(",", "").replace("%", "").strip()
        m = _NUM.search(s)
        return float(m.group()) if m else None
    except Exception:
        return None


def support_label(stated, trace, domain):
    """-> (label, cls, detail). label ∈ {supported, unsupported, unknown}; cls ∈ {a,b,c,d,None}."""
    exec_failed = bool(trace["stderr"].strip())
    if stated in (None, "") or (isinstance(stated, str) and stated.startswith("Pipeline not successful")):
        return "unknown", None, "no stated answer (abstained/failed)"
    if exec_failed:
        return "unsupported", "b", "final-try exec had errors but an answer was stated"
    if not trace["cited"]:
        return "unsupported", "c", "no citable file reference in the executed program"
    recomputed, err, detail = reexecute_restricted(trace, domain)
    if err == "uncited_read":
        return "unsupported", "c", detail
    if err == "missing_operand":
        return "unsupported", "d", detail
    if err in ("exec_error", "timeout", "no_output"):
        return "unknown", None, f"restricted re-exec {err}: {detail[:200]}"
    sn, rn = _norm_num(stated), _norm_num(recomputed)
    if sn is not None and rn is not None:
        ok = abs(rn - sn) <= max(TOL_REL * abs(sn), 1e-9)
        return ("supported" if ok else "unsupported"), (None if ok else "a"), f"stated={sn} recomputed={rn}"
    ss, rs = str(stated).strip().lower(), str(recomputed).strip().lower()
    if ss and rs:
        ok = ss == rs or ss in rs or rs in ss
        return ("supported" if ok else "unsupported"), (None if ok else "a"), f"stated={ss[:60]!r} recomputed={rs[:60]!r}"
    return "unknown", None, "non-comparable types"


# ---------------- driver ----------------
def gold_map():
    rows = {}
    for l in open(os.path.join(REPO, "datasets", "normalized", "kramabench_tasks.jsonl")):
        t = json.loads(l); rows[t["taskid"]] = t
    return rows


def run(sut, setting, domains):
    gm = gold_map()
    outdir = os.path.join(REPO, "records"); os.makedirs(outdir, exist_ok=True)
    for domain in domains:
        out = os.path.join(outdir, f"records_ledger_n2_kb_{domain}_{setting}.jsonl")
        W = CA.RecordWriter(out)
        stats = {"n": 0, "supported": 0, "unsupported": 0, "unknown": 0,
                 "a": 0, "b": 0, "c": 0, "d": 0}
        for qdir in sorted(glob.glob(os.path.join(SCRATCH, sut, f"{domain}-*"))):
            qid = os.path.basename(qdir)
            trace = capture_trace(qdir, domain)
            if trace is None:
                continue
            stated = stated_answer(trace)
            label, cls, detail = support_label(stated, trace, domain)
            gold = gm.get(qid, {})
            m = CA.CostMeter()
            m.add_llm((trace["tokens_in"], trace["tokens_out"])); m.turn(trace["turns"]); m.tool(1)
            rec = CA.build_record(
                taskid=qid, domain=domain, system=f"{sut}|{setting}",
                pred=stated, gold=gold.get("answer"), score=-1,   # official score joined from measures CSV
                sources_selected=trace["cited"], gold_sources=gold.get("gold_sources", []),
                meter=m, trace_path=trace["code_path"],
                provenance=[{"source_file": c, "columns": trace["columns"], "ops": trace["ops"]}
                            for c in trace["cited"]],
                grounding_label=label, unsupported_class=cls, support_detail=detail[:300],
                offered_files=trace["offered"][:40], uncited_refs=trace["uncited_refs"][:10],
                exec_failed=bool(trace["stderr"].strip()))
            rec["latency_s"] = trace["latency_s"]
            W.write(rec)
            stats["n"] += 1; stats[label] += 1
            if cls: stats[cls] += 1
        print(f"[{domain}/{setting}] {json.dumps(stats)} -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sut", required=True)
    ap.add_argument("--setting", required=True, choices=["oracle", "discovery"])
    ap.add_argument("--domains", default="environment,wildfire,legal,archeology")
    a = ap.parse_args()
    run(a.sut, a.setting, [d for d in a.domains.split(",") if d])
