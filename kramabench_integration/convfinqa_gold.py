"""Original ConvFinQA (czyssrs/ConvFinQA) gold-program loader/parser.

Provides per-QA: question, stated answer (exe_ans), the gold PROGRAM, and the **cited operand set**
(numeric literals + cited consts the gold derivation consumes) + program depth. These operands are the
"citations" for the operand-constrained support checker (LEDGER v2 STEP 1).

Data: datasets/convfinqa_original/data/dev.json (421 docs -> 542 QAs; all arithmetic ops in dev).
"""
from __future__ import annotations
import os, re, json, hashlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_JSON = os.path.join(REPO, "datasets", "convfinqa_original", "data", "dev.json")

_CALL = re.compile(r"(\w+)\(([^)]*)\)")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
ARITH_OPS = {"add", "subtract", "multiply", "divide", "exp", "greater"}


def parse_program(prog: str):
    """-> (calls[(op, [args])], operands set[float], consts set[float], depth int, ok bool).
    operands = numeric literals; consts = const_N values cited. ok=False if non-arithmetic/unparseable."""
    calls = [(op, [a.strip() for a in args.split(",") if a.strip()]) for op, args in _CALL.findall(prog or "")]
    operands, consts = set(), set()
    ok = bool(calls)
    for op, args in calls:
        if op not in ARITH_OPS:
            ok = False
        for a in args:
            if a.startswith("#") or a == "none":
                continue
            if a.startswith("const_"):
                m = _NUM.search(a.replace("m1", "-1"))
                if m:
                    consts.add(float(m.group()))
                else:
                    ok = False
            else:
                a2 = a.rstrip("%")
                if re.fullmatch(r"-?\d+(?:\.\d+)?", a2):
                    v = float(a2)
                    if a.endswith("%"):
                        v = v / 100.0
                    operands.add(v)
                else:
                    ok = False
    return calls, operands, consts, len(calls), ok


def _percent_context(question: str, answer: str, prog: str) -> bool:
    ql = (question or "").lower()
    return ("%" in str(answer) or "%" in ql or "percent" in ql or "const_100" in (prog or ""))


def load_qas(path: str = DEV_JSON):
    """Yield normalized QA dicts: {uid, qhash, question, exe_ans, answer_str, program, operands, consts,
    depth, percent_ctx, ok}."""
    data = json.load(open(path))
    out = []
    for ex in data:
        # context numbers (table + pre/post text) — for the SAME-DATA free-closure baseline row
        ctx_parts = []
        for row in ex.get("table", []) or []:
            ctx_parts += [str(c) for c in row]
        ctx_parts += (ex.get("pre_text") or []) + (ex.get("post_text") or [])
        ctx_nums = []
        for m in _NUM.findall(" ".join(ctx_parts).replace(",", "")):
            try:
                x = float(m)
                if abs(x) < 1e12:
                    ctx_nums.append(x)
            except ValueError:
                pass
        ctx_nums = sorted(set(ctx_nums))[:120]
        qas = [("qa", ex["qa"])] if "qa" in ex else [(k, ex[k]) for k in sorted(ex) if re.fullmatch(r"qa_\d+", k)]
        for tag, qa in qas:
            prog = qa.get("program", "")
            calls, operands, consts, depth, ok = parse_program(prog)
            exe = qa.get("exe_ans")
            if not isinstance(exe, (int, float)):        # 'greater' -> yes/no strings: exclude, count
                ok = False
            q = qa.get("question", "")
            out.append({
                "uid": f"{ex['id']}::{tag}",
                "qhash": hashlib.sha1(q.strip().lower().encode()).hexdigest()[:16],
                "question": q,
                "exe_ans": exe if isinstance(exe, (int, float)) else None,
                "answer_str": str(qa.get("answer", "")),
                "program": prog,
                "operands": sorted(operands),
                "consts": sorted(consts),
                "depth": depth,
                "percent_ctx": _percent_context(q, qa.get("answer", ""), prog),
                "ctx_numbers": ctx_nums,
                "ok": ok,
            })
    return out


def split_seed_scale(qas, n_seed=60, n_scale=300, rng_seed=13):
    """Deterministic disjoint split over VALID QAs, dedup by question hash (brief: dedupe; report overlap)."""
    import random
    valid = [q for q in qas if q["ok"] and q["exe_ans"] is not None]
    seen, uniq = set(), []
    dups = 0
    for q in valid:
        if q["qhash"] in seen:
            dups += 1; continue
        seen.add(q["qhash"]); uniq.append(q)
    rng = random.Random(rng_seed)
    rng.shuffle(uniq)
    seed, scale = uniq[:n_seed], uniq[n_seed:n_seed + n_scale]
    return seed, scale, {"total_qas": len(qas), "valid": len(valid), "dedup_dropped": dups,
                         "unique": len(uniq), "seed": len(seed), "scale": len(scale)}


def split_seed_rest(qas, n_seed=60, rng_seed=13):
    """Same deterministic seed as split_seed_scale, with all remaining unique QAs as the test split."""
    import random
    valid = [q for q in qas if q["ok"] and q["exe_ans"] is not None]
    seen, uniq = set(), []
    dups = 0
    for q in valid:
        if q["qhash"] in seen:
            dups += 1
            continue
        seen.add(q["qhash"])
        uniq.append(q)
    rng = random.Random(rng_seed)
    rng.shuffle(uniq)
    seed, rest = uniq[:n_seed], uniq[n_seed:]
    return seed, rest, {"total_qas": len(qas), "valid": len(valid), "dedup_dropped": dups,
                        "unique": len(uniq), "seed": len(seed), "rest": len(rest)}


if __name__ == "__main__":
    qas = load_qas()
    seed, scale, stats = split_seed_scale(qas)
    print("stats:", json.dumps(stats))
    ex = seed[0]
    print("example:", json.dumps({k: ex[k] for k in ("uid", "program", "operands", "consts", "depth", "exe_ans", "percent_ctx")}, ensure_ascii=False))
    from collections import Counter
    print("seed depth dist:", dict(Counter(q["depth"] for q in seed)))
    assert not (set(q["qhash"] for q in seed) & set(q["qhash"] for q in scale)), "seed/scale overlap!"
    print("seed/scale disjoint OK")
