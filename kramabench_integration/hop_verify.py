"""M1 — grounded bridge verification: DETERMINISTIC tool routines (NOT an LLM judging itself).
The model proposes a bridge value + WHAT to check; these routines compute pass/fail FROM THE DATA.
NON-VETO: a failed check only marks a bridge "suspect" and may trigger ONE re-attempt — it NEVER
overwrites a produced answer (the original always stays in the candidate pool).

SAFE checks (low false-positive, default ON): assert_schema, surface_boundary, source_contains.
HIGH-FP checks (default OFF, enable only if ablation shows net+ with <=1 loss/80): check_dedup, cross_source_agree.

All paths resolved with lakeqa_agent.resolve() (datagov dirs -> files/rows.txt).
"""
import os, re, duckdb

def _q(p): return p.replace("'", "''")   # duckdb single-quote escape

def _read_head(con, path, n=2000):
    return con.execute(f"SELECT * FROM read_csv_auto('{_q(path)}', ignore_errors=true, sample_size={n}) LIMIT 200").fetchall(), con.description

# codex M1 fix: controlled slot ontology with ALIASES (correct tables use codes like SCH_TYPE/CBSA/SURVYEAR,
# not literal prose). Each slot = a list of alias substrings; a slot is satisfied if any alias hits col-names+sample.
SLOT_ALIASES = {
  "geo_level":      ["county","cnty","cbsa","metropolitan","msa","district","lea","state","statefp","zip","tract","place"],
  "year":           ["year","survyear","fy","fiscal","sy20","20"],   # '20' catches 2019/2020/2021/2022 in values
  "school_type":    ["private","public","charter","sch_type","schtype","level","institution","postsecondary","elementary","secondary"],
  "measure_count":  ["count","number","total","enrollment","students","schools","offices","n_","num"],
  "money":          ["amount","expenditure","spend","cost","dollar","usd","$","pay","salary","revenue"],
  "person_count":   ["recipient","veteran","population","persons","people"],
}
def assert_schema(path, slots):
    """SAFE (codex-revised): does `path` parse as a TABLE and cover the REQUIRED structural SLOTS for the hop?
    slots = {"required":[slot,...], "optional":[slot,...]} using SLOT_ALIASES keys (geo_level/year/school_type/...).
    A slot is satisfied if any of its aliases appears in column-names+sampled-values. PASS iff ALL required slots hit.
    Catches opened-metadata / near-vocabulary-but-wrong-structure (e.g. school-OFFICES vs school-COUNTS).
    Back-compat: if `slots` is a plain list, treat each as a literal alias group (old behavior, half-threshold).
    """
    if not os.path.exists(path): return False, "file not found"
    try:
        con=duckdb.connect()
        rows, desc = _read_head(con, path)
        if desc is None or len(desc) < 2 or len(rows) < 1:
            return False, "not a parseable table (>=2 cols, >=1 row)"
        cols=[d[0].lower() for d in desc]
        sample=" ".join(str(r[i]) for r in rows[:50] for i in range(len(desc))).lower()
        hay=" ".join(cols)+" "+sample
        def slot_hit(slot):
            for a in SLOT_ALIASES.get(slot,[slot]):
                if a in hay: return a
            return None
        if isinstance(slots, dict):
            req=slots.get("required",[]); opt=slots.get("optional",[])
            req_hits={s:slot_hit(s) for s in req}; opt_hits={s:slot_hit(s) for s in opt}
            ok=all(v is not None for v in req_hits.values())
            return ok, f"cols={cols[:8]} required={req_hits} optional={opt_hits}"
        else:   # back-compat literal list
            hit=[t for t in slots if t and t.lower() in hay]
            return len(hit)>=max(1,len(slots)//2), f"cols={cols[:8]} matched={hit}/{slots}"
    except Exception as e:
        return False, "parse error: "+str(e)[:80]

def surface_boundary(path, column_hint, near=None):
    """SAFE: return the ACTUAL distinct values + counts of the column matching column_hint, so the
    model can confirm a filter (year/unit/>=-vs->) against real values instead of guessing.
    Returns (ok, detail-with-real-values). Never fails an answer; it surfaces evidence.
    """
    if not os.path.exists(path): return False, "file not found"
    try:
        con=duckdb.connect()
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_csv_auto('{_q(path)}', ignore_errors=true, sample_size=200000)")
        cols=[r[0] for r in con.execute("DESCRIBE t").fetchall()]
        col=next((c for c in cols if column_hint.lower() in c.lower()), None)
        if not col: return False, f"no column ~ '{column_hint}'; columns={cols[:10]}"
        vals=con.execute(f'SELECT "{col}" v, COUNT(*) c FROM t GROUP BY "{col}" ORDER BY c DESC LIMIT 20').fetchall()
        return True, f'column "{col}" distinct values (value:count): ' + ", ".join(f"{v}:{c}" for v,c in vals)
    except Exception as e:
        return False, "error: "+str(e)[:80]

# codex M1 fix: synonym expansion so paraphrased properties still match (record high==highest temperature, etc.)
PROP_SYNONYMS = {
  "record high":["record high","highest temperature","maximum temperature","hottest","all-time high"],
  "statue":["statue","monument","landmark","sculpture","roadside attraction","largest"],
  "buried":["buried","burial","grave","cemetery","interred","tomb"],
  "county seat":["county seat","seat of","administrative center"],
}
def _expand(terms):
    out=set()
    for t in terms:
        tl=str(t).lower(); out.add(tl)
        for k,syn in PROP_SYNONYMS.items():
            if tl in k or k in tl or tl in syn: out.update(syn)
    return [t for t in out if t]
def source_contains(path, entity, property_terms, window=600):
    """SAFE (entities, codex-revised): entity must co-occur with a discriminating property within `window`
    chars (~paragraph) of its source text; property_terms are SYNONYM-EXPANDED. Catches 'plausible wrong entity'.
    Aliases (New York County==Manhattan) are NOT deterministically checkable -> caller should abstain, not force.
    """
    if not os.path.exists(path): return False, "file not found"
    try:
        txt=open(path, errors="ignore").read()
    except Exception as e:
        return False, "read error: "+str(e)[:60]
    low=txt.lower(); ent=str(entity).lower().strip()
    if not ent: return False, "empty entity"
    pos=low.find(ent)
    if pos < 0: return False, f"entity '{entity}' not found in source"
    seg=low[max(0,pos-window):pos+len(ent)+window]
    pterms=_expand(property_terms)
    hit=[p for p in pterms if p in seg]
    ok=len(hit) >= 1 if property_terms else True
    snip=txt[max(0,pos-60):pos+len(ent)+60].replace("\n"," ")
    return ok, f"entity@{pos} property_hits={hit[:4]} (expanded from {property_terms}) :: …{snip}…"

# ---- HIGH-FALSE-POSITIVE checks (DEFAULT OFF; enable only if ablation shows net+ with <=1 loss/80) ----
def check_dedup(path, key_cols, enabled=False):
    if not enabled: return None, "dedup check OFF (high-FP, default off)"
    try:
        con=duckdb.connect(); con.execute(f"CREATE VIEW t AS SELECT * FROM read_csv_auto('{_q(path)}', ignore_errors=true, sample_size=200000)")
        tot=con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        keys=",".join(f'"{c}"' for c in key_cols)
        dis=con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT {keys} FROM t)").fetchone()[0]
        return (tot==dis), f"rows={tot} distinct_keys={dis} (dup={tot-dis})"
    except Exception as e:
        return None, "error: "+str(e)[:60]
