# KramaBench identity-theft running example (`legal-hard-1`)

This bundle reconstructs the complete causal chain from preserved artifacts:

`declared evidence -> final program -> host output -> stated answer -> restricted replay -> audit diagnosis`.

## Result

- Requested quantity: **mean report count**, not mean reporting rate, for metropolitan areas with inferred/interpolated 2023 population above one million.
- Declared evidence: 17 files (one national identity table plus 16 state identity tables).
- Actual host reads: 17 files (the national identity table plus 16 fraud-and-other tables).
- Declared/host-read intersection: 1 file; 16 host reads were undeclared and 16 declared state files were unread.
- Final host output: `10724.1429`; stated answer: `10724.1429`. They are equal.
- Earlier manuscript rendering `10,724.14` is a two-decimal abbreviation; the raw episode and four-decimal task answer are `10,724.1429`.
- Benchmark official answer: `12964.8727`; the bundled reference evidence and recomputation reproduce it.
- Restricted replay: exit code `0`, empty stderr, JSON output `{"main-task": null}`.
- Safe diagnosis: `missing_comparable_output`, plus `undeclared_host_reads`; no replay certificate.

The historical checker reported unsupported class `a` with `recomputed='none'`.  That record is preserved verbatim in `raw/ledger_record.json`.  `audit_record.json` explains why JSON null is better represented as a missing/non-comparable output than as an ordinary unequal value.

## Provenance note

The original episode retained the four programs, their stdout/stderr, prompt, response, token counts, and final answer, but not syscall-level read traces.  `host_reads.jsonl` is therefore explicitly labeled as a reconstruction from the exact final program, verified by a fresh `strace` run.  The filtered verification traces are under `raw/`.

## Paper text

`paper_context.tex` is a compact Introduction-ready paragraph.  It deliberately says “mean number of reports,” uses the raw four-decimal stated answer, and names the replay outcome precisely.

No second semantic-error episode is included: the repository's G2 human-support columns are still empty, so there is no sufficiently complete human-validated semantic-error trace to package without inventing evidence.
