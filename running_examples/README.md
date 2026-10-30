# Reproducible running examples

## Identity theft (`KramaBench legal-hard-1`)

The canonical, reviewable bundle is committed under `running_examples/identity_theft/`.
It contains the preserved episode, final program, declared/host/reference evidence,
host and replay outputs, audit diagnosis, manifests, paper-ready LaTeX, and a
machine-independent reproduction entrypoint.

Reproduce from a Git checkout:

```bash
cd running_examples/identity_theft
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

Expected terminal result: `"status": "PASS"`, with host value `10724.1429`,
restricted-replay value `null`, and benchmark-reference value `12964.8727`.

The reproduction script and its dependency list are self-contained in the example directory.
