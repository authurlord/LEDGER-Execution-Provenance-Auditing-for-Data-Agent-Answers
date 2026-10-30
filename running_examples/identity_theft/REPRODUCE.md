# Reproduce on another machine

Python 3.11 is recommended. From this directory:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

The command verifies every file listed in `SHA256SUMS`, reruns the exact final
program against the packaged host-read view, reruns it against the restricted
declared view, and independently recomputes the benchmark reference answer.
It requires neither the original smart01 absolute paths nor a full KramaBench
checkout. A successful run prints `"status": "PASS"`.
