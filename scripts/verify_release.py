"""Run recovered self-tests and frozen-record checks without external data or models."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = [
        ["-m", "experiments.sqlledger.runtime.canonicalize"],
        ["-m", "experiments.sqlledger.runtime.sqlite_reads"],
        ["-m", "experiments.sqlledger.runtime.restricted_replay"],
        ["-m", "experiments.sqlledger.audit.targeted_interventions"],
        ["-m", "kramabench_integration.provenance"],
        ["code/submission_assertions_rev2.py"],
        ["code/check_record_to_paper_consistency.py"],
        ["-m", "unittest", "discover", "-s", "tests/ipmc_rev2", "-v"],
        ["examples/audit_demo.py"],
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "code")])
    for args in checks:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                                text=True, capture_output=True, timeout=180)
        if result.returncode:
            print("FAIL:", " ".join(args), flush=True)
            print(result.stdout + result.stderr)
            raise SystemExit(result.returncode)
        print("PASS:", " ".join(args), flush=True)
    print(f"PASS: {len(checks)} check groups. Raw-data and model-serving experiments were not run.")


if __name__ == "__main__":
    main()
