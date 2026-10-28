#!/usr/bin/env python3
"""Automated test runner measuring line and branch coverage with threshold enforcement."""

import subprocess
import sys


def main():
    target_coverage = 80
    if len(sys.argv) > 1:
        try:
            target_coverage = int(sys.argv[1])
        except ValueError:
            pass

    print(f"=== [Coverage] Running test suite with branch coverage enforcement (target: {target_coverage}%) ===")

    # 1. Run unittest via coverage
    run_cmd = [sys.executable, "-m", "coverage", "run", "--branch", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    res = subprocess.run(run_cmd)
    if res.returncode != 0:
        print("[Coverage] Error: Test execution failed.")
        sys.exit(res.returncode)

    # 2. Generate report
    report_cmd = [sys.executable, "-m", "coverage", "report", "-m", f"--fail-under={target_coverage}"]
    res_rep = subprocess.run(report_cmd)
    sys.exit(res_rep.returncode)


if __name__ == "__main__":
    main()
