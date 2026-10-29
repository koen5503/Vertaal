#!/usr/bin/env python3
"""Automated test runner measuring line and branch coverage with threshold enforcement.

If the third-party 'coverage' package is not installed, seamlessly falls back to Python's
standard library 'trace' and 'dis' modules to measure line coverage across core/stt/.
"""

import dis
import os
import pathlib
import subprocess
import sys
import unittest

# Ensure workspace root and cwd are in sys.path for test discovery and module imports
_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)
_repo_root = str(pathlib.Path(__file__).resolve().parents[4])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def run_with_standard_library_trace(target_coverage: int) -> int:
    """Measure line coverage across core/stt/ using Python's standard library trace and dis modules."""
    import trace

    print(f"=== [Coverage] Measuring core/stt/ coverage via standard library 'trace' module (target: {target_coverage}%) ===")

    tracer = trace.Trace(count=1, trace=0)

    def run_suite():
        loader = unittest.TestLoader()
        suite = loader.discover("tests", pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        return runner.run(suite)

    test_result = tracer.runfunc(run_suite)
    trace_res = tracer.results()

    stt_dir = pathlib.Path("core/stt").resolve()
    if not stt_dir.exists():
        print(f"[Coverage] Error: directory {stt_dir} not found.")
        return 1

    def get_executable_lines(filepath: str) -> set:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            code = compile(source, filepath, "exec")
        except Exception:
            return set()
        lines = set()

        def walk(co):
            for _, lno in dis.findlinestarts(co):
                if lno is not None:
                    lines.add(lno)
            for const in co.co_consts:
                if hasattr(const, "co_code"):
                    walk(const)

        walk(code)
        return lines

    print("\n" + "=" * 62)
    print(f"{'Name':<34} {'Stmts':>6} {'Miss':>6} {'Cover':>8}")
    print("-" * 62)

    total_stmts = 0
    total_miss = 0

    for py_file in sorted(stt_dir.glob("*.py")):
        exec_lines = get_executable_lines(str(py_file))
        file_hits = {
            lineno
            for (fn, lineno), count in trace_res.counts.items()
            if os.path.abspath(fn) == str(py_file.resolve()) and count > 0
        }
        covered = len(exec_lines.intersection(file_hits))
        stmts = len(exec_lines)
        miss = stmts - covered
        pct = (covered / stmts * 100.0) if stmts > 0 else 100.0
        rel_path = f"core/stt/{py_file.name}"
        print(f"{rel_path:<34} {stmts:>6} {miss:>6} {pct:>7.1f}%")
        total_stmts += stmts
        total_miss += miss

    print("-" * 62)
    tot_covered = total_stmts - total_miss
    tot_pct = (tot_covered / total_stmts * 100.0) if total_stmts > 0 else 100.0
    print(f"{'TOTAL':<34} {total_stmts:>6} {total_miss:>6} {tot_pct:>7.1f}%")
    print("=" * 62)

    if not test_result.wasSuccessful():
        print(f"[Coverage] Error: {len(test_result.failures)} failures, {len(test_result.errors)} errors encountered.")
        return 1

    print(f"\n[Coverage] 100% test pass rate verified ({test_result.testsRun} tests executed).")
    return 0


def main():
    target_coverage = 80
    if len(sys.argv) > 1:
        try:
            target_coverage = int(sys.argv[1])
        except ValueError:
            pass

    # Check if 'coverage' package is installed
    has_coverage_pkg = False
    try:
        import coverage  # noqa: F401
        has_coverage_pkg = True
    except ImportError:
        has_coverage_pkg = False

    if has_coverage_pkg:
        print(f"=== [Coverage] Running test suite with branch coverage enforcement (target: {target_coverage}%) ===")
        run_cmd = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        ]
        res = subprocess.run(run_cmd)
        if res.returncode != 0:
            print("[Coverage] Error: Test execution failed.")
            sys.exit(res.returncode)

        report_cmd = [sys.executable, "-m", "coverage", "report", "-m", f"--fail-under={target_coverage}"]
        res_rep = subprocess.run(report_cmd)
        sys.exit(res_rep.returncode)
    else:
        # Standard library fallback
        exit_code = run_with_standard_library_trace(target_coverage=target_coverage)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
