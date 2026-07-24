#!/usr/bin/env python3
"""Run the self-contained AVBD soft-body component tests without a window."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetSoftBodyAVBD_64.exe"
RESULT_RE = re.compile(
    r"^=== Results: (?P<passed>\d+) PASSED, (?P<failed>\d+) "
    r"FAILED \(out of (?P<total>\d+)\) ===$"
)
TEST_RE = re.compile(r"^--- Test (?P<test_id>\d+):")


@dataclass(frozen=True)
class RunResult:
    passed: int
    failed: int
    total: int
    test_ids: tuple[int, ...]


def run_one(
    name: str,
    bin_dir: Path,
    timeout: float,
    test_id: int | None,
) -> tuple[bool, RunResult | None]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--case={test_id if test_id is not None else 'all'}",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_AVBD_SOFTBODY_VISUAL"] = "0"
    env.pop("PHYSX_AVBD_SOFTBODY_ROT_TRACE", None)
    env.pop("PHYSX_AVBD_SOFTBODY_ROT_TRACE_INTERVAL", None)
    if test_id is None:
        env.pop("PHYSX_AVBD_SOFTBODY_TEST_ID", None)
    else:
        env["PHYSX_AVBD_SOFTBODY_TEST_ID"] = str(test_id)
    process = run_headless_process(
        argv, cwd=bin_dir, env=env, timeout_seconds=timeout
    )
    combined = process.stdout
    if process.stderr:
        combined += ("\n" if combined else "") + process.stderr
    errors: list[str] = []
    if process.timed_out:
        errors.append("timed out")
    if process.visible_window_detected:
        errors.append(
            "visible window detected: "
            + ", ".join(process.visible_window_titles)
        )
    result_matches = [
        RESULT_RE.match(line.strip())
        for line in combined.splitlines()
        if RESULT_RE.match(line.strip())
    ]
    parsed: RunResult | None = None
    if len(result_matches) != 1:
        errors.append(
            f"result summary count is {len(result_matches)}, expected 1"
        )
    else:
        match = result_matches[0]
        assert match is not None
        test_ids = tuple(
            int(test_match.group("test_id"))
            for line in combined.splitlines()
            if (test_match := TEST_RE.match(line.strip()))
        )
        parsed = RunResult(
            passed=int(match.group("passed")),
            failed=int(match.group("failed")),
            total=int(match.group("total")),
            test_ids=test_ids,
        )
        expected_ids = (
            tuple(range(1, 20)) if test_id is None else (test_id,)
        )
        if parsed.test_ids != expected_ids:
            errors.append(
                f"test ids={parsed.test_ids}, expected {expected_ids}"
            )
        if parsed.failed != 0:
            errors.append(f"{parsed.failed} component assertions failed")
        if parsed.passed != parsed.total or parsed.total <= 0:
            errors.append(
                f"results={parsed.passed}/{parsed.total}, expected all pass"
            )
    if any(line.lstrip().startswith("FAIL:") for line in combined.splitlines()):
        errors.append("FAIL assertion line present")
    if process.returncode != 0:
        errors.append(f"exit code {process.returncode}, expected 0")
    print(
        "[SOFT_BODY_AVBD_RUN] "
        f"name={name} case={test_id if test_id is not None else 'all'} "
        f"exit={process.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[SOFT_BODY_AVBD_RUN_ERROR] name={name} error={error}")
    return not errors, parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("probe", "acceptance"), default="probe"
    )
    parser.add_argument("--test-id", type=int, choices=range(1, 20))
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(
            f"[SOFT_BODY_AVBD_RUNNER_ERROR] missing executable: {executable}"
        )
        return 2
    repeats = 2 if args.mode == "acceptance" else 1
    results: list[RunResult] = []
    passed = True
    for repeat in range(1, repeats + 1):
        run_passed, result = run_one(
            f"component-r{repeat}",
            bin_dir,
            args.timeout,
            args.test_id,
        )
        passed = passed and run_passed
        if result is not None:
            results.append(result)
    if len(results) == 2:
        repeat_ok = results[0] == results[1]
        passed = passed and repeat_ok
        print(
            "[SOFT_BODY_AVBD_REPEAT] "
            f"status={'PASS' if repeat_ok else 'FAIL'}"
        )
    print(
        "[SOFT_BODY_AVBD_SUMMARY] "
        f"mode={args.mode} runs={repeats} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
