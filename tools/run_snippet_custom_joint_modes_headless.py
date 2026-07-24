#!/usr/bin/env python3
"""Run the SnippetCustomJoint public generic-row mode pack without UI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetCustomJoint_64.exe"


@dataclass(frozen=True)
class RunSpec:
    name: str
    case_name: str
    solver: str
    execution: str
    expected_status: str
    expected_reason: str


FAILURE_REASONS = {
    "multi-output": "multi_row_not_consumed",
    "spring": "spring_row_not_consumed",
    "restitution": "restitution_row_not_consumed",
    "drive-limit": "drive_limit_row_not_consumed",
}


def specs(mode: str) -> list[RunSpec]:
    result: list[RunSpec] = []
    cases = tuple(FAILURE_REASONS)
    for case_name in cases:
        result.append(
            RunSpec(
                f"{case_name}-tgs-parallel",
                case_name,
                "tgs",
                "parallel",
                "PASS",
                "none",
            )
        )
        for execution in ("parallel", "sequential"):
            expect_failure = mode == "failure-first"
            result.append(
                RunSpec(
                    f"{case_name}-avbd-{execution}",
                    case_name,
                    "avbd",
                    execution,
                    "FAIL" if expect_failure else "PASS",
                    FAILURE_REASONS[case_name]
                    if expect_failure
                    else "none",
                )
            )
    return result


def parse_authority(line: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in line.split()[1:]:
        if "=" not in token:
            errors.append(f"malformed authority token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append(f"duplicate authority key: {key}")
        fields[key] = value
    return fields, errors


def run_one(
    spec: RunSpec, bin_dir: Path, timeout_seconds: float
) -> tuple[bool, dict[str, str]]:
    executable = bin_dir / EXECUTABLE
    argv = [
        str(executable),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case_name}",
        f"--execution={spec.execution}",
        "--frames=180",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
        "--impulse-frame=20",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    result = run_headless_process(
        argv, cwd=bin_dir, env=env, timeout_seconds=timeout_seconds
    )

    combined = result.stdout
    if result.stderr:
        combined += ("\n" if combined else "") + result.stderr
    authority_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_GATE] ")
    ]
    errors: list[str] = []
    fields: dict[str, str] = {}
    if result.timed_out:
        errors.append("timed out")
    if result.visible_window_detected:
        errors.append(
            "visible window detected: "
            + ", ".join(result.visible_window_titles)
        )
    if len(authority_lines) != 1:
        errors.append(
            f"authority count is {len(authority_lines)}, expected exactly 1"
        )
    else:
        fields, parse_errors = parse_authority(authority_lines[0])
        errors.extend(parse_errors)

    expected = {
        "schema": "1",
        "snippet": "SnippetCustomJoint",
        "case": spec.case_name,
        "solver": spec.solver,
        "execution": spec.execution,
        "frames": "180",
        "completedFrames": "180",
        "status": spec.expected_status,
        "reason": spec.expected_reason,
        "forceReads": "180",
        "nonFiniteForceReads": "0",
        "nonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
    }
    for key, value in expected.items():
        actual = fields.get(key)
        if actual != value:
            errors.append(f"{key}={actual!r}, expected {value!r}")

    expected_exit = 0 if spec.expected_status == "PASS" else 1
    if result.returncode != expected_exit:
        errors.append(
            f"exit code {result.returncode}, expected {expected_exit}"
        )
    if result.stderr:
        errors.append("stderr is not empty")

    print(
        f"[CUSTOM_MODE_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[CUSTOM_MODE_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "failure-first", "acceptance"),
        default="failure-first",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(f"[CUSTOM_MODE_RUNNER_ERROR] missing executable: {executable}")
        return 2
    if args.timeout <= 0.0:
        print("[CUSTOM_MODE_RUNNER_ERROR] --timeout must be positive")
        return 2

    selected = [
        spec
        for spec in specs(args.mode)
        if args.mode != "authority" or spec.solver == "tgs"
    ]
    all_passed = True
    physical_passes = 0
    physical_failures = 0
    for spec in selected:
        passed, fields = run_one(spec, bin_dir, args.timeout)
        all_passed = all_passed and passed
        physical_passes += fields.get("status") == "PASS"
        physical_failures += fields.get("status") == "FAIL"

    print(
        f"[CUSTOM_MODE_MATRIX] mode={args.mode} runs={len(selected)} "
        f"physicsPasses={physical_passes} "
        f"physicsFailures={physical_failures} "
        f"status={'PASS' if all_passed else 'FAIL'}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
