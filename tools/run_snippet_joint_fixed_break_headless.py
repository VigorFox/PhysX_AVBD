#!/usr/bin/env python3
"""Regress SnippetJoint fixed-joint break lifecycle without a window."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetJoint_64.exe"
FRAMES = 600
CASES = ("fixed-no-break", "fixed-break")


@dataclass(frozen=True)
class RunSpec:
    solver: str
    execution: str
    case: str

    @property
    def name(self) -> str:
        return f"{self.solver}-{self.execution}-{self.case}"


def parse_fields(line: str, prefix: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in line[len(prefix) :].split():
        if "=" not in token:
            errors.append(f"malformed token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append(f"duplicate key: {key}")
        fields[key] = value
    return fields, errors


def run_one(
    spec: RunSpec, bin_dir: Path, timeout: float
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        f"--execution={spec.execution}",
        f"--case={spec.case}",
        f"--frames={FRAMES}",
        "--dt=0.0166666666667",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    result = run_headless_process(
        argv, cwd=bin_dir, env=env, timeout_seconds=timeout
    )
    combined = result.stdout
    if result.stderr:
        combined += ("\n" if combined else "") + result.stderr

    prefix = "[AVBD_GATE] "
    lines = [
        line.strip() for line in combined.splitlines()
        if line.startswith(prefix)
    ]
    errors: list[str] = []
    gate: dict[str, str] = {}
    if result.timed_out:
        errors.append("timed out")
    if result.visible_window_detected:
        errors.append(
            "visible window detected: "
            + ", ".join(result.visible_window_titles)
        )
    if len(lines) != 1:
        errors.append(f"gate line count is {len(lines)}, expected 1")
    else:
        gate, parse_errors = parse_fields(lines[0], prefix)
        errors.extend(parse_errors)

    expected_break = "1" if spec.case == "fixed-break" else "0"
    expected_first_break = (
        gate.get("firstBrokenFrame") if expected_break == "1"
        else "4294967295"
    )
    exact = {
        "schema": "1",
        "snippet": "SnippetJoint",
        "case": spec.case,
        "joint": "fixed",
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(FRAMES),
        "completedFrames": str(FRAMES),
        "seed": "1",
        "dispatcherThreads": "2",
        "capability": "SUPPORTED",
        "validation": "GATED",
        "status": "PASS",
        "reason": "none",
        "nonFinite": "0",
        "physicsErrors": "0",
        "fetchFailures": "0",
        "fetchErrorState": "0",
        "launchFailures": "0",
        "expectedHits": "1",
        "hitChains": "1",
        "responseProjectiles": "1",
        "brokenCount": expected_break,
        "breakCallbacks": expected_break,
        "breakCallbackIdentityMatches": expected_break,
        "breakCallbackConstraintMismatches": "0",
        "breakCallbackExternalReferenceMismatches": "0",
        "breakCallbackTypeMismatches": "0",
        "breakCallbackBrokenFlagMismatches": "0",
        "breakCallbackDuplicateMismatches": "0",
        "breakCallbackPollMismatches": "0",
        "firstBrokenFrame": expected_first_break,
    }
    for key, expected in exact.items():
        if gate.get(key) != expected:
            errors.append(
                f"{key}={gate.get(key)!r}, expected {expected!r}"
            )
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")

    if spec.case == "fixed-break":
        try:
            first_broken = int(gate.get("firstBrokenFrame", "-1"))
        except ValueError:
            first_broken = -1
        if first_broken < 1 or first_broken > FRAMES:
            errors.append(
                f"firstBrokenFrame={first_broken}, expected [1,{FRAMES}]"
            )

    print(
        f"[FIXED_BREAK_RUN] name={spec.name} "
        f"status={gate.get('status', 'MISSING')} "
        f"reason={gate.get('reason', 'MISSING')} "
        f"hits={gate.get('hitChains', 'MISSING')}/"
        f"{gate.get('expectedHits', 'MISSING')} "
        f"broken={gate.get('brokenCount', 'MISSING')} "
        f"callbacks={gate.get('breakCallbacks', 'MISSING')} "
        f"identity={gate.get('breakCallbackIdentityMatches', 'MISSING')} "
        f"firstBroken={gate.get('firstBrokenFrame', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(f"[FIXED_BREAK_ERROR] name={spec.name} error={error}")
    return not errors, gate


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-no-break/fixed-break SnippetJoint gates through "
            "the hidden headless process helper."
        )
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        parser.error(f"executable not found: {bin_dir / EXECUTABLE}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    specs = tuple(
        RunSpec(solver, execution, case)
        for solver, execution in (
            ("tgs", "parallel"),
            ("avbd", "parallel"),
            ("avbd", "sequential"),
        )
        for case in CASES
    )
    passed = 0
    for spec in specs:
        run_passed, _ = run_one(spec, bin_dir, args.timeout)
        passed += int(run_passed)
    ok = passed == len(specs)
    print(
        f"[FIXED_BREAK_SUMMARY] runs={passed}/{len(specs)} "
        f"status={'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
