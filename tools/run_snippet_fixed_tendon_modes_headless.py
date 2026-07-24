#!/usr/bin/env python3
"""Run branch, linear-axis and limit FixedTendon modes without UI."""

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
EXECUTABLE = "SnippetFixedTendon_64.exe"


@dataclass(frozen=True)
class Fixture:
    tendon_mode: str
    topology: str
    axis: str
    offset: str
    stiffness: str
    damping: str
    limit_stiffness: str
    low_limit: str
    high_limit: str


@dataclass(frozen=True)
class RunSpec:
    name: str
    fixture: Fixture
    solver: str
    execution: str
    case_name: str
    expect_pass: bool


FIXTURES = (
    Fixture(
        "branch-angular",
        "branch",
        "swing2",
        "0.200000003",
        "5000",
        "100",
        "0",
        "3.40282347e+38",
        "-3.40282347e+38",
    ),
    Fixture(
        "branch-linear",
        "branch",
        "x",
        "0.200000003",
        "5000",
        "100",
        "0",
        "3.40282347e+38",
        "-3.40282347e+38",
    ),
    Fixture(
        "branch-limit-angular",
        "branch",
        "swing2",
        "0",
        "0",
        "0",
        "5000",
        "-0.200000003",
        "0.200000003",
    ),
)


def specs(mode: str) -> list[RunSpec]:
    result: list[RunSpec] = []
    for fixture in FIXTURES:
        for case_name in ("drive-a", "drive-b"):
            result.append(
                RunSpec(
                    f"{fixture.tendon_mode}-{case_name}-tgs",
                    fixture,
                    "tgs",
                    "parallel",
                    case_name,
                    True,
                )
            )
            if mode == "authority":
                continue
            for execution in ("parallel", "sequential"):
                result.append(
                    RunSpec(
                        (
                            f"{fixture.tendon_mode}-{case_name}-"
                            f"avbd-{execution}"
                        ),
                        fixture,
                        "avbd",
                        execution,
                        case_name,
                        mode == "acceptance",
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
        f"--tendon-mode={spec.fixture.tendon_mode}",
        "--frames=480",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = "480"
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

    required = {
        "schema": "1",
        "snippet": "SnippetFixedTendon",
        "tendonMode": spec.fixture.tendon_mode,
        "topology": spec.fixture.topology,
        "axis": spec.fixture.axis,
        "tendonJointCount": "3",
        "solver": spec.solver,
        "case": spec.case_name,
        "execution": spec.execution,
        "frames": "480",
        "completedFrames": "480",
        "coefficientA": "1",
        "reciprocalA": "1",
        "coefficientB": "-1",
        "reciprocalB": "-1",
        "offsetBase": spec.fixture.offset,
        "stiffness": spec.fixture.stiffness,
        "damping": spec.fixture.damping,
        "limitStiffness": spec.fixture.limit_stiffness,
        "lowLimit": spec.fixture.low_limit,
        "highLimit": spec.fixture.high_limit,
        "responseSamples": "420",
        "nonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": "1",
        "pvd": "0",
        "validation": "GATED",
    }
    for key, expected in required.items():
        actual = fields.get(key)
        if actual != expected:
            errors.append(f"{key}={actual!r}, expected {expected!r}")

    if spec.expect_pass:
        if result.returncode != 0:
            errors.append(f"exit code {result.returncode}, expected 0")
        if fields.get("status") != "PASS":
            errors.append(
                f"status={fields.get('status')!r}, expected 'PASS'"
            )
        if fields.get("reason") != "none":
            errors.append(
                f"reason={fields.get('reason')!r}, expected 'none'"
            )
    else:
        if result.returncode != 1:
            errors.append(f"exit code {result.returncode}, expected 1")
        if fields.get("status") != "FAIL":
            errors.append(
                f"status={fields.get('status')!r}, expected 'FAIL'"
            )
        if fields.get("reason") not in (
            "missing_coupled_response",
            "tendon_limit_error",
            "wrong_coupling_direction",
        ):
            errors.append(
                f"unexpected failure reason: {fields.get('reason')!r}"
            )

    print(
        f"[FIXED_TENDON_MODE_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            f"[FIXED_TENDON_MODE_RUN_ERROR] "
            f"name={spec.name} error={error}"
        )
    return not errors, fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "failure-first", "acceptance"),
        default="authority",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(
            f"[FIXED_TENDON_MODE_RUNNER_ERROR] "
            f"missing executable: {executable}"
        )
        return 2
    if args.timeout <= 0.0:
        print(
            "[FIXED_TENDON_MODE_RUNNER_ERROR] --timeout must be positive"
        )
        return 2

    all_passed = True
    physics_passes = 0
    physics_failures = 0
    run_specs = specs(args.mode)
    for spec in run_specs:
        passed, fields = run_one(spec, bin_dir, args.timeout)
        all_passed = all_passed and passed
        physics_passes += fields.get("status") == "PASS"
        physics_failures += fields.get("status") == "FAIL"
        if not passed:
            break

    print(
        f"[FIXED_TENDON_MODE_MATRIX] mode={args.mode} "
        f"completed={physics_passes + physics_failures} "
        f"total={len(run_specs)} physicsPasses={physics_passes} "
        f"physicsFailures={physics_failures} "
        f"status={'PASS' if all_passed else 'FAIL'}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
