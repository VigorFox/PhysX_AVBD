#!/usr/bin/env python3
"""Run the legacy D6 cone-limit authority without a visible window."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetJointDrive_64.exe"
DT = "0.0166666675"
FRAMES = 180


@dataclass(frozen=True)
class RunSpec:
    fixture: str
    solver: str
    execution: str
    expect_pass: bool | None

    @property
    def name(self) -> str:
        return f"{self.fixture}-{self.solver}-{self.execution}"


def specs(mode: str) -> list[RunSpec]:
    result: list[RunSpec] = []
    for fixture in ("inside", "outside"):
        result.append(RunSpec(fixture, "tgs", "parallel", True))
        if mode == "authority":
            continue
        for execution in ("parallel", "sequential"):
            expected: bool | None
            if mode == "acceptance":
                expected = True
            elif mode == "failure-first":
                expected = fixture == "inside"
            else:
                expected = None
            result.append(
                RunSpec(fixture, "avbd", execution, expected)
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


def finite_float(
    fields: dict[str, str], key: str, errors: list[str]
) -> float | None:
    raw = fields.get(key)
    try:
        value = float(raw) if raw is not None else math.nan
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        errors.append(f"{key}={raw!r}, expected finite numeric value")
        return None
    return value


def run_one(
    spec: RunSpec, bin_dir: Path, timeout_seconds: float
) -> tuple[bool, dict[str, str]]:
    case = f"legacy-angular-limit-cone-{spec.fixture}"
    executable = bin_dir / EXECUTABLE
    argv = [
        str(executable),
        "--headless",
        f"--solver={spec.solver}",
        f"--execution={spec.execution}",
        f"--case={case}",
        f"--dt={DT}",
        f"--frames={FRAMES}",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(FRAMES)
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
    fields: dict[str, str] = {}
    errors: list[str] = []
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
        "snippet": "SnippetJointDrive",
        "case": case,
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(FRAMES),
        "completedFrames": str(FRAMES),
        "dt": DT,
        "seed": "1",
        "dispatcherThreads": "2",
        "capability": "PARTIAL",
        "validation": "GATED",
        "nonFinite": "0",
        "physicsErrors": "0",
        "fetchFailures": "0",
        "fetchErrorState": "0",
        "pairCount": "1",
        "limitKind": "legacy-cone",
        "fixture": spec.fixture,
        "stimulusWitness": "1",
        "twistMotionReadback": "0",
        "swing1MotionReadback": "1",
        "swing2MotionReadback": "1",
        "sampleCount": str(FRAMES),
        "lateSampleCount": "60",
        "insideControlGate": (
            "NOT_APPLICABLE"
            if spec.fixture == "outside"
            else fields.get("status", "MISSING")
        ),
    }
    for key, expected in required.items():
        actual = fields.get(key)
        if actual != expected:
            errors.append(f"{key}={actual!r}, expected {expected!r}")

    for key, expected, tolerance in (
        ("limitY", math.pi / 9.0, 1.0e-6),
        ("limitZ", 7.0 * math.pi / 36.0, 1.0e-6),
    ):
        value = finite_float(fields, key, errors)
        if value is not None and not math.isclose(
            value, expected, rel_tol=0.0, abs_tol=tolerance
        ):
            errors.append(f"{key}={value!r}, expected {expected!r}")
    for key in (
        "initialConeAngle",
        "finalConeAngle",
        "minimumConeAngle",
        "maximumConeAngle",
        "maximumLateConeAngle",
        "maximumInsideDeviation",
        "finalEllipseRadius",
        "maximumLateEllipseRadius",
        "maximumInsideEllipseDeviation",
        "correction",
        "maxQuaternionNormError",
        "maxAbsPosition",
        "maxLinearSpeed",
        "maxAngularSpeed",
    ):
        finite_float(fields, key, errors)
    expected_initial_radius = math.sqrt(
        (
            (
                math.pi / 18.0
                if spec.fixture == "inside"
                else math.pi / 10.0
            )
            / (math.pi / 9.0)
        )
        ** 2
        + (
            (
                math.pi / 12.0
                if spec.fixture == "inside"
                else math.pi / 6.0
            )
            / (7.0 * math.pi / 36.0)
        )
        ** 2
    )
    initial_radius = finite_float(
        fields, "initialEllipseRadius", errors
    )
    if initial_radius is not None and not math.isclose(
        initial_radius,
        expected_initial_radius,
        rel_tol=0.0,
        abs_tol=1.0e-5,
    ):
        errors.append(
            "initialEllipseRadius="
            f"{initial_radius!r}, expected {expected_initial_radius!r}"
        )

    if spec.expect_pass is True:
        if result.returncode != 0:
            errors.append(f"exit code {result.returncode}, expected 0")
        for key in ("status", "legacyAngularLimitGate"):
            if fields.get(key) != "PASS":
                errors.append(
                    f"{key}={fields.get(key)!r}, expected 'PASS'"
                )
        if fields.get("reason") != "none":
            errors.append(
                f"reason={fields.get('reason')!r}, expected 'none'"
            )
    elif spec.expect_pass is False:
        if result.returncode != 1:
            errors.append(f"exit code {result.returncode}, expected 1")
        for key in ("status", "legacyAngularLimitGate"):
            if fields.get(key) != "FAIL":
                errors.append(
                    f"{key}={fields.get(key)!r}, expected 'FAIL'"
                )
        if fields.get("reason") != "legacy_cone_limit_not_enforced":
            errors.append(
                "reason="
                f"{fields.get('reason')!r}, expected "
                "'legacy_cone_limit_not_enforced'"
            )
    else:
        if fields.get("status") not in ("PASS", "FAIL"):
            errors.append(
                f"status={fields.get('status')!r}, expected PASS or FAIL"
            )
        expected_exit = 0 if fields.get("status") == "PASS" else 1
        if result.returncode != expected_exit:
            errors.append(
                f"exit code {result.returncode}, expected {expected_exit}"
            )

    print(
        f"[LEGACY_ANGULAR_LIMIT_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"initial={fields.get('initialConeAngle', 'MISSING')} "
        f"final={fields.get('finalConeAngle', 'MISSING')} "
        f"radius={fields.get('finalEllipseRadius', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            f"[LEGACY_ANGULAR_LIMIT_RUN_ERROR] "
            f"name={spec.name} error={error}"
        )
    return not errors, fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "failure-first", "probe", "acceptance"),
        default="authority",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(
            "[LEGACY_ANGULAR_LIMIT_RUNNER_ERROR] "
            f"missing executable: {executable}"
        )
        return 2
    if args.timeout <= 0.0:
        print(
            "[LEGACY_ANGULAR_LIMIT_RUNNER_ERROR] "
            "--timeout must be positive"
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
        f"[LEGACY_ANGULAR_LIMIT_MATRIX] mode={args.mode} "
        f"completed={physics_passes + physics_failures} "
        f"total={len(run_specs)} physicsPasses={physics_passes} "
        f"physicsFailures={physics_failures} "
        f"status={'PASS' if all_passed else 'FAIL'}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
