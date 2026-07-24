#!/usr/bin/env python3
"""Run the moving-kinematic SLERP position-drive gate without a window."""

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


@dataclass(frozen=True)
class Fixture:
    rate: str
    dt: str
    frames: int
    initial: str


@dataclass(frozen=True)
class RunSpec:
    name: str
    fixture: Fixture
    solver: str
    execution: str
    expect_pass: bool | None


FIXTURES = tuple(
    Fixture(rate, dt, frames, initial)
    for rate, dt, frames in (
        ("30hz", "0.0333333351", 90),
        ("60hz", "0.0166666675", 180),
        ("120hz", "0.00833333377", 360),
    )
    for initial in ("identity", "driven-pos20")
)


def specs(mode: str) -> list[RunSpec]:
    result: list[RunSpec] = []
    for fixture in FIXTURES:
        result.append(
            RunSpec(
                f"{fixture.initial}-{fixture.rate}-tgs",
                fixture,
                "tgs",
                "parallel",
                True,
            )
        )
        if mode == "authority":
            continue
        for execution in ("parallel", "sequential"):
            expect_pass = True if mode == "acceptance" else None
            if mode == "failure-first":
                expect_pass = False
            result.append(
                RunSpec(
                    (
                        f"{fixture.initial}-{fixture.rate}-"
                        f"avbd-{execution}"
                    ),
                    fixture,
                    "avbd",
                    execution,
                    expect_pass,
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
    fixture = spec.fixture
    executable = bin_dir / EXECUTABLE
    argv = [
        str(executable),
        "--headless",
        f"--solver={spec.solver}",
        f"--execution={spec.execution}",
        "--case=angular-position",
        "--drive=slerp",
        "--actor-a=kinematic",
        "--kinematic-motion=spin-world-y",
        "--endpoint=forward",
        f"--initial-relative={fixture.initial}",
        "--frame-a=rotx-neg45",
        "--frame-b=identity",
        f"--dt={fixture.dt}",
        f"--frames={fixture.frames}",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(fixture.frames)
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

    stationary_frames = round(0.5 / float(fixture.dt))
    required = {
        "schema": "1",
        "snippet": "SnippetJointDrive",
        "case": "angular-position",
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(fixture.frames),
        "completedFrames": str(fixture.frames),
        "dt": fixture.dt,
        "seed": "1",
        "dispatcherThreads": "2",
        "capability": "PARTIAL",
        "validation": "GATED",
        "nonFinite": "0",
        "physicsErrors": "0",
        "fetchFailures": "0",
        "fetchErrorState": "0",
        "pairCount": "1",
        "drive": "slerp",
        "actorA": "kinematic",
        "topology": "static-dynamic",
        "frameA": "rotx-neg45",
        "frameB": "identity",
        "bodyBRotation": "identity",
        "initialRelative": fixture.initial,
        "driveMode": "force",
        "endpoint": "forward",
        "kinematicMotion": "spin-world-y",
        "actorOrderValid": "1",
        "angularFrameWitnessValid": "1",
        "massReadback": "1",
        "inertiaReadbackX": "1",
        "inertiaReadbackY": "1",
        "inertiaReadbackZ": "1",
        "stiffnessReadback": "100",
        "dampingReadback": "20",
        "forceLimitReadback": "5",
        "driveLimitsAreForces": "1",
        "targetReadbackError": "0",
        "kinematicTargetFrames": str(fixture.frames),
        "kinematicMotionFrames": str(
            fixture.frames - stationary_frames
        ),
    }
    for key, expected in required.items():
        actual = fields.get(key)
        if actual != expected:
            errors.append(f"{key}={actual!r}, expected {expected!r}")

    final_travel = finite_float(fields, "finalKinematicTravel", errors)
    if final_travel is not None and not math.isclose(
        final_travel, 0.625, rel_tol=0.0, abs_tol=1.0e-4
    ):
        errors.append(
            f"finalKinematicTravel={final_travel!r}, expected 0.625"
        )
    finite_float(fields, "maximumKinematicAngularSpeedError", errors)
    finite_float(fields, "finalTargetError", errors)
    finite_float(fields, "lateErrorRatio", errors)
    finite_float(fields, "lateSpeedRms", errors)

    if spec.expect_pass is True:
        if result.returncode != 0:
            errors.append(f"exit code {result.returncode}, expected 0")
        for key, expected in (
            ("status", "PASS"),
            ("reason", "none"),
            ("positionDriveGate", "PASS"),
            ("positionFiniteLimitGate", "PASS"),
            ("kinematicMotionGate", "PASS"),
        ):
            if fields.get(key) != expected:
                errors.append(
                    f"{key}={fields.get(key)!r}, expected {expected!r}"
                )
    elif spec.expect_pass is False:
        if result.returncode != 1:
            errors.append(f"exit code {result.returncode}, expected 1")
        for key, expected in (
            ("status", "FAIL"),
            ("reason", "kinematic_relative_target_not_tracked"),
            ("positionDriveGate", "FAIL"),
            ("positionFiniteLimitGate", "FAIL"),
            ("kinematicMotionGate", "FAIL"),
        ):
            if fields.get(key) != expected:
                errors.append(
                    f"{key}={fields.get(key)!r}, expected {expected!r}"
                )
    else:
        if result.returncode not in (0, 1):
            errors.append(
                f"probe exit code {result.returncode}, expected 0 or 1"
            )
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
        f"[MOVING_KINEMATIC_SLERP_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            f"[MOVING_KINEMATIC_SLERP_RUN_ERROR] "
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
            "[MOVING_KINEMATIC_SLERP_RUNNER_ERROR] "
            f"missing executable: {executable}"
        )
        return 2
    if args.timeout <= 0.0:
        print(
            "[MOVING_KINEMATIC_SLERP_RUNNER_ERROR] "
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
        f"[MOVING_KINEMATIC_SLERP_MATRIX] mode={args.mode} "
        f"completed={physics_passes + physics_failures} "
        f"total={len(run_specs)} physicsPasses={physics_passes} "
        f"physicsFailures={physics_failures} "
        f"status={'PASS' if all_passed else 'FAIL'}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
