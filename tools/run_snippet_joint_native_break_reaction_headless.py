#!/usr/bin/env python3
"""Gate native prismatic/revolute reaction and break semantics headlessly."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetJoint_64.exe"
FREQUENCY = 60
FRAMES = 600
WARMUP_FRAMES = 120
CASES = ("native-reaction", "native-no-break", "native-break")
JOINTS = ("prismatic", "revolute")
REACTION_RATIO_MINIMUM = 0.9
REACTION_RATIO_MAXIMUM = 1.1
REACTION_DIRECTION_MINIMUM = 0.99
REACTION_ORTHOGONAL_RATIO_MAXIMUM = 0.01
CONSTRAINED_ERROR_MAXIMUM = 1.0e-3
CONSTRAINED_SPEED_MAXIMUM = 1.0e-3
POST_BREAK_SPEED_MINIMUM = 1.0
LOW_BREAK_THRESHOLD = 50.0
HIGH_BREAK_THRESHOLD = 200.0


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    case: str
    joint: str


def make_specs(mode: str) -> tuple[RunSpec, ...]:
    if mode == "authority":
        lanes = (("tgs", "parallel"),)
    elif mode == "probe":
        lanes = (("avbd", "parallel"), ("avbd", "sequential"))
    else:
        lanes = (
            ("tgs", "parallel"),
            ("avbd", "parallel"),
            ("avbd", "sequential"),
        )
    return tuple(
        RunSpec(
            f"{solver}-{execution}-{case}-{joint}",
            solver,
            execution,
            case,
            joint,
        )
        for solver, execution in lanes
        for case in CASES
        for joint in JOINTS
    )


def parse_fields(
    line: str, prefix: str
) -> tuple[dict[str, str], list[str]]:
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


def parse_float(
    fields: dict[str, str], key: str, errors: list[str]
) -> float | None:
    try:
        value = float(fields[key])
    except (KeyError, ValueError):
        errors.append(f"{key}={fields.get(key)!r}, expected float")
        return None
    if not math.isfinite(value):
        errors.append(f"{key}={value}, expected finite")
        return None
    return value


def run_one(
    spec: RunSpec, bin_dir: Path, timeout: float, mode: str
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case}",
        f"--joint={spec.joint}",
        f"--execution={spec.execution}",
        f"--frames={FRAMES}",
        f"--dt={1.0 / FREQUENCY:.12g}",
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

    prefixes = {
        "gate": "[AVBD_GATE] ",
        "fixture": "[PROBE] [SnippetJointNativeBreakReaction] ",
        "cleanup": "[PROBE] [SnippetJointNativeBreakReactionCleanup] ",
    }
    parsed: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    if result.timed_out:
        errors.append("timed out")
    if result.visible_window_detected:
        errors.append(
            "visible window detected: "
            + ", ".join(result.visible_window_titles)
        )
    for name, prefix in prefixes.items():
        lines = [
            line.strip()
            for line in combined.splitlines()
            if line.startswith(prefix)
        ]
        if len(lines) != 1:
            errors.append(f"{name} line count is {len(lines)}, expected 1")
            parsed[name] = {}
        else:
            parsed[name], parse_errors = parse_fields(lines[0], prefix)
            errors.extend(parse_errors)

    gate = parsed["gate"]
    fixture = parsed["fixture"]
    cleanup = parsed["cleanup"]
    expected_status = "PASS" if mode != "probe" else gate.get("status")
    if expected_status not in {"PASS", "FAIL"}:
        errors.append(
            f"status={expected_status!r}, expected PASS or physical FAIL"
        )
        expected_status = "ERROR"

    gate_exact = {
        "schema": "1",
        "snippet": "SnippetJoint",
        "case": spec.case,
        "joint": spec.joint,
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(FRAMES),
        "completedFrames": str(FRAMES),
        "seed": "1",
        "dispatcherThreads": "2",
        "capability": "PARTIAL",
        "validation": "PROBE",
        "nonFinite": "0",
        "physicsErrors": "0",
        "fetchFailures": "0",
        "fetchErrorState": "0",
        "launchFailures": "0",
        "status": expected_status,
    }
    for key, expected in gate_exact.items():
        if gate.get(key) != expected:
            errors.append(
                f"gate {key}={gate.get(key)!r}, expected {expected!r}"
            )

    fixture_exact = {
        "case": spec.case,
        "joint": spec.joint,
        "loadKind": "angular" if spec.joint == "revolute" else "linear",
        "actorOrderValid": "1",
        "stateSamples": str(FRAMES),
        "forceReads": str(FRAMES),
        "nonFiniteSamples": "0",
        "initialDynamicActors": "1",
        "initialStaticActors": "0",
        "initialConstraints": "1",
        "finalDynamicActors": "1",
        "finalStaticActors": "0",
        "finalConstraints": "1",
    }
    for key, expected in fixture_exact.items():
        if fixture.get(key) != expected:
            errors.append(
                f"fixture {key}={fixture.get(key)!r}, expected {expected!r}"
            )
    cleanup_exact = {
        "dynamicActors": "0",
        "staticActors": "0",
        "constraints": "0",
        "cleanupComplete": "1",
    }
    for key, expected in cleanup_exact.items():
        if cleanup.get(key) != expected:
            errors.append(
                f"cleanup {key}={cleanup.get(key)!r}, "
                f"expected {expected!r}"
            )

    expected_exit = 0 if expected_status == "PASS" else 1
    if result.returncode != expected_exit:
        errors.append(
            f"exit code {result.returncode}, expected {expected_exit}"
        )
    if expected_status == "PASS" and gate.get("reason") != "none":
        errors.append(
            f"reason={gate.get('reason')!r}, expected 'none'"
        )
    if expected_status == "FAIL" and gate.get("reason") in {None, "none"}:
        errors.append("physical FAIL lacks a reason")
    values = {
        key: parse_float(fixture, key, errors)
        for key in (
            "reactionRatio",
            "reactionDirectionDot",
            "reactionOrthogonalRatio",
            "breakForceReadback",
            "breakTorqueReadback",
            "maximumPositionError",
            "maximumRotationError",
            "maximumLinearSpeed",
            "maximumAngularSpeed",
            "steadyMaximumPositionError",
            "steadyMaximumRotationError",
            "steadyMaximumLinearSpeed",
            "steadyMaximumAngularSpeed",
        )
    }
    reaction_samples = fixture.get("reactionSamples")
    broken_count = fixture.get("brokenPollCount")
    callback_count = fixture.get("breakCallbackCount")
    callback_identity = fixture.get("breakCallbackIdentityMatches")
    first_broken = fixture.get("firstBrokenFrame")

    semantics_pass = False
    if all(value is not None for value in values.values()):
        relevant_threshold = (
            values["breakTorqueReadback"]
            if spec.joint == "revolute"
            else values["breakForceReadback"]
        )
        irrelevant_threshold = (
            values["breakForceReadback"]
            if spec.joint == "revolute"
            else values["breakTorqueReadback"]
        )
        expected_threshold = (
            LOW_BREAK_THRESHOLD
            if spec.case == "native-break"
            else (
                HIGH_BREAK_THRESHOLD
                if spec.case == "native-no-break"
                else float.fromhex("0x1.fffffep+127")
            )
        )
        threshold_valid = (
            abs(relevant_threshold - expected_threshold)
            <= max(1.0e-5, abs(expected_threshold) * 1.0e-6)
            and irrelevant_threshold >= 3.0e38
        )
        if spec.case == "native-break":
            try:
                first_broken_value = int(first_broken or "-1")
            except ValueError:
                first_broken_value = -1
            post_break_speed = (
                values["maximumAngularSpeed"]
                if spec.joint == "revolute"
                else values["maximumLinearSpeed"]
            )
            semantics_pass = (
                threshold_valid
                and broken_count == "1"
                and callback_count == "1"
                and callback_identity == "1"
                and 1 <= first_broken_value <= WARMUP_FRAMES
                and post_break_speed >= POST_BREAK_SPEED_MINIMUM
            )
        else:
            relevant_speed = (
                values["steadyMaximumAngularSpeed"]
                if spec.joint == "revolute"
                else values["steadyMaximumLinearSpeed"]
            )
            semantics_pass = (
                threshold_valid
                and reaction_samples == str(FRAMES - WARMUP_FRAMES)
                and broken_count == "0"
                and callback_count == "0"
                and values["reactionRatio"] >= REACTION_RATIO_MINIMUM
                and values["reactionRatio"] <= REACTION_RATIO_MAXIMUM
                and values["reactionDirectionDot"]
                >= REACTION_DIRECTION_MINIMUM
                and values["reactionOrthogonalRatio"]
                <= REACTION_ORTHOGONAL_RATIO_MAXIMUM
                and values["steadyMaximumPositionError"]
                <= CONSTRAINED_ERROR_MAXIMUM
                and values["steadyMaximumRotationError"]
                <= CONSTRAINED_ERROR_MAXIMUM
                and relevant_speed <= CONSTRAINED_SPEED_MAXIMUM
            )
        if expected_status == "PASS" and not semantics_pass:
            errors.append("PASS status lacks native reaction/break semantics")
        if expected_status == "FAIL" and semantics_pass:
            errors.append("FAIL status lacks an independent red metric")

    print(
        f"[NATIVE_BREAK_REACTION_RUN] name={spec.name} "
        f"status={gate.get('status', 'MISSING')} "
        f"reason={gate.get('reason', 'MISSING')} "
        f"meanReaction={fixture.get('meanReaction', 'MISSING')} "
        f"ratio={fixture.get('reactionRatio', 'MISSING')} "
        f"direction={fixture.get('reactionDirectionDot', 'MISSING')} "
        f"broken={broken_count or 'MISSING'} "
        f"callback={callback_count or 'MISSING'} "
        f"firstBroken={first_broken or 'MISSING'} "
        f"linearSpeed={fixture.get('maximumLinearSpeed', 'MISSING')} "
        f"angularSpeed={fixture.get('maximumAngularSpeed', 'MISSING')} "
        f"steadyLinearSpeed="
        f"{fixture.get('steadyMaximumLinearSpeed', 'MISSING')} "
        f"steadyAngularSpeed="
        f"{fixture.get('steadyMaximumAngularSpeed', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    for error in errors:
        print(
            f"[NATIVE_BREAK_REACTION_ERROR] "
            f"name={spec.name} error={error}"
        )
    return not errors, gate


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run native SnippetJoint prismatic/revolute reaction and "
            "break gates without a window."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("authority", "probe", "acceptance"),
        default="probe",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        parser.error(f"executable not found: {executable}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    specs = make_specs(args.mode)
    passed_runs = 0
    status_counts = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    for spec in specs:
        passed, gate = run_one(spec, bin_dir, args.timeout, args.mode)
        passed_runs += int(passed)
        status = gate.get("status", "ERROR")
        status_counts[status if status in status_counts else "ERROR"] += 1

    passed = passed_runs == len(specs)
    print(
        f"[NATIVE_BREAK_REACTION_SUMMARY] mode={args.mode} "
        f"runs={passed_runs}/{len(specs)} "
        f"physicalPass={status_counts['PASS']} "
        f"physicalFail={status_counts['FAIL']} "
        f"error={status_counts['ERROR']} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
