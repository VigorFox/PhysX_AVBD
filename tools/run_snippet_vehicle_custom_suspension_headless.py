#!/usr/bin/env python3
"""Run SnippetVehicleCustomSuspension with an amplitude-zero negative control."""

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
DEFAULT_VEHICLE_DATA_DIR = (
    REPO_ROOT / "physx" / "snippets" / "media" / "vehicledata"
)
EXECUTABLE = "SnippetVehicleCustomSuspension_64.exe"
REQUESTED_FRAMES = 900
CASES = ("custom-dance", "zero-amplitude")


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    case_name: str
    repeat: int


def specs(mode: str) -> tuple[RunSpec, ...]:
    if mode == "authority":
        return tuple(
            RunSpec(f"tgs-parallel-{case}-r1", "tgs", "parallel", case, 1)
            for case in CASES
        )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    repeats = (1, 2) if mode == "acceptance" else (1,)
    return tuple(
        RunSpec(
            f"{solver}-{execution}-{case}-r{repeat}",
            solver,
            execution,
            case,
            repeat,
        )
        for repeat in repeats
        for solver, execution in lanes
        for case in CASES
    )


def parse_gate(line: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in line.split()[1:]:
        if "=" not in token:
            errors.append(f"malformed gate token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append(f"duplicate gate key: {key}")
        fields[key] = value
    return fields, errors


def validate_numeric_fields(
    fields: dict[str, str], errors: list[str]
) -> None:
    for key in (
        "customMaxMagnitude",
        "customAccumulatedMagnitude",
        "minJounce",
        "maxJounce",
        "maxJounceSpeed",
        "maxSuspensionForce",
        "maxTheta",
        "minHeight",
        "maxHeight",
        "heightSpan",
        "maxLinearSpeed",
        "maxAngularSpeed",
        "minUpY",
    ):
        try:
            if not math.isfinite(float(fields[key])):
                errors.append(f"{key} is non-finite")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected finite float")

    for key in (
        "completedFrames",
        "wheels",
        "constraints",
        "roadHitFrames",
        "roadHitSamples",
        "activeConstraintFrames",
        "activeConstraintRows",
        "sleepingFrames",
        "customCalls",
        "customOnGroundCalls",
        "customNonZeroForceCalls",
        "customNonFinite",
    ):
        try:
            if int(fields[key]) < 0:
                errors.append(f"{key} is negative")
        except (KeyError, ValueError):
            errors.append(f"{key}={fields.get(key)!r}, expected integer")


def run_one(
    spec: RunSpec,
    bin_dir: Path,
    vehicle_data_dir: Path,
    timeout: float,
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        f"--vehicleDataPath={vehicle_data_dir}",
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case_name}",
        f"--execution={spec.execution}",
        f"--frames={REQUESTED_FRAMES}",
        "--dt=0.016667",
        "--dispatcher-threads=4",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(REQUESTED_FRAMES)
    result = run_headless_process(
        argv, cwd=bin_dir, env=env, timeout_seconds=timeout
    )
    combined = result.stdout
    if result.stderr:
        combined += ("\n" if combined else "") + result.stderr
    lines = [
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
    if len(lines) != 1:
        errors.append(f"gate count is {len(lines)}, expected exactly 1")
    else:
        fields, parse_errors = parse_gate(lines[0])
        errors.extend(parse_errors)

    required = {
        "schema": "1",
        "snippet": "SnippetVehicleCustomSuspension",
        "solver": spec.solver,
        "case": spec.case_name,
        "execution": spec.execution,
        "frames": str(REQUESTED_FRAMES),
        "completedFrames": str(REQUESTED_FRAMES),
        "wheels": "4",
        "constraints": "1",
        "customNonFinite": "0",
        "nonFinite": "0",
        "suspensionNonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": "1",
        "pvd": "0",
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
    }
    for key, expected in required.items():
        if fields.get(key) != expected:
            errors.append(
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )
    validate_numeric_fields(fields, errors)
    try:
        if int(fields["customOnGroundCalls"]) <= 0:
            errors.append("custom component did not receive ground input")
        if int(fields["roadHitSamples"]) <= REQUESTED_FRAMES:
            errors.append("insufficient road-query coverage")
        sleeping = int(fields["sleepingFrames"])
        calls_per_frame = 4 * 3
        minimum_calls = (REQUESTED_FRAMES - sleeping) * calls_per_frame
        maximum_calls = minimum_calls + (calls_per_frame if sleeping else 0)
        custom_calls = int(fields["customCalls"])
        if (
            custom_calls < minimum_calls
            or custom_calls > maximum_calls
            or custom_calls % calls_per_frame
        ):
            errors.append(
                f"customCalls={fields['customCalls']}, "
                f"expected [{minimum_calls}, {maximum_calls}] "
                "from post-fetch sleep sampling"
            )
        nonzero = int(fields["customNonZeroForceCalls"])
        maximum = float(fields["customMaxMagnitude"])
        accumulated = float(fields["customAccumulatedMagnitude"])
        if spec.case_name == "custom-dance":
            if nonzero <= 0 or maximum <= 1000.0 or accumulated <= 0.0:
                errors.append("official custom force did not become active")
            if sleeping:
                errors.append("official custom force allowed actor sleep")
        elif nonzero or maximum >= 1e-5 or accumulated >= 1e-5:
            errors.append("zero-amplitude control produced custom force")
        elif sleeping <= 0:
            errors.append("zero-amplitude control did not reach sleep")
    except (KeyError, ValueError):
        pass
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")

    print(
        f"[VEHICLE_CUSTOM_SUSPENSION_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            "[VEHICLE_CUSTOM_SUSPENSION_RUN_ERROR] "
            f"name={spec.name} error={error}"
        )
    return not errors, fields


def compare_case_contrasts(
    results: dict[str, dict[str, str]], mode: str
) -> bool:
    passed = True
    lanes = (
        (("tgs", "parallel"),)
        if mode == "authority"
        else (
            ("tgs", "parallel"),
            ("avbd", "parallel"),
            ("avbd", "sequential"),
        )
    )
    for solver, execution in lanes:
        custom = results[f"{solver}-{execution}-custom-dance-r1"]
        control = results[f"{solver}-{execution}-zero-amplitude-r1"]
        try:
            height_delta = abs(
                float(custom["heightSpan"]) - float(control["heightSpan"])
            )
            angular_delta = abs(
                float(custom["maxAngularSpeed"])
                - float(control["maxAngularSpeed"])
            )
            contrast_ok = height_delta > 0.05 or angular_delta > 0.1
        except (KeyError, ValueError):
            height_delta = math.nan
            angular_delta = math.nan
            contrast_ok = False
        passed = passed and contrast_ok
        print(
            "[VEHICLE_CUSTOM_SUSPENSION_CONTRAST] "
            f"lane={solver}-{execution} "
            f"heightSpanDelta={height_delta:.9g} "
            f"angularSpeedDelta={angular_delta:.9g} "
            f"status={'PASS' if contrast_ok else 'FAIL'}"
        )
    return passed


def compare_repeats(results: dict[str, dict[str, str]]) -> bool:
    passed = True
    exact_keys = (
        "completedFrames",
        "roadHitFrames",
        "roadHitSamples",
        "activeConstraintFrames",
        "activeConstraintRows",
        "sleepingFrames",
        "customCalls",
        "customOnGroundCalls",
        "customNonZeroForceCalls",
    )
    numeric_keys = (
        "customMaxMagnitude",
        "customAccumulatedMagnitude",
        "minJounce",
        "maxJounce",
        "maxJounceSpeed",
        "maxSuspensionForce",
        "maxTheta",
        "minHeight",
        "maxHeight",
        "heightSpan",
        "maxLinearSpeed",
        "maxAngularSpeed",
        "minUpY",
    )
    for solver, execution in (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    ):
        for case in CASES:
            first = results[f"{solver}-{execution}-{case}-r1"]
            second = results[f"{solver}-{execution}-{case}-r2"]
            mismatches: list[str] = []
            for key in exact_keys:
                if first.get(key) != second.get(key):
                    mismatches.append(key)
            for key in numeric_keys:
                try:
                    if abs(float(first[key]) - float(second[key])) > 1e-4:
                        mismatches.append(key)
                except (KeyError, ValueError):
                    mismatches.append(key)
            pair_ok = not mismatches
            passed = passed and pair_ok
            print(
                "[VEHICLE_CUSTOM_SUSPENSION_REPEAT] "
                f"pair={solver}-{execution}-{case} "
                f"status={'PASS' if pair_ok else 'FAIL'} "
                f"mismatches={','.join(mismatches) if mismatches else 'none'}"
            )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "probe", "acceptance"),
        default="probe",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument(
        "--vehicle-data-dir",
        type=Path,
        default=DEFAULT_VEHICLE_DATA_DIR,
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    bin_dir = args.bin_dir.resolve()
    vehicle_data_dir = args.vehicle_data_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[VEHICLE_CUSTOM_SUSPENSION_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    for filename in ("Base.json", "DirectDrive.json"):
        if not (vehicle_data_dir / filename).is_file():
            print(
                "[VEHICLE_CUSTOM_SUSPENSION_RUNNER_ERROR] "
                f"missing vehicle data: {vehicle_data_dir / filename}"
            )
            return 2
    if args.timeout <= 0:
        print(
            "[VEHICLE_CUSTOM_SUSPENSION_RUNNER_ERROR] "
            "--timeout must be positive"
        )
        return 2

    infrastructure_ok = True
    results: dict[str, dict[str, str]] = {}
    for spec in specs(args.mode):
        passed, fields = run_one(
            spec, bin_dir, vehicle_data_dir, args.timeout
        )
        infrastructure_ok = infrastructure_ok and passed
        results[spec.name] = fields
    contrast_ok = (
        compare_case_contrasts(results, args.mode)
        if infrastructure_ok
        else False
    )
    repeat_ok = (
        compare_repeats(results)
        if infrastructure_ok
        and contrast_ok
        and args.mode == "acceptance"
        else True
    )
    accepted = infrastructure_ok and contrast_ok and repeat_ok
    print(
        f"[VEHICLE_CUSTOM_SUSPENSION_MATRIX] mode={args.mode} "
        f"infrastructure={'PASS' if infrastructure_ok else 'FAIL'} "
        f"contrast={'PASS' if contrast_ok else 'FAIL'} "
        f"repeatability={'PASS' if repeat_ok else 'FAIL'} "
        f"status={'PASS' if accepted else 'FAIL'}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
