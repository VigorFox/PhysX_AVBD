#!/usr/bin/env python3
"""Run the SnippetCustomJoint failure-first or acceptance matrix safely."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

from avbd_joint_objective_ir_gate import validate_joint_objective_ir
from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetCustomJoint_64.exe"


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    ratio: str
    break_mode: str
    expected_status: str
    expected_reason: str


def add_solver_lanes(
    result: list[RunSpec], mode: str, ratio: str, break_mode: str
) -> None:
    result.append(
        RunSpec(
            f"ratio-{ratio}-break-{break_mode}-tgs-parallel",
            "tgs",
            "parallel",
            ratio,
            break_mode,
            "PASS",
            "none",
        )
    )
    for execution in ("parallel", "sequential"):
        result.append(
            RunSpec(
                f"ratio-{ratio}-break-{break_mode}-avbd-{execution}",
                "avbd",
                execution,
                ratio,
                break_mode,
                "FAIL" if mode == "failure-first" else "PASS",
                "missing_output_torque"
                if mode == "failure-first"
                else "none",
            )
        )


def specs(mode: str) -> list[RunSpec]:
    result: list[RunSpec] = []
    for ratio in ("1", "2"):
        add_solver_lanes(result, mode, ratio, "none")
    for break_mode in ("below", "above"):
        add_solver_lanes(result, mode, "1", break_mode)
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


def parse_float_field(
    fields: dict[str, str], key: str, errors: list[str]
) -> float | None:
    value = fields.get(key)
    if value is None:
        errors.append(f"missing numeric field: {key}")
        return None
    try:
        return float(value)
    except ValueError:
        errors.append(f"{key}={value!r}, expected finite numeric value")
        return None


def run_one(
    spec: RunSpec, bin_dir: Path, timeout_seconds: float
) -> tuple[bool, str]:
    executable = bin_dir / EXECUTABLE
    argv = [
        str(executable),
        "--headless",
        f"--solver={spec.solver}",
        "--case=impulse",
        f"--execution={spec.execution}",
        "--frames=180",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
        f"--ratio={spec.ratio}",
        "--impulse-frame=20",
        f"--break-mode={spec.break_mode}",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = "180"
    if spec.solver == "avbd":
        env["PHYSX_AVBD_ITER_DIAG"] = "1"
        env["PHYSX_AVBD_ITER_DIAG_EVERY"] = "60"
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

    expected_exit = 0 if spec.expected_status == "PASS" else 1
    if result.returncode != expected_exit:
        errors.append(
            f"exit code {result.returncode}, expected {expected_exit}"
        )

    required = {
        "schema": "1",
        "snippet": "SnippetCustomJoint",
        "solver": spec.solver,
        "case": "impulse",
        "execution": spec.execution,
        "frames": "180",
        "completedFrames": "180",
        "status": spec.expected_status,
        "reason": spec.expected_reason,
        "ratio": spec.ratio,
        "breakMode": spec.break_mode,
        "impulseEvents": "1",
        "forceReads": "180",
        "nonFiniteForceReads": "0",
        "outputForceMinimum": "100",
        "outputTorqueMinimum": "500",
        "outputTorqueRatioMinimum": "4.9",
        "outputTorqueRatioMaximum": "5.1",
        "breakFrameMaximumOffset": "2",
        "nonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
    }
    for key, expected in required.items():
        actual = fields.get(key)
        if actual != expected:
            errors.append(f"{key}={actual!r}, expected {expected!r}")

    max_linear_force = parse_float_field(fields, "maxLinearForce", errors)
    max_angular_force = parse_float_field(fields, "maxAngularForce", errors)
    torque_ratio = parse_float_field(
        fields, "outputTorqueToLinearRatio", errors
    )
    signed_torque_ratio = parse_float_field(
        fields, "signedTorqueToLinearYRatio", errors
    )
    peak_linear_x = parse_float_field(fields, "peakLinearForceX", errors)
    peak_linear_z = parse_float_field(fields, "peakLinearForceZ", errors)
    peak_angular_x = parse_float_field(fields, "peakAngularForceX", errors)
    peak_angular_y = parse_float_field(fields, "peakAngularForceY", errors)
    broken_reads = parse_float_field(fields, "brokenReads", errors)
    first_broken_frame = parse_float_field(
        fields, "firstBrokenFrame", errors
    )
    if max_linear_force is not None and max_linear_force < 100.0:
        errors.append(
            f"maxLinearForce={max_linear_force}, expected at least 100"
        )
    if spec.expected_status == "PASS":
        if max_angular_force is not None and max_angular_force < 500.0:
            errors.append(
                f"maxAngularForce={max_angular_force}, expected at least 500"
            )
        if torque_ratio is not None and not 4.9 <= torque_ratio <= 5.1:
            errors.append(
                f"outputTorqueToLinearRatio={torque_ratio}, "
                "expected within [4.9, 5.1]"
            )
        if (
            signed_torque_ratio is not None
            and not 4.9 <= signed_torque_ratio <= 5.1
        ):
            errors.append(
                f"signedTorqueToLinearYRatio={signed_torque_ratio}, "
                "expected within [4.9, 5.1]"
            )
    else:
        if max_angular_force is not None and abs(max_angular_force) > 1.0e-4:
            errors.append(
                f"maxAngularForce={max_angular_force}, "
                "expected isolated missing-torque witness <= 1e-4"
            )
        if torque_ratio is not None and abs(torque_ratio) > 1.0e-6:
            errors.append(
                f"outputTorqueToLinearRatio={torque_ratio}, "
                "expected isolated missing-torque witness <= 1e-6"
            )
        if (
            signed_torque_ratio is not None
            and abs(signed_torque_ratio) > 1.0e-6
        ):
            errors.append(
                f"signedTorqueToLinearYRatio={signed_torque_ratio}, "
                "expected isolated missing-torque witness <= 1e-6"
            )

    for key, value in (
        ("peakLinearForceX", peak_linear_x),
        ("peakLinearForceZ", peak_linear_z),
        ("peakAngularForceX", peak_angular_x),
        ("peakAngularForceY", peak_angular_y),
    ):
        if value is not None and abs(value) > 1.0e-3:
            errors.append(f"{key}={value}, expected pulley-axis residual <= 1e-3")

    if spec.break_mode == "below":
        expected_break_force = "3000"
        if fields.get("angularBreakForce") != expected_break_force:
            errors.append(
                f"angularBreakForce={fields.get('angularBreakForce')!r}, "
                f"expected {expected_break_force!r}"
            )
        if spec.expected_status == "PASS":
            if broken_reads is not None and broken_reads <= 0:
                errors.append("brokenReads must be positive for below bracket")
            if (
                first_broken_frame is not None
                and not 20 <= first_broken_frame <= 22
            ):
                errors.append(
                    f"firstBrokenFrame={first_broken_frame}, "
                    "expected within [20, 22]"
                )
    else:
        if spec.break_mode == "above" and fields.get(
            "angularBreakForce"
        ) != "10000":
            errors.append(
                f"angularBreakForce={fields.get('angularBreakForce')!r}, "
                "expected '10000'"
            )
        if spec.expected_status == "PASS":
            if broken_reads is not None and broken_reads != 0:
                errors.append(
                    f"brokenReads={broken_reads}, expected zero for intact lane"
                )
            if (
                first_broken_frame is not None
                and first_broken_frame != 4294967295
            ):
                errors.append(
                    f"firstBrokenFrame={first_broken_frame}, "
                    "expected UINT32_MAX for intact lane"
                )

    if spec.solver == "avbd":
        objective_errors, _ = validate_joint_objective_ir(
            combined, expected_owner="PositionAL"
        )
        errors.extend(objective_errors)

    print(
        f"[CUSTOM_JOINT_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"exit={result.returncode} runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[CUSTOM_JOINT_RUN_ERROR] name={spec.name} error={error}")
    return not errors, combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("failure-first", "acceptance"),
        default="acceptance",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(f"[CUSTOM_JOINT_RUNNER_ERROR] missing executable: {executable}")
        return 2
    if args.timeout <= 0.0:
        print("[CUSTOM_JOINT_RUNNER_ERROR] --timeout must be positive")
        return 2

    run_specs = specs(args.mode)
    all_passed = True
    completed = 0
    for spec in run_specs:
        passed, _ = run_one(spec, bin_dir, args.timeout)
        completed += 1
        all_passed = all_passed and passed

    print(
        f"[CUSTOM_JOINT_MATRIX] mode={args.mode} "
        f"completed={completed} total={len(run_specs)} "
        f"status={'PASS' if all_passed else 'FAIL'}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
