#!/usr/bin/env python3
"""Run compliant and multi-mimic SnippetMimicJoint modes without UI."""

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
EXECUTABLE = "SnippetMimicJoint_64.exe"


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    case_name: str
    mimic_mode: str
    natural_frequency: str
    damping_ratio: str
    mimic_count: str


def specs(include_avbd: bool) -> list[RunSpec]:
    result: list[RunSpec] = []
    for mimic_mode, frequency, damping, count in (
        ("compliant", "6", "1", "1"),
        ("multi", "0", "0", "2"),
    ):
        for case_name in ("drive-a", "drive-b"):
            result.append(
                RunSpec(
                    f"{mimic_mode}-{case_name}-tgs",
                    "tgs",
                    "parallel",
                    case_name,
                    mimic_mode,
                    frequency,
                    damping,
                    count,
                )
            )
            if include_avbd:
                for execution in ("parallel", "sequential"):
                    result.append(
                        RunSpec(
                            f"{mimic_mode}-{case_name}-avbd-{execution}",
                            "avbd",
                            execution,
                            case_name,
                            mimic_mode,
                            frequency,
                            damping,
                            count,
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


def number(
    fields: dict[str, str], key: str, errors: list[str]
) -> float | None:
    try:
        return float(fields[key])
    except (KeyError, ValueError):
        errors.append(f"{key}={fields.get(key)!r}, expected finite number")
        return None


def run_one(
    spec: RunSpec,
    mode: str,
    bin_dir: Path,
    timeout_seconds: float,
) -> tuple[bool, dict[str, str]]:
    executable = bin_dir / EXECUTABLE
    argv = [
        str(executable),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case_name}",
        f"--execution={spec.execution}",
        "--frames=360",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
        "--ratio=1",
        "--offset=0.25",
        f"--mimic-mode={spec.mimic_mode}",
        f"--natural-frequency={spec.natural_frequency}",
        f"--damping-ratio={spec.damping_ratio}",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = "360"
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
        "snippet": "SnippetMimicJoint",
        "solver": spec.solver,
        "case": spec.case_name,
        "execution": spec.execution,
        "frames": "360",
        "completedFrames": "360",
        "mimicMode": spec.mimic_mode,
        "mimicCount": spec.mimic_count,
        "ratio": "1",
        "ratioReadback": "1",
        "offset": "0.25",
        "offsetReadback": "0.25",
        "naturalFrequency": spec.natural_frequency,
        "naturalFrequencyReadback": spec.natural_frequency,
        "dampingRatio": spec.damping_ratio,
        "dampingRatioReadback": spec.damping_ratio,
        "responseSamples": "300",
        "nonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": "1",
        "pvd": "0",
    }
    for key, expected in required.items():
        actual = fields.get(key)
        if actual != expected:
            errors.append(f"{key}={actual!r}, expected {expected!r}")

    expect_pass = (
        mode in ("authority", "acceptance") or spec.solver == "tgs"
    )
    if mode == "failure-first" and spec.solver == "avbd":
        expect_pass = False

    expected_status = "PASS" if expect_pass else "FAIL"
    expected_exit = 0 if expect_pass else 1
    if result.returncode != expected_exit:
        errors.append(
            f"exit code {result.returncode}, expected {expected_exit}"
        )
    if fields.get("status") != expected_status:
        errors.append(
            f"status={fields.get('status')!r}, expected {expected_status!r}"
        )
    if fields.get("validation") != "GATED":
        errors.append(
            f"validation={fields.get('validation')!r}, expected 'GATED'"
        )
    if expect_pass and fields.get("reason") != "none":
        errors.append(
            f"reason={fields.get('reason')!r}, expected 'none'"
        )
    if not expect_pass and fields.get("reason") not in {
        "missing_coupled_response",
        "missing_second_coupled_response",
        "wrong_coupling_direction",
        "wrong_second_coupling_direction",
        "mimic_position_error",
        "mimic_velocity_error",
        "second_mimic_error",
        "compliant_mimic_unbounded",
    }:
        errors.append(
            f"reason={fields.get('reason')!r}, expected physical mimic failure"
        )

    if expect_pass:
        range_a = number(fields, "rangeA", errors)
        range_b = number(fields, "rangeB", errors)
        if range_a is not None and range_a < 0.15:
            errors.append(f"rangeA={range_a}, expected >= 0.15")
        if range_b is not None and range_b < 0.15:
            errors.append(f"rangeB={range_b}, expected >= 0.15")
        if spec.mimic_mode == "multi":
            range_c = number(fields, "rangeC", errors)
            range_d = number(fields, "rangeD", errors)
            if fields.get("secondResponseSamples") != "300":
                errors.append(
                    "secondResponseSamples="
                    f"{fields.get('secondResponseSamples')!r}, expected '300'"
                )
            if range_c is not None and range_c < 0.15:
                errors.append(f"rangeC={range_c}, expected >= 0.15")
            if range_d is not None and range_d < 0.15:
                errors.append(f"rangeD={range_d}, expected >= 0.15")

    print(
        f"[MIMIC_MODE_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[MIMIC_MODE_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "failure-first", "acceptance"),
        default="authority",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(f"[MIMIC_MODE_RUNNER_ERROR] missing executable: {executable}")
        return 2
    if args.timeout <= 0.0:
        print("[MIMIC_MODE_RUNNER_ERROR] --timeout must be positive")
        return 2

    all_passed = True
    physics_passes = 0
    physics_failures = 0
    for spec in specs(args.mode != "authority"):
        passed, fields = run_one(
            spec, args.mode, bin_dir, args.timeout
        )
        all_passed = all_passed and passed
        physics_passes += fields.get("status") == "PASS"
        physics_failures += fields.get("status") == "FAIL"
        if not passed:
            break

    print(
        f"[MIMIC_MODE_MATRIX] mode={args.mode} "
        f"physicsPasses={physics_passes} "
        f"physicsFailures={physics_failures} "
        f"status={'PASS' if all_passed else 'FAIL'}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
