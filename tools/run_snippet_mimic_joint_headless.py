#!/usr/bin/env python3
"""Run the SnippetMimicJoint failure-first or acceptance matrix headlessly."""

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
    ratio: str
    offset: str


def specs() -> list[RunSpec]:
    result: list[RunSpec] = []
    for case_name in ("drive-a", "drive-b"):
        for ratio, offset in (("1", "0.25"), ("-2", "0.25")):
            result.append(
                RunSpec(
                    f"{case_name}-ratio-{ratio}-offset-{offset}-tgs",
                    "tgs",
                    "parallel",
                    case_name,
                    ratio,
                    offset,
                )
            )
            for execution in ("parallel", "sequential"):
                result.append(
                    RunSpec(
                        f"{case_name}-ratio-{ratio}-offset-{offset}"
                        f"-avbd-{execution}",
                        "avbd",
                        execution,
                        case_name,
                        ratio,
                        offset,
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
    spec: RunSpec, mode: str, bin_dir: Path, timeout_seconds: float
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
        f"--ratio={spec.ratio}",
        f"--offset={spec.offset}",
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
        "ratio": spec.ratio,
        "ratioReadback": spec.ratio,
        "offset": spec.offset,
        "offsetReadback": spec.offset,
        "axisA": "twist",
        "axisB": "x",
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

    if mode == "acceptance":
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
        if fields.get("validation") != "GATED":
            errors.append(
                f"validation={fields.get('validation')!r}, expected 'GATED'"
            )
    elif result.returncode not in (0, 1):
        errors.append(
            f"probe exit code {result.returncode}, expected physics result 0/1"
        )
    if fields.get("status") not in ("PASS", "FAIL"):
        errors.append(
            f"status={fields.get('status')!r}, expected PASS or FAIL"
        )
    if fields.get("validation") not in ("PROBE", "GATED"):
        errors.append(
            f"validation={fields.get('validation')!r}, expected PROBE/GATED"
        )

    print(
        f"[MIMIC_JOINT_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(f"[MIMIC_JOINT_RUN_ERROR] name={spec.name} error={error}")
    return not errors, fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("probe", "acceptance"), default="probe"
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(f"[MIMIC_JOINT_RUNNER_ERROR] missing executable: {executable}")
        return 2
    if args.timeout <= 0.0:
        print("[MIMIC_JOINT_RUNNER_ERROR] --timeout must be positive")
        return 2

    all_infrastructure_passed = True
    physics_passes = 0
    physics_failures = 0
    for spec in specs():
        passed, fields = run_one(spec, args.mode, bin_dir, args.timeout)
        all_infrastructure_passed = all_infrastructure_passed and passed
        physics_passes += fields.get("status") == "PASS"
        physics_failures += fields.get("status") == "FAIL"
        if not passed:
            break

    print(
        f"[MIMIC_JOINT_MATRIX] mode={args.mode} "
        f"physicsPasses={physics_passes} physicsFailures={physics_failures} "
        f"status={'PASS' if all_infrastructure_passed else 'FAIL'}"
    )
    return 0 if all_infrastructure_passed else 1


if __name__ == "__main__":
    sys.exit(main())
