#!/usr/bin/env python3
"""Run SnippetCustomGeometry voxel-contact coverage headlessly."""

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
EXECUTABLE = "SnippetCustomGeometry_64.exe"
CASES = ("drop", "impact")


@dataclass(frozen=True)
class RunSpec:
    name: str
    solver: str
    execution: str
    case_name: str


def specs(mode: str) -> tuple[RunSpec, ...]:
    authority = tuple(
        RunSpec(f"{case_name}-tgs", "tgs", "parallel", case_name)
        for case_name in CASES
    )
    if mode == "authority":
        return authority
    avbd = tuple(
        RunSpec(
            f"{case_name}-avbd-{execution}",
            "avbd",
            execution,
            case_name,
        )
        for case_name in CASES
        for execution in ("parallel", "sequential")
    )
    return authority + avbd


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
    spec: RunSpec,
    mode: str,
    bin_dir: Path,
    timeout_seconds: float,
) -> tuple[bool, dict[str, str]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case_name}",
        f"--execution={spec.execution}",
        "--frames=180",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = "180"
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
        "snippet": "SnippetCustomGeometry",
        "solver": spec.solver,
        "case": spec.case_name,
        "execution": spec.execution,
        "frames": "180",
        "completedFrames": "180",
        "identityErrors": "0",
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
    for key in (
        "generateCalls",
        "generatedContacts",
        "callbackCount",
        "pairCount",
        "reportedPoints",
        "nonzeroImpulseCount",
    ):
        try:
            if int(fields.get(key, "0")) <= 0:
                errors.append(
                    f"{key}={fields.get(key)!r}, expected positive"
                )
        except ValueError:
            errors.append(f"{key}={fields.get(key)!r}, expected integer")

    require_pass = mode in ("authority", "acceptance")
    if require_pass:
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
    elif result.returncode not in (0, 1):
        errors.append(
            f"probe exit code {result.returncode}, expected physics result 0/1"
        )
    if fields.get("status") not in ("PASS", "FAIL"):
        errors.append(
            f"status={fields.get('status')!r}, expected PASS or FAIL"
        )
    if fields.get("validation") != "GATED":
        errors.append(
            f"validation={fields.get('validation')!r}, expected GATED"
        )

    print(
        f"[CUSTOM_GEOMETRY_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            f"[CUSTOM_GEOMETRY_RUN_ERROR] name={spec.name} error={error}"
        )
    return not errors, fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("authority", "probe", "acceptance"),
        default="probe",
    )
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    if not (bin_dir / EXECUTABLE).is_file():
        print(
            "[CUSTOM_GEOMETRY_RUNNER_ERROR] "
            f"missing executable: {bin_dir / EXECUTABLE}"
        )
        return 2
    if args.timeout <= 0.0:
        print("[CUSTOM_GEOMETRY_RUNNER_ERROR] --timeout must be positive")
        return 2

    all_passed = True
    physics_passes = 0
    physics_failures = 0
    for spec in specs(args.mode):
        passed, fields = run_one(spec, args.mode, bin_dir, args.timeout)
        all_passed = all_passed and passed
        physics_passes += fields.get("status") == "PASS"
        physics_failures += fields.get("status") == "FAIL"
        if not passed:
            break
    print(
        f"[CUSTOM_GEOMETRY_MATRIX] mode={args.mode} "
        f"physicsPasses={physics_passes} physicsFailures={physics_failures} "
        f"status={'PASS' if all_passed else 'FAIL'}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
