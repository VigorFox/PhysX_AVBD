#!/usr/bin/env python3
"""Run missing SpatialTendon feature modes without opening a window."""

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
EXECUTABLE = "SnippetSpatialTendon_64.exe"


@dataclass(frozen=True)
class Fixture:
    spatial_mode: str
    case_name: str
    axis: str
    attachment_count: str
    leaf_count: str
    second_leaf_coefficient: str
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
    expect_pass: bool


FIXTURES = (
    Fixture(
        "moving-middle",
        "middle-drive",
        "swing2",
        "3",
        "1",
        "0",
        "5000",
        "100",
        "0",
        "3.40282347e+38",
        "-3.40282347e+38",
    ),
    Fixture(
        "multi-leaf",
        "offset-actuation",
        "swing2",
        "4",
        "2",
        "1",
        "5000",
        "100",
        "0",
        "3.40282347e+38",
        "-3.40282347e+38",
    ),
    Fixture(
        "linear-axis",
        "offset-actuation",
        "x",
        "3",
        "1",
        "0",
        "5000",
        "100",
        "0",
        "3.40282347e+38",
        "-3.40282347e+38",
    ),
    Fixture(
        "limit",
        "offset-actuation",
        "swing2",
        "3",
        "1",
        "0",
        "0",
        "0",
        "5000",
        "2.82842708",
        "3.22842717",
    ),
)


def specs(mode: str) -> list[RunSpec]:
    result: list[RunSpec] = []
    for fixture in FIXTURES:
        result.append(
            RunSpec(
                f"{fixture.spatial_mode}-tgs",
                fixture,
                "tgs",
                "parallel",
                True,
            )
        )
        if mode == "authority":
            continue
        for execution in ("parallel", "sequential"):
            result.append(
                RunSpec(
                    f"{fixture.spatial_mode}-avbd-{execution}",
                    fixture,
                    "avbd",
                    execution,
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


def parse_finite_float(
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


def validate_expected_failure(
    fields: dict[str, str], errors: list[str]
) -> None:
    if fields.get("reason") != "missing_coupled_response":
        errors.append(
            "reason="
            f"{fields.get('reason')!r}, expected 'missing_coupled_response'"
        )
    range_a = parse_finite_float(fields, "rangeA", errors)
    range_b = parse_finite_float(fields, "rangeB", errors)
    if range_a is not None and range_a >= 0.03:
        errors.append(f"rangeA={range_a}, expected < 0.03 failure witness")
    if range_b is not None and range_b >= 0.03:
        errors.append(f"rangeB={range_b}, expected < 0.03 failure witness")


def run_one(
    spec: RunSpec, bin_dir: Path, timeout_seconds: float
) -> tuple[bool, dict[str, str]]:
    executable = bin_dir / EXECUTABLE
    argv = [
        str(executable),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.fixture.case_name}",
        f"--execution={spec.execution}",
        f"--spatial-mode={spec.fixture.spatial_mode}",
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

    fixture = spec.fixture
    required = {
        "schema": "1",
        "snippet": "SnippetSpatialTendon",
        "spatialMode": fixture.spatial_mode,
        "axis": fixture.axis,
        "attachmentCount": fixture.attachment_count,
        "leafCount": fixture.leaf_count,
        "solver": spec.solver,
        "case": fixture.case_name,
        "execution": spec.execution,
        "frames": "480",
        "completedFrames": "480",
        "validation": "GATED",
        "coefficientRoot": "1",
        "coefficientMiddle": "1",
        "coefficientLeaf": "1",
        "coefficientSecondLeaf": fixture.second_leaf_coefficient,
        "offsetBase": "0.200000003",
        "restLength": "3.02842708",
        "stiffness": fixture.stiffness,
        "damping": fixture.damping,
        "limitStiffness": fixture.limit_stiffness,
        "lowLimit": fixture.low_limit,
        "highLimit": fixture.high_limit,
        "responseSamples": "420",
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
        validate_expected_failure(fields, errors)

    print(
        f"[SPATIAL_TENDON_MODE_RUN] name={spec.name} "
        f"status={fields.get('status', 'MISSING')} "
        f"reason={fields.get('reason', 'MISSING')} "
        f"exit={result.returncode} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if combined:
        print(combined.rstrip())
    for error in errors:
        print(
            f"[SPATIAL_TENDON_MODE_RUN_ERROR] "
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
            "[SPATIAL_TENDON_MODE_RUNNER_ERROR] "
            f"missing executable: {executable}"
        )
        return 2
    if args.timeout <= 0.0:
        print(
            "[SPATIAL_TENDON_MODE_RUNNER_ERROR] "
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
        f"[SPATIAL_TENDON_MODE_MATRIX] mode={args.mode} "
        f"completed={physics_passes + physics_failures} "
        f"total={len(run_specs)} physicsPasses={physics_passes} "
        f"physicsFailures={physics_failures} "
        f"status={'PASS' if all_passed else 'FAIL'}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
