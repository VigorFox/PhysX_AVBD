#!/usr/bin/env python3
"""Run the Phase-0 rigid body-static normal ownership fixtures headlessly."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetContactModification_64.exe"
CASES = (
    "ownership-shallow",
    "ownership-deep",
    "finite-max-impulse",
    "ownership-bounce",
)
SELECTABLE_CASES = CASES + (
    "ownership-deep-tilted",
    "restitution-tilted",
    "finite-max-impulse-tilted",
    "finite-max-impulse-offcenter",
)
OWNERSHIP_PATTERN = re.compile(r"normalOwnership\(([^)]*)\)")


@dataclass(frozen=True)
class RunSpec:
    case_name: str
    solver: str
    execution: str

    @property
    def name(self) -> str:
        return f"{self.case_name}-{self.solver}-{self.execution}"


def parse_fields(text: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in text.split():
        if "=" not in token:
            errors.append(f"malformed token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append(f"duplicate key: {key}")
        fields[key] = value
    return fields, errors


def specs(selected_case: str | None) -> list[RunSpec]:
    cases = (selected_case,) if selected_case else CASES
    result: list[RunSpec] = []
    for case_name in cases:
        result.append(RunSpec(case_name, "tgs", "parallel"))
        result.append(RunSpec(case_name, "avbd", "parallel"))
        result.append(RunSpec(case_name, "avbd", "sequential"))
    return result


def summarize_ownership(
    lines: list[str],
) -> tuple[dict[str, float], list[str]]:
    count_fields = (
        "alRows",
        "alEvals",
        "depenEligibleRows",
        "depenCorrections",
        "finiteImpulseSkips",
        "authoredFiniteSkips",
        "velocityCorrections",
        "restitutionCorrections",
    )
    sum_fields = ("depenDistance", "velocityDelta")
    max_fields = ("depenMax", "velocityMax")
    summary = {key: 0.0 for key in count_fields + sum_fields + max_fields}
    errors: list[str] = []

    for line in lines:
        match = OWNERSHIP_PATTERN.search(line)
        if not match:
            errors.append("diagnostic line missing normalOwnership group")
            continue
        fields, parse_errors = parse_fields(match.group(1))
        errors.extend(parse_errors)
        for key in count_fields + sum_fields + max_fields:
            try:
                value = float(fields[key])
            except (KeyError, ValueError):
                errors.append(f"missing/non-numeric diagnostic field: {key}")
                continue
            if key in max_fields:
                summary[key] = max(summary[key], value)
            else:
                summary[key] += value
    return summary, errors


def run_one(
    spec: RunSpec, bin_dir: Path, timeout_seconds: float
) -> tuple[bool, dict[str, str], dict[str, float]]:
    argv = [
        str(bin_dir / EXECUTABLE),
        "--headless",
        f"--solver={spec.solver}",
        f"--case={spec.case_name}",
        f"--execution={spec.execution}",
        "--frames=120",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_SNIPPET_SOLVER"] = spec.solver
    env["PHYSX_SNIPPET_FRAME_COUNT"] = "120"
    env["PHYSX_AVBD_ITER_DIAG"] = "1" if spec.solver == "avbd" else "0"
    env["PHYSX_AVBD_ITER_DIAG_EVERY"] = "1"
    env["PHYSX_AVBD_ITER_DIAG_SEQUENTIAL"] = (
        "1" if spec.execution == "sequential" else "0"
    )
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
    diagnostic_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[avbd:iters] ")
    ]
    errors: list[str] = []
    fields: dict[str, str] = {}
    ownership: dict[str, float] = {}

    if result.timed_out:
        errors.append("timed out")
    if result.visible_window_detected:
        errors.append(
            "visible window detected: "
            + ", ".join(result.visible_window_titles)
        )
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}, expected 0")
    if len(authority_lines) != 1:
        errors.append(
            f"authority count is {len(authority_lines)}, expected exactly 1"
        )
    else:
        fields, parse_errors = parse_fields(
            " ".join(authority_lines[0].split()[1:])
        )
        errors.extend(parse_errors)

    required = {
        "schema": "2",
        "snippet": "SnippetContactModification",
        "solver": spec.solver,
        "case": spec.case_name,
        "execution": spec.execution,
        "frames": "120",
        "completedFrames": "120",
        "status": "PASS",
        "reason": "none",
        "validation": "GATED",
        "nonFinite": "0",
        "fetchFailures": "0",
        "fatalErrors": "0",
        "cleanupComplete": "1",
        "pvd": "0",
    }
    for key, expected in required.items():
        if fields.get(key) != expected:
            errors.append(
                f"{key}={fields.get(key)!r}, expected {expected!r}"
            )

    if spec.solver == "avbd":
        if not diagnostic_lines:
            errors.append("missing AVBD iteration diagnostics")
        else:
            ownership, diagnostic_errors = summarize_ownership(
                diagnostic_lines
            )
            errors.extend(diagnostic_errors)
            if ownership["alRows"] <= 0 or ownership["alEvals"] <= 0:
                errors.append("body-static AL ownership was not observed")
            if (
                spec.case_name
                in (
                    "ownership-shallow",
                    "ownership-deep",
                    "ownership-deep-tilted",
                    "ownership-bounce",
                )
                and ownership["depenEligibleRows"] <= 0
            ):
                errors.append("depenetration eligibility was not observed")
            if (
                spec.case_name
                in (
                    "finite-max-impulse",
                    "finite-max-impulse-tilted",
                    "finite-max-impulse-offcenter",
                )
                and ownership["authoredFiniteSkips"] <= 0
            ):
                errors.append(
                    "authored finite-impulse depenetration skip was not observed"
                )
            if (
                spec.case_name == "ownership-bounce"
                and (
                    ownership["velocityCorrections"] <= 0
                    or ownership["restitutionCorrections"] <= 0
                )
            ):
                errors.append("material restitution ownership was not observed")
    elif diagnostic_lines:
        errors.append("TGS unexpectedly emitted AVBD diagnostics")

    print(
        "[RIGID_NORMAL_OWNERSHIP_RUN] "
        f"name={spec.name} status={fields.get('status', 'MISSING')} "
        f"diagLines={len(diagnostic_lines)} "
        f"runner={'PASS' if not errors else 'FAIL'}"
    )
    if ownership:
        print(
            "[RIGID_NORMAL_OWNERSHIP_STATS] "
            f"name={spec.name} "
            + " ".join(
                f"{key}={value:.9g}"
                for key, value in sorted(ownership.items())
            )
        )
    concise_output = "\n".join(
        line
        for line in combined.splitlines()
        if not line.startswith("[avbd:iters] ")
    )
    if concise_output:
        print(concise_output.rstrip())
    for error in errors:
        print(
            "[RIGID_NORMAL_OWNERSHIP_ERROR] "
            f"name={spec.name} error={error}"
        )
    return not errors, fields, ownership


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=SELECTABLE_CASES)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    bin_dir = args.bin_dir.resolve()
    executable = bin_dir / EXECUTABLE
    if not executable.is_file():
        print(
            "[RIGID_NORMAL_OWNERSHIP_RUNNER_ERROR] "
            f"missing executable: {executable}"
        )
        return 2
    if args.timeout <= 0.0:
        print(
            "[RIGID_NORMAL_OWNERSHIP_RUNNER_ERROR] "
            "--timeout must be positive"
        )
        return 2

    passed = 0
    failed = 0
    for spec in specs(args.case):
        run_passed, _, _ = run_one(spec, bin_dir, args.timeout)
        passed += run_passed
        failed += not run_passed
        if not run_passed:
            break

    expected = len(specs(args.case))
    matrix_passed = failed == 0 and passed == expected
    print(
        "[RIGID_NORMAL_OWNERSHIP_MATRIX] "
        f"passed={passed} failed={failed} expected={expected} "
        f"status={'PASS' if matrix_passed else 'FAIL'}"
    )
    return 0 if matrix_passed else 1


if __name__ == "__main__":
    sys.exit(main())
