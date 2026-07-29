#!/usr/bin/env python3
"""Inventory AVBD mixed-solve ownership on fixed CPU Snippet gates.

This runner is deliberately read-only with respect to solver behavior.  It
reuses the accepted cross-Snippet argv contracts, launches every executable
through the hidden Job-Object guard, enables low-frequency AVBD ownership
diagnostics, and aggregates sampled row/correction counts.  It excludes the
expensive deformable deep-diagnostic cases; their frozen P3L/P3R artifacts
remain the authority for that population.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import run_avbd_cross_snippets_headless as cross
from snippet_headless_process import (
    run_headless_process,
    windows_creation_flags,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
DEFAULT_CASES = (
    "joint-passive-parallel",
    "joint-impact-all-sequential",
    "avbd-articulation-full-sequential",
    "articulation-rc-cycle-sequential",
    "hello-world-stack",
    "hello-world-ball",
    "gear-steady-sequential",
    "serialization-spherical-binary",
)
EXCLUDED_DEFORMABLE_CASES = {
    "deformable-mesh-stack",
    "deformable-mesh-sphere",
}
ALL_SPECS = {
    spec.name: spec
    for spec in cross.full_specs()
    if spec.name not in EXCLUDED_DEFORMABLE_CASES
}

INTEGER_FIELDS = (
    "rows",
    "objectivePositionRows",
    "objectivePointRows",
    "objectiveManifoldRows",
    "objectiveComponentRows",
    "objectiveJointRows",
    "objectiveUnsupportedRows",
    "objectiveLegacyRows",
    "objectiveInvalidRows",
    "objectiveFingerprint",
    "contactObjectiveSlots",
    "contactObjectivePositionSlots",
    "contactObjectivePointSlots",
    "contactObjectiveManifoldSlots",
    "contactObjectiveComponentSlots",
    "contactObjectiveJointSlots",
    "contactObjectiveUnsupportedSlots",
    "contactObjectiveLegacySlots",
    "contactObjectiveInvalidSlots",
    "contactObjectiveLegacyNormalSlots",
    "contactObjectiveLegacyTangentSlots",
    "contactObjectiveLegacyRigidStaticTangentSlots",
    "contactObjectiveLegacyDynamicTangentSlots",
    "contactObjectiveLegacyDeformableTangentSlots",
    "contactObjectiveLegacyJointMixedTangentSlots",
    "contactObjectiveLegacyOtherTangentSlots",
    "contactObjectiveFingerprint",
    "jointObjectiveRows",
    "jointObjectivePositionRows",
    "jointObjectiveFinalizeRows",
    "jointObjectiveUnsupportedRows",
    "jointObjectiveLegacyRows",
    "jointObjectiveInvalidRows",
    "jointObjectiveFingerprint",
    "alRows",
    "velocityCorrections",
    "restitutionCorrections",
    "lockLin",
    "limLin",
    "lockAng",
    "limAng",
    "linDrv",
    "angDrv",
    "cone",
    "positionAlTargetEvals",
    "bodyStaticSweepTargetRows",
    "bodyStaticSweepTargetCorrections",
    "bodyStaticFallbackRows",
    "bodyStaticFallbackCorrections",
    "genericNormalRows",
    "genericNormalCorrections",
    "genericTangentRows",
    "genericTangentCorrections",
    "deformPositionTangentRows",
    "deformFrictionRawRows",
    "deformFrictionDominantRows",
    "deformFrictionCorrections",
    "deformFinalizeSpatialCorrections",
    "deformFinalizeComFallbackCorrections",
)
REAL_FIELDS = (
    "bodyStaticSweepTargetImpulse",
    "bodyStaticFallbackImpulse",
    "genericNormalImpulse",
    "genericTangentImpulse",
    "deformFrictionImpulse",
)
DIAGNOSTIC_PREFIXES = (
    "[avbd:objective-ir] ",
    "[avbd:contact-objective-ir] ",
    "[avbd:joint-objective-ir] ",
    "[avbd:iters] ",
    "[avbd:friction-target] ",
    "[avbd:surface-ownership] ",
)
OBJECTIVE_PARTITION_FIELDS = (
    "objectivePositionRows",
    "objectivePointRows",
    "objectiveManifoldRows",
    "objectiveComponentRows",
    "objectiveJointRows",
    "objectiveUnsupportedRows",
    "objectiveLegacyRows",
    "objectiveInvalidRows",
)
CONTACT_OBJECTIVE_PARTITION_FIELDS = (
    "contactObjectivePositionSlots",
    "contactObjectivePointSlots",
    "contactObjectiveManifoldSlots",
    "contactObjectiveComponentSlots",
    "contactObjectiveJointSlots",
    "contactObjectiveUnsupportedSlots",
    "contactObjectiveLegacySlots",
    "contactObjectiveInvalidSlots",
)
CONTACT_OBJECTIVE_LEGACY_SOURCE_FIELDS = (
    "contactObjectiveLegacyNormalSlots",
    "contactObjectiveLegacyTangentSlots",
)
CONTACT_OBJECTIVE_LEGACY_TANGENT_TOPOLOGY_FIELDS = (
    "contactObjectiveLegacyRigidStaticTangentSlots",
    "contactObjectiveLegacyDynamicTangentSlots",
    "contactObjectiveLegacyDeformableTangentSlots",
    "contactObjectiveLegacyJointMixedTangentSlots",
    "contactObjectiveLegacyOtherTangentSlots",
)
JOINT_OBJECTIVE_PARTITION_FIELDS = (
    "jointObjectivePositionRows",
    "jointObjectiveFinalizeRows",
    "jointObjectiveUnsupportedRows",
    "jointObjectiveLegacyRows",
    "jointObjectiveInvalidRows",
)
NUMBER_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9]*)="
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
ITER_CONTEXT_PATTERN = re.compile(
    r"(normalOwnership|jointRows)\(([^)]*)\)"
)


@dataclass
class InventoryResult:
    name: str
    executable: str
    execution: str
    requested_frames: int
    cadence: int
    diagnostic_samples: int
    passed: bool
    exit_code: int | None
    timed_out: bool
    visible_window_detected: bool
    residual_process: bool
    authority_count: int
    executable_sha256_before: str
    executable_sha256_after: str
    integer_totals: dict[str, int]
    real_totals: dict[str, str]
    errors: list[str]
    log: str


def parse_numeric_fields(line: str) -> dict[str, Decimal]:
    values: dict[str, Decimal] = {}
    for key, raw_value in NUMBER_PATTERN.findall(line):
        try:
            values[key] = Decimal(raw_value)
        except InvalidOperation:
            continue
    return values


def parse_diagnostic_line(line: str, prefix: str) -> dict[str, Decimal]:
    parsed = parse_numeric_fields(line)
    if prefix != "[avbd:iters] ":
        return parsed

    # The iters line deliberately reuses names such as linDrv/angDrv/cone in
    # jointRows(...) and jointLambdaMax(...).  Only jointRows is a row count;
    # parsing the whole line would let the later floating-point lambda value
    # overwrite the earlier integer count.
    contextual: dict[str, Decimal] = {}
    if "frame" in parsed:
        contextual["frame"] = parsed["frame"]
    for context, body in ITER_CONTEXT_PATTERN.findall(line):
        if context not in ("normalOwnership", "jointRows"):
            continue
        contextual.update(parse_numeric_fields(body))
    return contextual


def validate_authority(
    stdout: str, spec: cross.RunSpec, exit_code: int | None
) -> tuple[int, list[str]]:
    authority_lines = [
        line.strip()
        for line in stdout.splitlines()
        if line.startswith("[AVBD_GATE] ")
    ]
    errors: list[str] = []
    authority: dict[str, str] = {}
    if len(authority_lines) != 1:
        errors.append(
            f"authority count is {len(authority_lines)}, expected 1"
        )
    else:
        authority, parse_errors = cross.parse_authority(authority_lines[0])
        errors.extend(parse_errors)

    expected = {
        "schema": "1",
        "snippet": spec.snippet,
        "case": spec.case,
        "solver": spec.solver,
        "execution": spec.execution,
        "requestedFrames": str(spec.frames),
        "completedFrames": str(spec.frames),
        "seed": "1",
        "dispatcherThreads": "2",
        "capability": spec.capability,
        "validation": spec.validation,
        "status": spec.expected_status,
        "reason": spec.expected_reason,
        "nonFinite": "0",
        "physicsErrors": "0",
        "physicsWarnings": "0",
        **dict(spec.required_fields),
    }
    for key, expected_value in expected.items():
        actual_value = authority.get(key)
        if actual_value != expected_value:
            errors.append(
                f"{key}={actual_value or 'MISSING'}, "
                f"expected {expected_value}"
            )
    expected_exit = 0 if spec.expected_status == "PASS" else 1
    if exit_code != expected_exit:
        errors.append(f"exit code {exit_code}, expected {expected_exit}")
    return len(authority_lines), errors


def aggregate_diagnostics(
    stdout: str,
) -> tuple[int, dict[str, int], dict[str, str], list[str]]:
    integer_totals = {field: 0 for field in INTEGER_FIELDS}
    real_values = {field: Decimal(0) for field in REAL_FIELDS}
    samples = 0
    errors: list[str] = []
    prefix_counts = {prefix: 0 for prefix in DIAGNOSTIC_PREFIXES}

    for line_number, line in enumerate(stdout.splitlines(), start=1):
        prefix = next(
            (candidate for candidate in DIAGNOSTIC_PREFIXES
             if line.startswith(candidate)),
            None,
        )
        if prefix is None:
            continue
        prefix_counts[prefix] += 1
        if prefix == "[avbd:iters] ":
            samples += 1
        parsed = parse_diagnostic_line(line, prefix)
        if "frame" not in parsed:
            errors.append(
                f"diagnostic line {line_number} is missing frame"
            )
        if prefix == "[avbd:objective-ir] ":
            missing = [
                field
                for field in (
                    "rows",
                    *OBJECTIVE_PARTITION_FIELDS,
                    "objectiveFingerprint",
                )
                if field not in parsed
            ]
            if missing:
                errors.append(
                    f"diagnostic line {line_number} is missing "
                    + ", ".join(missing)
                )
            else:
                partition_rows = sum(
                    parsed[field] for field in OBJECTIVE_PARTITION_FIELDS
                )
                if parsed["rows"] != partition_rows:
                    errors.append(
                        f"diagnostic line {line_number} has rows="
                        f"{parsed['rows']} but partition={partition_rows}"
                    )
                if parsed["objectiveInvalidRows"] != 0:
                    errors.append(
                        f"diagnostic line {line_number} has "
                        "objectiveInvalidRows="
                        f"{parsed['objectiveInvalidRows']}"
                    )
        elif prefix == "[avbd:contact-objective-ir] ":
            missing = [
                field
                for field in (
                    "contactObjectiveSlots",
                    *CONTACT_OBJECTIVE_PARTITION_FIELDS,
                    *CONTACT_OBJECTIVE_LEGACY_SOURCE_FIELDS,
                    *CONTACT_OBJECTIVE_LEGACY_TANGENT_TOPOLOGY_FIELDS,
                    "contactObjectiveFingerprint",
                )
                if field not in parsed
            ]
            if missing:
                errors.append(
                    f"diagnostic line {line_number} is missing "
                    + ", ".join(missing)
                )
            else:
                partition_slots = sum(
                    parsed[field]
                    for field in CONTACT_OBJECTIVE_PARTITION_FIELDS
                )
                if parsed["contactObjectiveSlots"] != partition_slots:
                    errors.append(
                        f"diagnostic line {line_number} has "
                        "contactObjectiveSlots="
                        f"{parsed['contactObjectiveSlots']} but "
                        f"partition={partition_slots}"
                    )
                if parsed["contactObjectiveInvalidSlots"] != 0:
                    errors.append(
                        f"diagnostic line {line_number} has "
                        "contactObjectiveInvalidSlots="
                        f"{parsed['contactObjectiveInvalidSlots']}"
                    )
                legacy_sources = sum(
                    parsed[field]
                    for field in CONTACT_OBJECTIVE_LEGACY_SOURCE_FIELDS
                )
                if parsed["contactObjectiveLegacySlots"] != legacy_sources:
                    errors.append(
                        f"diagnostic line {line_number} has "
                        "contactObjectiveLegacySlots="
                        f"{parsed['contactObjectiveLegacySlots']} but "
                        f"legacy source partition={legacy_sources}"
                    )
                legacy_tangent_topologies = sum(
                    parsed[field]
                    for field in
                    CONTACT_OBJECTIVE_LEGACY_TANGENT_TOPOLOGY_FIELDS
                )
                if (
                    parsed["contactObjectiveLegacyTangentSlots"] !=
                    legacy_tangent_topologies
                ):
                    errors.append(
                        f"diagnostic line {line_number} has "
                        "contactObjectiveLegacyTangentSlots="
                        f"{parsed['contactObjectiveLegacyTangentSlots']} "
                        "but topology partition="
                        f"{legacy_tangent_topologies}"
                    )
                if (parsed["contactObjectiveSlots"] > 0 and
                        parsed["contactObjectivePositionSlots"] == 0):
                    errors.append(
                        f"diagnostic line {line_number} has contact "
                        "source slots but no PositionAL geometry owner"
                    )
        elif prefix == "[avbd:joint-objective-ir] ":
            missing = [
                field
                for field in (
                    "jointObjectiveRows",
                    *JOINT_OBJECTIVE_PARTITION_FIELDS,
                    "jointObjectiveFingerprint",
                )
                if field not in parsed
            ]
            if missing:
                errors.append(
                    f"diagnostic line {line_number} is missing "
                    + ", ".join(missing)
                )
            else:
                partition_rows = sum(
                    parsed[field]
                    for field in JOINT_OBJECTIVE_PARTITION_FIELDS
                )
                if parsed["jointObjectiveRows"] != partition_rows:
                    errors.append(
                        f"diagnostic line {line_number} has "
                        "jointObjectiveRows="
                        f"{parsed['jointObjectiveRows']} but "
                        f"partition={partition_rows}"
                    )
                if parsed["jointObjectiveInvalidRows"] != 0:
                    errors.append(
                        f"diagnostic line {line_number} has "
                        "jointObjectiveInvalidRows="
                        f"{parsed['jointObjectiveInvalidRows']}"
                    )
        for field in INTEGER_FIELDS:
            if field in parsed:
                value = parsed[field]
                if value != value.to_integral_value() or value < 0:
                    errors.append(
                        f"diagnostic line {line_number} has invalid "
                        f"integer {field}={value}"
                    )
                else:
                    integer_totals[field] += int(value)
        for field in REAL_FIELDS:
            if field in parsed:
                value = parsed[field]
                if not value.is_finite() or value < 0:
                    errors.append(
                        f"diagnostic line {line_number} has invalid "
                        f"real {field}={value}"
                    )
                else:
                    real_values[field] += value

    if samples == 0:
        errors.append("no [avbd:iters] diagnostic samples")
    for prefix, count in prefix_counts.items():
        if count != samples:
            errors.append(
                f"{prefix.strip()} count is {count}, expected {samples}"
            )
    real_totals = {
        field: format(value, "f") for field, value in real_values.items()
    }
    return samples, integer_totals, real_totals, errors


def make_output_root(requested: Path | None) -> Path:
    if requested is not None:
        output_root = requested.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        output_root = (
            Path(tempfile.gettempdir())
            / f"PhysX_AVBD_mixed_owner_inventory_{stamp}"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def run_one(
    bin_dir: Path,
    output_root: Path,
    spec: cross.RunSpec,
    cadence: int,
    timeout_seconds: float,
) -> InventoryResult:
    executable = bin_dir / spec.executable
    argv = cross.make_command(executable, spec)
    environment = os.environ.copy()
    environment["PHYSX_SNIPPET_HEADLESS"] = "1"
    environment["PHYSX_AVBD_ITER_DIAG"] = "1"
    environment["PHYSX_AVBD_ITER_DIAG_EVERY"] = str(cadence)

    before_hash = cross.sha256(executable)
    completed = run_headless_process(
        argv,
        cwd=bin_dir,
        env=environment,
        timeout_seconds=timeout_seconds,
    )
    after_hash = cross.sha256(executable)
    residual_process = cross.process_is_running(
        spec.executable, windows_creation_flags()
    )

    errors: list[str] = []
    if completed.timed_out:
        errors.append(f"timeout after {timeout_seconds:g} seconds")
    if completed.visible_window_detected:
        errors.append(
            "visible child window detected: "
            + ", ".join(completed.visible_window_titles)
        )
    if completed.stderr:
        errors.append(
            f"stderr is not empty "
            f"({len(completed.stderr.encode('utf-8'))} bytes)"
        )
    if after_hash != before_hash:
        errors.append("executable SHA-256 changed during the run")
    if residual_process:
        errors.append(f"residual process detected: {spec.executable}")

    authority_count, authority_errors = validate_authority(
        completed.stdout, spec, completed.returncode
    )
    errors.extend(authority_errors)
    (
        diagnostic_samples,
        integer_totals,
        real_totals,
        diagnostic_errors,
    ) = aggregate_diagnostics(completed.stdout)
    errors.extend(diagnostic_errors)

    log_path = output_root / f"{spec.name}.log"
    log_text = (
        f"COMMAND: {subprocess.list2cmdline(argv)}\n"
        "HEADLESS_ENV: PHYSX_SNIPPET_HEADLESS=1\n"
        f"ITER_DIAG: 1\n"
        f"ITER_DIAG_EVERY: {cadence}\n"
        f"CREATE_NO_WINDOW: {int(os.name == 'nt')}\n"
        f"KILL_ON_JOB_CLOSE: {int(os.name == 'nt')}\n"
        f"VISIBLE_WINDOW_DETECTED: "
        f"{int(completed.visible_window_detected)}\n"
        f"EXECUTABLE_SHA256_BEFORE: {before_hash}\n"
        f"EXECUTABLE_SHA256_AFTER: {after_hash}\n"
        f"EXIT_CODE: {completed.returncode}\n"
        f"TIMED_OUT: {int(completed.timed_out)}\n"
        f"RESIDUAL_PROCESS: {int(residual_process)}\n"
        "--- STDOUT ---\n"
        f"{completed.stdout}"
        "\n--- STDERR ---\n"
        f"{completed.stderr}"
        "\n--- RUNNER ERRORS ---\n"
        + ("\n".join(errors) if errors else "none")
        + "\n"
    )
    log_path.write_text(log_text, encoding="utf-8")

    return InventoryResult(
        name=spec.name,
        executable=spec.executable,
        execution=spec.execution,
        requested_frames=spec.frames,
        cadence=cadence,
        diagnostic_samples=diagnostic_samples,
        passed=not errors,
        exit_code=completed.returncode,
        timed_out=completed.timed_out,
        visible_window_detected=completed.visible_window_detected,
        residual_process=residual_process,
        authority_count=authority_count,
        executable_sha256_before=before_hash,
        executable_sha256_after=after_hash,
        integer_totals=integer_totals,
        real_totals=real_totals,
        errors=errors,
        log=str(log_path),
    )


def write_summaries(
    output_root: Path,
    results: list[InventoryResult],
    cadence: int,
) -> None:
    integer_totals = {
        field: sum(result.integer_totals[field] for result in results)
        for field in INTEGER_FIELDS
    }
    real_totals = {
        field: format(
            sum(
                Decimal(result.real_totals[field])
                for result in results
            ),
            "f",
        )
        for field in REAL_FIELDS
    }
    payload = {
        "schema": 1,
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "cadence": cadence,
        "readOnlyInventory": True,
        "deformableDeepDiagnosticsExcluded": True,
        "runCount": len(results),
        "passCount": sum(result.passed for result in results),
        "integerTotals": integer_totals,
        "realTotals": real_totals,
        "results": [asdict(result) for result in results],
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    fieldnames = (
        "name",
        "passed",
        "execution",
        "requested_frames",
        "cadence",
        "diagnostic_samples",
        *INTEGER_FIELDS,
        *REAL_FIELDS,
        "errors",
        "log",
    )
    with (output_root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "name": result.name,
                    "passed": int(result.passed),
                    "execution": result.execution,
                    "requested_frames": result.requested_frames,
                    "cadence": result.cadence,
                    "diagnostic_samples": result.diagnostic_samples,
                    **result.integer_totals,
                    **result.real_totals,
                    "errors": " | ".join(result.errors),
                    "log": result.log,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(ALL_SPECS),
        dest="selected_cases",
        help="select a fixed non-deformable cross14 case; may be repeated",
    )
    parser.add_argument(
        "--cadence",
        type=int,
        default=60,
        help="emit one diagnostic sample on frame 1 and each cadence frame",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()

    if args.list_cases:
        for name in ALL_SPECS:
            print(name)
        return 0
    if args.cadence <= 0 or args.timeout <= 0:
        parser.error("--cadence and --timeout must be positive")

    selected_cases = args.selected_cases or list(DEFAULT_CASES)
    output_root = make_output_root(args.output_root)
    bin_dir = args.bin_dir.resolve()
    print(f"ARTIFACT_ROOT={output_root}")
    print(f"RUN_COUNT={len(selected_cases)}")

    results: list[InventoryResult] = []
    for index, name in enumerate(selected_cases, start=1):
        result = run_one(
            bin_dir,
            output_root,
            ALL_SPECS[name],
            args.cadence,
            args.timeout,
        )
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        totals = result.integer_totals
        print(
            f"[{index:02d}/{len(selected_cases):02d}] {status} "
            f"{name} samples={result.diagnostic_samples} "
            f"fallbackRows={totals['bodyStaticFallbackRows']} "
            f"fallbackCorrections="
            f"{totals['bodyStaticFallbackCorrections']} "
            f"materialTangentRows={totals['genericTangentRows']} "
            f"restitutionCorrections={totals['restitutionCorrections']} "
            f"jointRows="
            f"{totals['lockLin'] + totals['limLin'] + totals['lockAng'] + totals['limAng'] + totals['linDrv'] + totals['angDrv'] + totals['cone']}"
        )
        for error in result.errors:
            print(f"  error: {error}")
        if result.visible_window_detected:
            break

    write_summaries(output_root, results, args.cadence)
    passed = sum(result.passed for result in results)
    print(
        f"SUMMARY runs={len(results)} accepted={passed} "
        f"runnerFailures={len(results) - passed}"
    )
    return 0 if passed == len(results) == len(selected_cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
