#!/usr/bin/env python3
"""Run the accepted cross-snippet AVBD gates without shell argument parsing.

Each snippet is started from an explicit argv list with ``--headless``, the
``PHYSX_SNIPPET_HEADLESS`` environment sentinel, ``CREATE_NO_WINDOW``, and
``SW_HIDE``, and a kill-on-close Job Object on Windows.  The runner polls for
visible child windows, terminates the process tree, and aborts the matrix if
one appears.  It accepts a result only when there is exactly one canonical
``[AVBD_GATE]`` authority line and its lifecycle, frame, and solver fields match
the frozen cross-gate contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from snippet_headless_process import (
    run_headless_process,
    windows_creation_flags,
    windows_startup_info,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
COMMON_ARGS = (
    "--dt=0.0166666675",
    "--seed=1",
    "--dispatcher-threads=2",
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    executable: str
    snippet: str
    case: str
    execution: str
    frames: int
    capability: str
    validation: str
    extra_args: tuple[str, ...] = ()
    required_fields: tuple[tuple[str, str], ...] = ()
    solver: str = "avbd"
    expected_status: str = "PASS"
    expected_reason: str = "none"


@dataclass
class RunResult:
    name: str
    command: list[str]
    executable_sha256_before: str
    executable_sha256_after: str
    actual_status: str
    actual_reason: str
    exit_code: int | None
    authority_count: int
    timed_out: bool
    visible_window_detected: bool
    visible_window_titles: list[str]
    residual_process: bool
    passed: bool
    errors: list[str]
    log: str


def fields(**values: str) -> tuple[tuple[str, str], ...]:
    return tuple(values.items())


def full_specs() -> list[RunSpec]:
    return [
        RunSpec(
            "joint-passive-parallel",
            "SnippetJoint_64.exe",
            "SnippetJoint",
            "passive",
            "parallel",
            1400,
            "SUPPORTED",
            "GATED",
            required_fields=fields(
                fetchFailures="0", fetchErrorState="0", launchFailures="0"
            ),
        ),
        RunSpec(
            "joint-impact-all-sequential",
            "SnippetJoint_64.exe",
            "SnippetJoint",
            "impact-all",
            "sequential",
            360,
            "SUPPORTED",
            "GATED",
            required_fields=fields(
                fetchFailures="0",
                fetchErrorState="0",
                launchFailures="0",
                expectedHits="5",
                hitChains="5",
            ),
        ),
        RunSpec(
            "avbd-articulation-full-parallel",
            "SnippetAvbdArticulation_64.exe",
            "SnippetAvbdArticulation",
            "full-suite",
            "parallel",
            5040,
            "SUPPORTED",
            "ACCEPTED",
            required_fields=fields(
                simulateFailures="0",
                fetchFailures="0",
                fetchPending="0",
                fetchErrorState="0",
                runtimeInvariantFailed="0",
                initializationFailed="0",
                cleanupFailed="0",
                oraclePass="1",
                checkRatio="31/31",
            ),
        ),
        RunSpec(
            "avbd-articulation-full-sequential",
            "SnippetAvbdArticulation_64.exe",
            "SnippetAvbdArticulation",
            "full-suite",
            "sequential",
            5040,
            "SUPPORTED",
            "ACCEPTED",
            required_fields=fields(
                simulateFailures="0",
                fetchFailures="0",
                fetchPending="0",
                fetchErrorState="0",
                runtimeInvariantFailed="0",
                initializationFailed="0",
                cleanupFailed="0",
                oraclePass="1",
                checkRatio="31/31",
            ),
        ),
        RunSpec(
            "articulation-rc-cycle-parallel",
            "SnippetArticulationRC_64.exe",
            "SnippetArticulationRC",
            "scissor-cycle",
            "parallel",
            3600,
            "SUPPORTED",
            "ACCEPTED",
            required_fields=fields(
                simulateFailures="0",
                fetchFailures="0",
                fetchPending="0",
                fetchErrorState="0",
                runtimeInvariantFailed="0",
                cleanupFailed="0",
                oraclePass="1",
                phaseRegressionFailed="0",
            ),
        ),
        RunSpec(
            "articulation-rc-cycle-sequential",
            "SnippetArticulationRC_64.exe",
            "SnippetArticulationRC",
            "scissor-cycle",
            "sequential",
            3600,
            "SUPPORTED",
            "ACCEPTED",
            required_fields=fields(
                simulateFailures="0",
                fetchFailures="0",
                fetchPending="0",
                fetchErrorState="0",
                runtimeInvariantFailed="0",
                cleanupFailed="0",
                oraclePass="1",
                phaseRegressionFailed="0",
            ),
        ),
        RunSpec(
            "hello-world-stack",
            "SnippetHelloWorld_64.exe",
            "SnippetHelloWorld",
            "stack-settle",
            "parallel",
            600,
            "SUPPORTED",
            "GATED",
            required_fields=fields(
                fetchFailures="0",
                fetchErrorState="0",
                launchFailures="0",
                sunkBodies="0",
            ),
        ),
        RunSpec(
            "hello-world-ball",
            "SnippetHelloWorld_64.exe",
            "SnippetHelloWorld",
            "ball-shot",
            "parallel",
            600,
            "SUPPORTED",
            "GATED",
            required_fields=fields(
                fetchFailures="0",
                fetchErrorState="0",
                launchFailures="0",
                sunkBodies="0",
            ),
        ),
        RunSpec(
            "deformable-mesh-stack",
            "SnippetDeformableMesh_64.exe",
            "SnippetDeformableMesh",
            "moving-mesh-stack",
            "parallel",
            7200,
            "SUPPORTED",
            "GATED",
            required_fields=fields(
                simulateFailures="0",
                fetchFailures="0",
                fetchErrorState="0",
                cleanupCompleted="1",
                maxFullFallThroughBodies="0",
                settledSunkBoxes="0",
                softBody="0",
                cloth="0",
            ),
        ),
        RunSpec(
            "deformable-mesh-sphere",
            "SnippetDeformableMesh_64.exe",
            "SnippetDeformableMesh",
            "sphere-shot",
            "parallel",
            180,
            "SUPPORTED",
            "GATED",
            required_fields=fields(
                simulateFailures="0",
                fetchFailures="0",
                fetchErrorState="0",
                cleanupCompleted="1",
                maxFullFallThroughBodies="0",
                sphereFirstContactObserved="1",
                softBody="0",
                cloth="0",
            ),
        ),
        RunSpec(
            "gear-steady-parallel",
            "SnippetGearJoint_64.exe",
            "SnippetGearJoint",
            "steady",
            "parallel",
            1200,
            "PARTIAL",
            "PROBE",
            required_fields=fields(
                fetchFailures="0",
                fetchErrorState="0",
                cleanupComplete="1",
                topologyOk="1",
            ),
        ),
        RunSpec(
            "gear-steady-sequential",
            "SnippetGearJoint_64.exe",
            "SnippetGearJoint",
            "steady",
            "sequential",
            1200,
            "PARTIAL",
            "PROBE",
            required_fields=fields(
                fetchFailures="0",
                fetchErrorState="0",
                cleanupComplete="1",
                topologyOk="1",
            ),
        ),
        RunSpec(
            "serialization-spherical-binary",
            "SnippetSerialization_64.exe",
            "SnippetSerialization",
            "spherical-chain",
            "parallel",
            240,
            "SUPPORTED",
            "GATED",
            ("--format=binary", "--cycles=1"),
            fields(
                format="binary",
                fetchFailures="0",
                serializationGate="PASS",
                structureGate="PASS",
                physicsGate="PASS",
                lifecycleGate="PASS",
            ),
        ),
        RunSpec(
            "serialization-spherical-xml",
            "SnippetSerialization_64.exe",
            "SnippetSerialization",
            "spherical-chain",
            "parallel",
            240,
            "SUPPORTED",
            "GATED",
            ("--format=xml", "--cycles=1"),
            fields(
                format="xml",
                fetchFailures="0",
                serializationGate="PASS",
                structureGate="PASS",
                physicsGate="PASS",
                lifecycleGate="PASS",
            ),
        ),
    ]


def gear_external_specs() -> list[RunSpec]:
    required = fields(
        fetchFailures="0",
        fetchErrorState="0",
        cleanupComplete="1",
        topologyOk="1",
        impulseEvents="1",
        driveEnabledReadback="0",
        impulseResponseSamples="4",
    )
    return [
        RunSpec(
            "gear-external-tgs-parallel",
            "SnippetGearJoint_64.exe",
            "SnippetGearJoint",
            "external-impulse",
            "parallel",
            1200,
            "PARTIAL",
            "PROBE",
            required_fields=required,
            solver="tgs",
        ),
        RunSpec(
            "gear-external-avbd-parallel",
            "SnippetGearJoint_64.exe",
            "SnippetGearJoint",
            "external-impulse",
            "parallel",
            1200,
            "PARTIAL",
            "PROBE",
            required_fields=required,
        ),
        RunSpec(
            "gear-external-avbd-sequential",
            "SnippetGearJoint_64.exe",
            "SnippetGearJoint",
            "external-impulse",
            "sequential",
            1200,
            "PARTIAL",
            "PROBE",
            required_fields=required,
        ),
    ]


def gear_physical_specs() -> list[RunSpec]:
    common = fields(
        fetchFailures="0",
        fetchErrorState="0",
        cleanupComplete="1",
        topologyOk="1",
    )
    external = fields(
        fetchFailures="0",
        fetchErrorState="0",
        cleanupComplete="1",
        topologyOk="1",
        impulseEvents="1",
        driveEnabledReadback="0",
        impulseResponseSamples="4",
    )
    variants = (
        ("steady-positive", "steady", ("--ratio=2.5",), common),
        ("steady-negative", "steady", ("--ratio=-2.5",), common),
        ("unit-ratio", "unit-ratio", (), common),
        ("phase-offset", "phase-offset", (), common),
        ("reverse", "reverse", (), common),
        ("sinusoidal", "sinusoidal", (), common),
        ("external", "external-impulse", (), external),
    )
    lanes = (
        ("tgs-parallel", "tgs", "parallel"),
        ("avbd-parallel", "avbd", "parallel"),
        ("avbd-sequential", "avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for variant_name, case_name, extra_args, required in variants:
        for lane_name, solver, execution in lanes:
            specs.append(
                RunSpec(
                    f"gear-{variant_name}-{lane_name}",
                    "SnippetGearJoint_64.exe",
                    "SnippetGearJoint",
                    case_name,
                    execution,
                    1200,
                    "PARTIAL",
                    "PROBE",
                    extra_args=extra_args,
                    required_fields=required,
                    solver=solver,
                )
            )
    return specs


SUITES = {
    "cross14": full_specs,
    "gear-external3": gear_external_specs,
    "gear-physical21": gear_physical_specs,
}


def parse_authority(line: str) -> tuple[dict[str, str], list[str]]:
    parsed: dict[str, str] = {}
    errors: list[str] = []
    for token in line.split()[1:]:
        if "=" not in token:
            errors.append(f"malformed authority token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in parsed:
            errors.append(f"duplicate authority key: {key}")
        parsed[key] = value
    return parsed, errors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def process_is_running(image_name: str, creationflags: int) -> bool:
    if os.name != "nt":
        return False
    check = subprocess.run(
        ["tasklist.exe", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=creationflags,
        startupinfo=windows_startup_info(),
        shell=False,
    )
    return image_name.casefold() in check.stdout.casefold()


def make_command(executable: Path, spec: RunSpec) -> list[str]:
    return [
        str(executable),
        "--headless",
        f"--solver={spec.solver}",
        *COMMON_ARGS,
        f"--case={spec.case}",
        f"--execution={spec.execution}",
        f"--frames={spec.frames}",
        *spec.extra_args,
    ]


def run_one(
    bin_dir: Path,
    output_root: Path,
    spec: RunSpec,
    timeout_seconds: float,
) -> RunResult:
    executable = bin_dir / spec.executable
    argv = make_command(executable, spec)
    log_path = output_root / f"{spec.name}.log"
    environment = os.environ.copy()
    environment["PHYSX_SNIPPET_HEADLESS"] = "1"
    creationflags = windows_creation_flags()
    before_hash = sha256(executable)
    timed_out = False
    exit_code: int | None = None
    stdout = ""
    stderr = ""
    completed = run_headless_process(
        argv,
        cwd=bin_dir,
        env=environment,
        timeout_seconds=timeout_seconds,
    )
    exit_code = completed.returncode
    stdout = completed.stdout
    stderr = completed.stderr
    timed_out = completed.timed_out

    authority_lines = [
        line.strip()
        for line in stdout.splitlines()
        if line.startswith("[AVBD_GATE] ")
    ]
    authority: dict[str, str] = {}
    errors: list[str] = []
    if timed_out:
        errors.append(f"timeout after {timeout_seconds:g} seconds")
    if completed.visible_window_detected:
        errors.append(
            "visible child window detected; process tree terminated: "
            + ", ".join(completed.visible_window_titles)
        )
    if len(authority_lines) != 1:
        errors.append(f"authority count is {len(authority_lines)}, expected 1")
    else:
        authority, parse_errors = parse_authority(authority_lines[0])
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
    for key, value in expected.items():
        if authority.get(key) != value:
            errors.append(
                f"{key}={authority.get(key, 'MISSING')}, expected {value}"
            )
    if authority.get("case") == "config-error":
        errors.append("headless invocation was rejected as config-error")
    expected_exit_code = 0 if spec.expected_status == "PASS" else 1
    if exit_code != expected_exit_code:
        errors.append(
            f"exit code {exit_code}, expected {expected_exit_code}"
        )
    if stderr:
        errors.append(f"stderr is not empty ({len(stderr.encode('utf-8'))} bytes)")

    after_hash = sha256(executable)
    if after_hash != before_hash:
        errors.append("executable SHA-256 changed during the run")
    residual_process = process_is_running(spec.executable, creationflags)
    if residual_process:
        errors.append(f"residual process detected: {spec.executable}")

    log_text = (
        f"COMMAND: {subprocess.list2cmdline(argv)}\n"
        "HEADLESS_ENV: PHYSX_SNIPPET_HEADLESS=1\n"
        f"CREATE_NO_WINDOW: {int(os.name == 'nt')}\n"
        f"STARTUPINFO_SW_HIDE: {int(os.name == 'nt')}\n"
        f"KILL_ON_JOB_CLOSE: {int(os.name == 'nt')}\n"
        f"VISIBLE_WINDOW_DETECTED: {int(completed.visible_window_detected)}\n"
        "VISIBLE_WINDOW_TITLES: "
        + (", ".join(completed.visible_window_titles) or "none")
        + "\n"
        f"EXECUTABLE_SHA256_BEFORE: {before_hash}\n"
        f"EXECUTABLE_SHA256_AFTER: {after_hash}\n"
        f"EXIT_CODE: {exit_code}\n"
        f"TIMED_OUT: {int(timed_out)}\n"
        f"RESIDUAL_PROCESS: {int(residual_process)}\n"
        "--- STDOUT ---\n"
        f"{stdout}"
        "\n--- STDERR ---\n"
        f"{stderr}"
        "\n--- RUNNER ERRORS ---\n"
        + ("\n".join(errors) if errors else "none")
        + "\n"
    )
    log_path.write_text(log_text, encoding="utf-8")
    return RunResult(
        name=spec.name,
        command=argv,
        executable_sha256_before=before_hash,
        executable_sha256_after=after_hash,
        actual_status=authority.get("status", "MISSING"),
        actual_reason=authority.get("reason", "missing"),
        exit_code=exit_code,
        authority_count=len(authority_lines),
        timed_out=timed_out,
        visible_window_detected=completed.visible_window_detected,
        visible_window_titles=list(completed.visible_window_titles),
        residual_process=residual_process,
        passed=not errors,
        errors=errors,
        log=str(log_path),
    )


def make_output_root(requested: Path | None) -> Path:
    if requested is not None:
        root = requested.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        root = Path(tempfile.gettempdir()) / f"PhysX_AVBD_cross_headless_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--suite",
        action="append",
        choices=tuple(SUITES),
        help="Select a named suite; may be repeated (default: cross14).",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="selected_cases",
        help="Run only the named runner case from selected suites; may be repeated.",
    )
    parser.add_argument("--list-cases", action="store_true")
    options = parser.parse_args()
    suite_names = options.suite or ["cross14"]
    specs = []
    for suite_name in suite_names:
        specs.extend(SUITES[suite_name]())

    if options.list_cases:
        for spec in specs:
            print(spec.name)
        return 0
    if options.timeout <= 0:
        parser.error("--timeout must be positive")
    bin_dir = options.bin_dir.resolve()
    if not bin_dir.is_dir():
        parser.error(f"binary directory does not exist: {bin_dir}")
    if options.selected_cases:
        by_name = {spec.name: spec for spec in specs}
        unknown = sorted(set(options.selected_cases) - set(by_name))
        if unknown:
            parser.error(f"unknown runner case(s): {', '.join(unknown)}")
        specs = [by_name[name] for name in options.selected_cases]
    for spec in specs:
        executable = bin_dir / spec.executable
        if not executable.is_file():
            parser.error(f"executable does not exist: {executable}")

    output_root = make_output_root(options.output_root)
    manifest = {
        "schema": 1,
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "repoRoot": str(REPO_ROOT),
        "binDir": str(bin_dir),
        "headlessEnvironment": "PHYSX_SNIPPET_HEADLESS=1",
        "createNoWindow": os.name == "nt",
        "startupInfoSwHide": os.name == "nt",
        "killOnJobClose": os.name == "nt",
        "visibleWindowGuard": os.name == "nt",
        "shell": False,
        "runCount": len(specs),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"ARTIFACT_ROOT={output_root}", flush=True)
    print(f"RUN_COUNT={len(specs)}", flush=True)

    results: list[RunResult] = []
    for index, spec in enumerate(specs, start=1):
        result = run_one(bin_dir, output_root, spec, options.timeout)
        results.append(result)
        outcome = "OK" if result.passed else "BAD"
        print(
            f"[{index:02d}/{len(specs):02d}] {outcome} {spec.name} "
            f"status={result.actual_status} reason={result.actual_reason} "
            f"exit={result.exit_code} residual={int(result.residual_process)}",
            flush=True,
        )
        for error in result.errors:
            print(f"  {error}", flush=True)
        if result.visible_window_detected:
            print("ABORTED: visible snippet window detected", flush=True)
            break

    (output_root / "summary.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    with (output_root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "name",
                "actual_status",
                "actual_reason",
                "exit_code",
                "authority_count",
                "timed_out",
                "visible_window_detected",
                "residual_process",
                "passed",
                "log",
            ),
        )
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({key: row[key] for key in writer.fieldnames})

    failures = sum(not result.passed for result in results)
    print(
        f"SUMMARY runs={len(results)} accepted={len(results) - failures} "
        f"runnerFailures={failures}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
