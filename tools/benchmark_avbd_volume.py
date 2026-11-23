#!/usr/bin/env python3
"""Compare two Volume snippet builds serially, including physical gate fields.

Use matching Particle3 inputs with the baseline/candidate solver DLLs. Trajectory
and numerical differences are reported, never required to vanish.
Timings cover simulate + fetchResults for showcase;
other cases report process time, which includes cooking and gate diagnostics.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from run_snippet_avbd_migrations import MATRICES, fields


CASES = {name: frames for name, frames, _ in MATRICES["volume"][1]}
TIMING_FIELDS = {"averageStepMs", "maxStepMs"}


def fingerprint(exe):
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [exe] + sorted(exe.parent.glob("*.dll"))
    }


def run(exe, case, repeat, label, args):
    command = [str(exe), "--particle3", "--headless", "--solver=avbd", "--case=" + case,
               "--frames=" + str(CASES[case]), "--dispatcher-threads=2"]
    env = os.environ.copy()
    env.pop("PHYSX_AVBD_PVD_CAPTURE_PATH", None)
    env.pop("PHYSX_AVBD_HEADLESS_PROGRESS", None)
    env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")
    start = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=exe.parent, env=env,
                                 capture_output=True, timeout=args.timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} {case}: timeout after {args.timeout}s") from exc
    wall_ms = (time.perf_counter() - start) * 1000
    output = process.stdout.decode("utf-8", errors="replace") + "\n" + \
        process.stderr.decode("utf-8", errors="replace")
    log_path = args.output / f"{case}-{repeat}-{label}.log"
    log_path.write_text(output, encoding="utf-8")
    gates = [fields(line) for line in output.splitlines()
             if line.startswith("[AVBD_GATE] ")]
    if len(gates) != 1:
        raise RuntimeError(f"{label} {case}: expected one gate; see {log_path}")
    gate = gates[0]
    expected = "FAIL" if case in args.expected_failures else "PASS"
    valid = (gate.get("status") == expected and gate.get("case") == case
             and gate.get("snippet") == "SnippetDeformableVolumeAVBD" and gate.get("solver") == "avbd"
             and gate.get("frames") == str(CASES[case])
             and gate.get("completedFrames") == str(CASES[case])
             and all(gate.get(key) == "0" for key in ("nonFinite", "fatalErrors"))
             and gate.get("fetchFailures", "0") == "0"
             and process.returncode == (1 if expected == "FAIL" else 0))
    bench = next((fields(line) for line in output.splitlines()
                  if line.startswith("[AVBD_BENCH] ")), {})
    physics = {key: value for key, value in gate.items() if key not in TIMING_FIELDS}
    physics.update({key: value for key, value in bench.items() if key not in TIMING_FIELDS})
    print(f"{case} {label} #{repeat}: {gate.get('status')} "
          f"step={gate.get('averageStepMs', 'n/a')} ms wall={wall_ms:.1f} ms",
          flush=True)
    return dict(command=command, returncode=process.returncode, gate=gate,
                physics=physics, wallMs=wall_ms, valid=valid, log=str(log_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-exe", required=True, type=Path)
    parser.add_argument("--candidate-exe", required=True, type=Path)
    parser.add_argument("--cases", default="visual-showcase")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expected-failures", default="")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selected = list(CASES) if args.cases == "all" else args.cases.split(",")
    args.expected_failures = set(filter(None, args.expected_failures.split(",")))
    if args.repeats < 1 or args.timeout <= 0 or not selected or \
            (set(selected) | args.expected_failures) - CASES.keys():
        parser.error("invalid cases, repeat count, timeout, or expected failures")
    exes = {"baseline": args.baseline_exe.resolve(),
            "candidate": args.candidate_exe.resolve()}
    if any(not exe.is_file() for exe in exes.values()):
        parser.error("both snippet executables must exist")
    if exes["baseline"] == exes["candidate"]:
        parser.error("baseline and candidate must be separate builds")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(platform=platform.platform(), processor=platform.processor(),
                  binaries={label: fingerprint(exe) for label, exe in exes.items()},
                  expectedFailures=sorted(args.expected_failures), cases={})
    ok = True
    try:
        for case in selected:
            pairs = []
            report["cases"][case] = dict(runs=pairs)
            for repeat in range(1, args.repeats + 1):
                # Alternate AB / BA; never run two timed simulations together.
                order = ("baseline", "candidate") if repeat % 2 else ("candidate", "baseline")
                pair = {label: run(exes[label], case, repeat, label, args) for label in order}
                pairs.append(pair)
                before, after = pair["baseline"]["physics"], pair["candidate"]["physics"]
                differences = {key: [before.get(key), after.get(key)]
                               for key in before.keys() | after.keys()
                               if before.get(key) != after.get(key)}
                pair["differences"] = differences
                ok = all(pair[label]["valid"] for label in exes) and ok
                print(f"  field differences (diagnostic only): {differences}", flush=True)
            timing = "averageStepMs" if case == "visual-showcase" else "processWallMs"
            medians = {}
            for label in exes:
                samples = [float(pair[label]["gate"][timing]) if timing == "averageStepMs"
                           else pair[label]["wallMs"] for pair in pairs]
                medians[label] = statistics.median(samples)
            report["cases"][case].update(timing=timing, medianMs=medians,
                                        speedup=medians["baseline"] / medians["candidate"])
            print(f"  {timing}: {medians['baseline']:.3f} -> {medians['candidate']:.3f} ms "
                  f"({medians['baseline'] / medians['candidate']:.2f}x)", flush=True)
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        report["error"] = str(exc)
        print(str(exc), flush=True)
        ok = False
    report["comparisonPassed"] = ok
    report["acceptancePolicy"] = "parallel-quality-not-bitwise-20261109"
    report["qualification"] = "PASS" if ok and not args.expected_failures else "NOT_VERIFIED"
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # A passing comparison may contain explicitly listed existing failed gates.
    print(f"comparison={'PASS' if ok else 'FAIL'} expectedFailures={sorted(args.expected_failures)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
