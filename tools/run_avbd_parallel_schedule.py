#!/usr/bin/env python3
"""Check Particle3 physics and concurrency across dispatcher sizes, serially.

Numerical and trajectory differences are diagnostic, never acceptance gates.
Native joint paths use benchmark_avbd_native_parallel.py.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
from run_snippet_avbd_migrations import fields

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("checked", "debug", "release"), default="checked")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or not args.threads or any(n < 1 for n in args.threads):
        parser.error("frames and thread counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    binary = ROOT / "physx/bin/win.x86_64.vc143.md" / args.config
    fingerprints = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [binary / "SnippetDeformableVolumeAVBD_64.exe"] + list(binary.glob("*.dll"))}
    report = {"config": args.config, "frames": args.frames, "before": fingerprints(), "runs": []}
    expected = None
    success = True
    for count in args.threads:
        env = os.environ.copy()
        env["PHYSX_AVBD_SCHEDULE_STATS"] = "1"
        env["PHYSX_AVBD_HEADLESS_PROGRESS"] = "1"
        command = [str(binary / "SnippetDeformableVolumeAVBD_64.exe"), "--particle3", "--headless",
                   "--solver=avbd", "--case=visual-showcase", f"--frames={args.frames}",
                   f"--dispatcher-threads={count}"]
        started = time.perf_counter()
        timed_out = False
        log = args.output / f"workers-{count}.log"
        with log.open("wb") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code, timed_out = None, True
        text = log.read_text(encoding="utf-8", errors="replace")
        gates = [fields(line) for line in text.splitlines() if line.startswith("[AVBD_GATE] ")]
        schedules = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                     for line in text.splitlines() if line.startswith("[AVBD_SCHEDULE] ")]
        gate = gates[0] if len(gates) == 1 else {}
        complete = (not timed_out and gate.get("completedFrames") == str(args.frames)
                    and gate.get("frames") == str(args.frames)
                    and gate.get("snippet") == "SnippetDeformableVolumeAVBD"
                    and gate.get("case") == "visual-showcase" and gate.get("solver") == "avbd"
                    and gate.get("nonFinite") == "0" and gate.get("fatalErrors") == "0"
                    and gate.get("fetchFailures", "0") == "0")
        physical = complete and code == 0 and gate.get("status") == "PASS"
        valid_stats = all(row.get("peakTasks", "").isdigit() and row.get("workers", "").isdigit()
                          for row in schedules)
        peak = max((int(row["peakTasks"]) for row in schedules), default=0) if valid_stats else -1
        concurrency = (valid_stats and len(schedules) == args.frames and
                       all(int(row["workers"]) == count and 0 <= int(row["peakTasks"]) <= count
                           for row in schedules) and (peak > 1 if count > 1 else peak == 0))
        physics = {k: v for k, v in gate.items() if k not in
                   ("averageStepMs", "maxStepMs", "dispatcherThreads", "sceneDispatcherThreads")}
        if expected is None:
            expected = physics
        equal = complete and bool(physics) and physics == expected
        # Each execution must satisfy its own physical contract. Equality of
        # two failed trajectories cannot qualify either run.
        success &= physical and concurrency
        wall = time.perf_counter() - started
        report["runs"].append(dict(threads=count, command=command, exit=code, timeout=timed_out,
                                  wallSeconds=wall, withinOriginalBudget=wall <= 600, gate=gate,
                                  peakTasks=peak, concurrency=concurrency, equal=equal, physical=physical))
        print(f"workers={count} equalityDiagnostic={equal} peakTasks={peak} physical={physical}", flush=True)
    report["after"] = fingerprints()
    success &= report["before"] == report["after"]
    report["comparison"] = "PASS" if success else "FAIL"
    report["acceptancePolicy"] = "parallel-quality-not-bitwise-20261109"
    report["qualification"] = "PASS" if (success and args.frames == 360 and
                                            all(row["physical"] and row["withinOriginalBudget"]
                                                for row in report["runs"])) else "NOT_VERIFIED"
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if success and (args.frames != 360 or report["qualification"] == "PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
