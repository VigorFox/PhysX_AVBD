#!/usr/bin/env python3
"""Measure native AVBD joint scenes across worker counts without equality gates.

Run simulations serially, alternate thread-count order between repetitions,
and report each physical gate independently. Process CPU / wall time measures
average occupied logical cores across setup, physics, observer and teardown;
it is not solver-only utilization. Diagnostic logging is opt-in.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

from avbd_gate import parse_fields

ROOT = Path(__file__).resolve().parents[1]
EXES = {"surface": "SnippetDeformableSurfaceAVBD_64.exe",
        "volume": "SnippetDeformableVolumeAVBD_64.exe"}


def telemetry_fields(line):
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def process_cpu_seconds(process):
    if os.name != "nt":
        return None  # No guessed CPU utilization on unsupported platforms.
    from ctypes import wintypes
    get_times = ctypes.WinDLL("kernel32", use_last_error=True).GetProcessTimes
    get_times.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    get_times.restype = wintypes.BOOL
    times = [wintypes.FILETIME() for _ in range(4)]
    if not get_times(int(process._handle), *(ctypes.byref(t) for t in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    return sum((t.dwHighDateTime << 32) + t.dwLowDateTime for t in times[2:]) / 1e7


def physical_pass(gate, code, timed_out, frames, case, snippet):
    model = ("rigid_patch_joint_sheet" if snippet == "surface" else
             "convex_cell_joint_volume" if case == "visual-showcase" else "rigid_cell_joint_volume")
    return (not timed_out and code == 0 and gate.get("status") == "PASS"
            and gate.get("frames") == str(frames) and gate.get("completedFrames") == str(frames)
            and gate.get("case") == case and gate.get("solver") == "avbd"
            and gate.get("snippet") == ("SnippetDeformableSurfaceAVBD" if snippet == "surface" else "SnippetJointSoftBody")
            and gate.get("schema") == "1"
            and gate.get("model") == model
            and all(gate.get(k) == "0" for k in ("nonFinite", "fetchFailures", "fatalErrors")))


def run(exe, args, workers, repeat):
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ("PHYSX_AVBD_", "PHYSX_NP_GEOMETRY_", "PHYSX_SNIPPET_"))}
    budget = getattr(args, "budget", None)
    reference = getattr(args, "reference_bounded", False)
    if reference:
        env["PHYSX_AVBD_COMMON_STEP"] = "bounded"
    elif budget is not None:
        env.update(PHYSX_AVBD_COMMON_STEP="budget", PHYSX_AVBD_LOCAL_ITERATIONS=str(budget))
    if args.diagnostics:
        env["PHYSX_AVBD_COMMON_STATS"] = "summary"
        env["PHYSX_AVBD_BOUNDED_PROFILE"] = "1"
        env["PHYSX_AVBD_HEADLESS_PROGRESS"] = "1"
    command = [str(exe), "--headless", "--solver=avbd", "--case=" + args.case,
               f"--frames={args.frames}", f"--dispatcher-threads={workers}"]
    if not getattr(args, "legacy_rigid6_default", False):
        command.append("--rigid6")
    if getattr(args, "grid", None) is not None:
        command.append(f"--grid={args.grid}")
    log = args.output / f"workers-{workers}-repeat-{repeat}.log"
    started = time.perf_counter()
    with log.open("wb") as stream:
        process = subprocess.Popen(command, cwd=exe.parent, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        requested_cpus = getattr(args, 'cpus', None)
        inherited_cpus = None
        if requested_cpus:
            import psutil
            inherited_cpus = psutil.Process(process.pid).cpu_affinity()
        timed_out = False
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
            timed_out = True
        wall = time.perf_counter() - started
        cpu = process_cpu_seconds(process)
    lines = log.read_text(encoding="utf8", errors="replace").splitlines()
    gates = [parse_fields(line) for line in lines if line.startswith("[AVBD_GATE] ")]
    gate = gates[0][0] if len(gates) == 1 and not gates[0][1] else {}
    route = [telemetry_fields(line) for line in lines if line.startswith("[AVBD_" + args.snippet.upper() + "_ROUTE] ")]
    native_route = (len(route) == 1 and route[0].get("commonStep") == ("bounded" if reference else "budget")
                    and route[0].get("workers") == str(workers)
                    and gate.get("dispatcherThreads") == str(workers))
    if requested_cpus:
        native_route = native_route and sorted(inherited_cpus) == sorted(requested_cpus)
    if getattr(args, "grid", None) is not None:
        native_route = native_route and gate.get("grid") == str(args.grid)
    step = float(gate.get("averageStepMs", "nan"))
    finite_time = math.isfinite(step) and step > 0
    schedules = [telemetry_fields(line) for line in lines if line.startswith("[AVBD_BOUNDED_SCHEDULE] ")]
    result = dict(workers=workers, repeat=repeat, command=command, exit=code, timedOut=timed_out,
                  wallSeconds=wall, processCpuSeconds=cpu,
                  averageOccupiedCores=cpu / wall if cpu is not None else None,
                  averageStepMs=step if finite_time else None, gate=gate, nativeRoute=native_route,
                  logicalCpus=inherited_cpus, physicalPass=native_route and finite_time and physical_pass(
                      gate, code, timed_out, args.frames, args.case, args.snippet), log=str(log))
    if args.diagnostics:
        local = [telemetry_fields(line) for line in lines if line.startswith("[AVBD_LOCAL_BUDGET] ")
                 and "iterations=" in line]
        if not reference:
            expected_budget = budget if budget is not None else 4
            result["localBudget"] = expected_budget
            result["localLedger"] = {key: sum(int(s.get(key, 0)) for s in local)
                                     for key in ("iterations", "primalBlocks", "dualRows", "tasks",
                                                 "queryRows", "geometryQueries", "geometryPointVisits", "transportedProofs", "reselectedProofs", "warmQueries")}
            result["budgetRespected"] = bool(local) and all(
                int(s.get("budget", -1)) == expected_budget and 0 < int(s["iterations"]) <= expected_budget for s in local)
            result["physicalPass"] = result["physicalPass"] and result["budgetRespected"]
            groups = [telemetry_fields(line) for line in lines if line.startswith("[AVBD_LOCAL_GROUPS] ")]
            result["localGroups"] = {key: sum(int(s.get(key, 0)) for s in groups)
                for key in ("primalGroups", "factorizationAttempts", "acceptedGroups", "materialEvaluations", "materialEndpointUses")}
            admission = [telemetry_fields(line) for line in lines if line.startswith("[AVBD_LOCAL_ADMISSION] ")]
            result["localAdmission"] = {key: sum(int(s.get(key, 0)) for s in admission)
                for key in ("primalQueries", "primalPointVisits", "searchTrials", "exhaustedSearches")}
            tiles = [telemetry_fields(line) for line in lines if line.startswith("[AVBD_LOCAL_TILES] ")]
            result["localTiles"] = {key: sum(int(s.get(key, 0)) for s in tiles)
                for key in ("blocks", "leaves", "jobs", "waves")}
            result["localTiles"].update({key: max((int(s.get(key, 0)) for s in tiles), default=0)
                for key in ("maxIncidentRows", "scratchBytes")})
        result["schedule"] = {key: sum(int(s[key]) for s in schedules)
                              for key in ("waves", "batches", "tasks", "sweeps")}
        result["accepted"] = sum(l.startswith("[AVBD_COMMON_FRAME] status=ACCEPTED") for l in lines)
        result["rejected"] = sum(l.startswith("[AVBD_COMMON_FRAME] status=REJECTED") for l in lines)
        result["nonstationary"] = sum(l.startswith("[AVBD_COMMON_FRAME] status=ACCEPTED")
                                      and "stationary=0" in l for l in lines)
        profiles = [telemetry_fields(line) for line in lines if line.startswith("[AVBD_BOUNDED_PROFILE] ")]
        if profiles:
            result["profile"] = {key: sum(float(p[key]) for p in profiles) for key in profiles[0] if key != "jobMaxMs"}
            result["profile"]["jobMaxMs"] = max(float(p["jobMaxMs"]) for p in profiles)
            result["profile"]["frames"] = len(profiles)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snippet", choices=EXES, required=True)
    parser.add_argument("--case", default="free-flight")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--grid", type=int, help="Explicit asset resolution; omitted keeps the snippet default")
    parser.add_argument("--config", choices=("checked", "release", "debug"), default="checked")
    parser.add_argument("--binary-dir", type=Path, help="Isolated build snapshot; keep its configuration explicit")
    parser.add_argument("--legacy-rigid6-default", action="store_true", help="Use an older Rigid6 build that predates --rigid6")
    parser.add_argument("--threads", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--diagnostics", action="store_true")
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--budget", type=int, help="Override default finite local iteration count (1..128; native snippet default 4)")
    route.add_argument("--reference-bounded", action="store_true", help="Explicit slow numerical reference; never a default or fallback")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames is None:
        args.frames = 60 if args.snippet == "surface" and args.case == "free-flight" else 360
    if not 1 <= args.frames <= 10000 or args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 \
            or any(n < 0 or n > 16 for n in args.threads) or len(set(args.threads)) != len(args.threads) \
            or (args.grid is not None and not 2 <= args.grid <= 32) \
            or (args.budget is not None and not 1 <= args.budget <= 128):
        parser.error("invalid frames, repeats, timeout, or worker counts (0..16, unique)")
    binary = args.binary_dir.resolve() if args.binary_dir else ROOT / "physx/bin/win.x86_64.vc143.md" / args.config
    exe = binary / EXES[args.snippet]
    if not exe.is_file():
        parser.error("missing executable: " + str(exe))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    fingerprint = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [exe] + sorted(exe.parent.glob("*.dll"))}
    report = dict(acceptancePolicy="parallel-quality-not-bitwise-20261109", snippet=args.snippet,
                  route="bounded-reference" if args.reference_bounded else "budget",
                  localBudget=None if args.reference_bounded else (args.budget if args.budget is not None else 4),
                  case=args.case, frames=args.frames, grid=args.grid, config=args.config, diagnostics=args.diagnostics,
                  logicalProcessors=os.cpu_count(), processor=platform.processor(), platform=platform.platform(),
                  before=fingerprint(), runs=[], performanceQualification="NOT_ASSESSED")
    try:
        for repeat in range(1, args.repeats + 1):
            for workers in (args.threads if repeat % 2 else list(reversed(args.threads))):
                result = run(exe, args, workers, repeat)
                report["runs"].append(result)
                (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
                print(f"workers={workers} repeat={repeat} physical={result['physicalPass']} "
                      f"stepMs={result['averageStepMs']} occupiedCores={result['averageOccupiedCores']}", flush=True)
    except (OSError, ValueError) as exc:
        report["error"] = str(exc)
    report["after"] = fingerprint()
    summary = {}
    for workers in args.threads:
        runs = [r for r in report["runs"] if r["workers"] == workers]
        # Failed runs stay in the report and cannot supply a qualified speedup.
        qualified = len(runs) == args.repeats and all(r["physicalPass"] for r in runs)
        cpu = [r["averageOccupiedCores"] for r in runs if r["averageOccupiedCores"] is not None]
        summary[str(workers)] = dict(allPhysicalPass=qualified, runs=len(runs),
            medianStepMs=statistics.median(r["averageStepMs"] for r in runs) if qualified else None,
            medianOccupiedCores=statistics.median(cpu) if cpu else None)
    reference = summary.get("0", summary.get("1", {})).get("medianStepMs")
    for value in summary.values():
        value["stepSpeedup"] = reference / value["medianStepMs"] if reference and value["medianStepMs"] else None
    report["summary"] = summary
    passed = (len(report["runs"]) == args.repeats * len(args.threads) and
              all(r["physicalPass"] for r in report["runs"]) and report["before"] == report["after"] and "error" not in report)
    report["sceneQuality"] = "PASS" if passed else "FAIL"
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
