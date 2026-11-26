"""Measure 4 -> 8 strong scaling in one Checked scene.

Process CPU includes headless observers. Windows uses ``*_64.exe`` under
``physx/bin/win.x86_64.vc143.md``. Linux uses the same cases against
``physx/bin/linux.x86_64/<config>`` (``SnippetDeformable*AVBD`` or ``*_64``).
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics as st
import subprocess

from avbd_benchmark_host import cpu_ticks, process_affinity, runtime_binaries, wait_child


SLOW = [*range(91, 98), *range(163, 168), *range(274, 280)]
CASES = {"dense": ("Surface", "static-box-four-sheets", 150), "cloth": ("Surface", "static-box", 150),
         "five": ("Volume", "visual-showcase", 360)}


def tags(lines, name):
    return [dict(field.split("=", 1) for field in line.split()[1:] if "=" in field)
            for line in lines if line.startswith("[" + name + "] ")]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(sdk):
    files = runtime_binaries(sdk, ("SnippetDeformableSurfaceAVBD", "SnippetDeformableVolumeAVBD"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def check_comparison_binaries(baseline, candidate):
    solver = (r"PhysX_64\.dll" if os.name == "nt" else r"libPhysX(?:_64)?\.so(?:\.[0-9]+)*")
    require(baseline.keys() == candidate.keys(), "Unmatched runtime dependencies or snippets")
    require(any(re.fullmatch(solver, name) for name in baseline), "Missing shared PhysX solver library")
    require(all(value == candidate[name] for name, value in baseline.items()
                if not re.fullmatch(solver, name)), "Unmatched runtime dependencies or snippets")


def run(sdk, out, case, workers, rep, profile, timeout):
    kind, name, frames = CASES[case]
    exe = next((p for p in runtime_binaries(sdk, (f"SnippetDeformable{kind}AVBD",))
                if p.name.startswith(f"SnippetDeformable{kind}AVBD")), None)
    require(exe is not None and exe.is_file(), f"Missing SnippetDeformable{kind}AVBD under {sdk}")
    stem = f"{case}-{workers}-" + ("profile" if profile else str(rep))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PHYSX_AVBD_", "PHYSX_NP_", "PHYSX_SNIPPET_"))}
    env.update(PHYSX_AVBD_COMMON_STEP="budget", PHYSX_AVBD_LOCAL_ITERATIONS="4")
    csvfile = out / (stem + ".csv")
    if profile:
        env.update(PHYSX_AVBD_COMMON_STATS="1", PHYSX_AVBD_LOCAL_PROFILE="1")
    else:
        env["PHYSX_SNIPPET_STEP_TIMES"] = str(csvfile)
    log = out / (stem + ".log")
    with log.open("wb") as stream:
        before = cpu_ticks()
        with subprocess.Popen([str(exe), "--headless", "--case=" + name, f"--frames={frames}",
                               f"--dispatcher-threads={workers}"], env=env, stdout=stream, stderr=subprocess.STDOUT) as child:
            try:
                rc, cpu = wait_child(child, timeout)
            except subprocess.TimeoutExpired:
                raise
        after = cpu_ticks()
    lines = log.read_text(errors="replace").splitlines()
    gates = tags(lines, "AVBD_GATE")
    routes = tags(lines, "AVBD_" + kind.upper() + "_ROUTE")
    require(rc == 0 and len(gates) == len(routes) == 1, f"Incomplete run or failed gate: {log}")
    gate, route = gates[0], routes[0]
    require(gate["status"] == "PASS" and int(gate["completedFrames"]) == frames
            and route["model"] == "particle3" and route["iterations"] == "4", f"Physical gate failed: {log}")
    if case == "dense":
        replicas = tags(lines, "DRAPE_REPLICA")
        require(len(replicas) == 3 and all(r["status"] == "PASS" for r in replicas), str(log))
    if case == "five":
        events = {r["pair"]: r for r in tags(lines, "FIVE_REBOUND")}
        for pair in ("cone-ground", "prism-sphere"):
            require(0 < int(events[pair]["contactFrame"]) < int(events[pair]["departureFrame"]) <= 330, str(log))
    record = dict(case=case, workers=workers, rep=rep, gate=gate,
                  cpuProcessMsPerFrame=cpu / (10000 * frames) if cpu is not None else None,
                  otherBusyPct=(100 * max(0, after[1]-before[1]-cpu) / (after[0]-before[0])) if before else None)
    selected = SLOW if case == "five" else list(range(51, 151))
    if profile:
        phases = tags(lines, "AVBD_COMMON_FRAME")
        require(len(phases) == frames and all(r["iterations"] == "4" and r["bdf"] == r["poseForcePublications"] == "1"
                                              for r in phases), str(log))
        dispatch = tags(lines, "AVBD_LOCAL_DISPATCH")
        require(len(dispatch) == frames and all(int(r["queuePeak"]) <= 4 * workers and int(r["primalMaxRows"]) <= 256
                                               and int(r["primalMaxBlocks"]) <= 32 for r in dispatch), str(log))
        if case == "dense":
            grid, = tags(lines, "DRAPE_GRID")
            nodes = 4 * int(grid["vertices"])
            counts = tags(lines, "AVBD_BLOCK_DOF")
            require(len(counts) == frames and all(int(r["particle3"]) == nodes and int(r["nodalSolves"]) == 4*nodes
                                                  for r in counts), str(log))
            require(all(int(counts[i-1]["mixedContactRows"]) > 0 for i in selected), str(log))
        def values(name, field, **match):
            rows = [r for r in tags(lines, name) if all(r[k] == v for k, v in match.items())]
            require(len(rows) == frames, (name, len(rows), log))
            return [float(rows[i-1][field]) for i in selected]
        def avg(name, field, **match):
            return st.mean(values(name, field, **match))
        span = avg("AVBD_LOCAL_STAGE", "spanMs", stage="0")
        record.update(primalSpanMs=span, primalConcurrency=avg("AVBD_LOCAL_STAGE", "taskMs", stage="0") / span,
                      primalWaves=avg("AVBD_LOCAL_STAGE", "waves", stage="0"), wideJobs=avg("AVBD_LOCAL_TILES", "jobs"),
                      warmMs=avg("AVBD_LOCAL_PROFILE", "warmMs"), compileMs=avg("AVBD_LOCAL_PREPARE", "ms", part="4"),
                      backendMs=avg("AVBD_LOCAL_PROFILE", "backendMs"), colorMs=avg("AVBD_LOCAL_PROFILE", "colorMs"),
                      contacts=avg("AVBD_BLOCK_DOF", "mixedContactRows"), materialRows=avg("AVBD_BLOCK_DOF", "nodalMaterialRows"),
                      particles=avg("AVBD_BLOCK_DOF", "particle3"))
        checks = tags(lines, "AVBD_WARM_CHECKS")
        material = tags(lines, "AVBD_MATERIAL_PACK")
        if material:
            require(len(material) == frames and all(int(r["reusedTopologyRows"]) <= int(r["rows"])
                                                    for r in material), str(log))
            record.update(packedRows=avg("AVBD_MATERIAL_PACK","rows"), reusedTopologyRows=avg("AVBD_MATERIAL_PACK","reusedTopologyRows"),
                          topologyBytes=avg("AVBD_MATERIAL_PACK","topologyBytes"), packSpanMs=avg("AVBD_LOCAL_STAGE","spanMs",stage="9"))
        if checks:
            require(len(checks) == frames and all(int(r["masks"]) == (int(r["checkedRows"])+31)//32
                                                  and int(r["failedRows"]) <= int(r["checkedRows"]) for r in checks), str(log))
            warm_span = avg("AVBD_LOCAL_STAGE", "spanMs", stage="8")
            record.update(warmCheckedRows=avg("AVBD_WARM_CHECKS", "checkedRows"),
                          warmFailedRows=avg("AVBD_WARM_CHECKS", "failedRows"), warmCheckSpanMs=warm_span,
                          warmCheckConcurrency=avg("AVBD_LOCAL_STAGE", "taskMs", stage="8")/warm_span if warm_span else 0)
    else:
        with csvfile.open() as f:
            rows = list(csv.DictReader(f))
        require([int(r["frame"]) for r in rows] == list(range(1, frames + 1)), str(csvfile))
        times = [float(r["simulateFetchMs"]) for r in rows]
        require(all(t > 0 and math.isfinite(t) for t in times), str(csvfile))
        win = [times[i-1] for i in selected]
        record.update(meanMs=st.mean(times), windowMs=st.mean(win), windowP95Ms=sorted(win)[(len(win)-1)*95//100])
    return record


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("sdk", type=Path)
    cli.add_argument("out", type=Path)
    cli.add_argument("--pairs", type=int, default=4)
    cli.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    cli.add_argument("--target", type=float, default=1.8, help="Proposed dense whole-run speedup target.")
    cli.add_argument("--affinity-mask", type=lambda s: int(s, 0), help="Optional common process mask inherited by both worker counts.")
    cli.add_argument("--timeout", type=float, default=300, help="Per-run wall timeout in seconds.")
    cli.add_argument("--compare-sdk", type=Path, help="Interleave a second solver library using identical snippets and dependencies.")
    args = cli.parse_args()
    if args.pairs < 2 or not math.isfinite(args.target) or args.target <= 0:
        cli.error("Use at least two pairs and a positive target.")
    sdk, out = args.sdk.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sdks = {"production": sdk}
    if args.compare_sdk:
        sdks["candidate"] = args.compare_sdk.resolve()
    identities = {arm: digest(path) for arm, path in sdks.items()}
    if args.compare_sdk:
        check_comparison_binaries(identities["production"], identities["candidate"])
    result = dict(binary=identities["production"], runs=[], profiles=[], summary={}, target=args.target,
                  processMask=process_affinity(args.affinity_mask))
    if args.compare_sdk:
        result.update(candidateBinary=identities["candidate"], candidateSummary={})
        (out / "candidate").mkdir()
    def acquire(arm, case, workers, i, profile):
        row = run(sdks[arm], out if arm == "production" else out / arm, case, workers, i, profile, args.timeout)
        return dict(row, arm=arm)
    def save():
        (out / "summary.json").write_text(json.dumps(result, indent=2))
    for i in range(args.pairs):
        for case in (args.cases if i % 2 == 0 else reversed(args.cases)):
            for workers in ((4, 8) if i % 2 == 0 else (8, 4)):
                for arm in (list(sdks) if (i + (workers == 8)) % 2 == 0 else reversed(sdks)):
                    result["runs"].append(acquire(arm, case, workers, i, False))
                    save()
    for case in args.cases:
        for workers in (4, 8):
            for arm in sdks:
                result["profiles"].append(acquire(arm, case, workers, 0, True))
        for arm in sdks:
            ordered = [[r for r in result["runs"] if r["case"] == case and r["workers"] == w and r["arm"] == arm]
                       for w in (4, 8)]
            summary = {}
            for key in ("meanMs", "windowMs", "windowP95Ms", "cpuProcessMsPerFrame"):
                if ordered[0][0][key] is None:
                    continue
                a, b = [[r[key] for r in rows] for rows in ordered]
                ratios = [x/y for x, y in zip(a, b)]
                summary[key] = dict(w4=st.median(a), w8=st.median(b), speedup=st.median(ratios), paired=ratios)
            result["summary" if arm == "production" else "candidateSummary"][case] = summary
        save()
    require(identities == {arm: digest(path) for arm, path in sdks.items()}, "Binaries changed during the measurement")
    if "dense" in args.cases:
        tested = "candidateSummary" if args.compare_sdk else "summary"
        result["denseTargetMet"] = result[tested]["dense"]["meanMs"]["speedup"] >= args.target
    result["complete"] = True
    save()
    print(json.dumps({k: result[k] for k in ("summary", "candidateSummary", "denseTargetMet") if k in result}))
    return 1 if result.get("denseTargetMet") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
