"""Measure 4 -> 8 strong scaling in one Checked scene. Windows process CPU includes headless observers."""
import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess

from benchmark_particle3_showcases import cpu_ticks


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
    files = list(sdk.glob("*.dll")) + [sdk / f"SnippetDeformable{s}AVBD_64.exe" for s in ("Surface", "Volume")]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def process_affinity(mask):
    if os.name != "nt":
        require(mask is None, "The optional CPU mask requires Windows")
        return None
    dll = ctypes.windll.kernel32
    handle = ctypes.c_void_p(-1)
    current, system = ctypes.c_size_t(), ctypes.c_size_t()
    def read():
        if not dll.GetProcessAffinityMask(handle, ctypes.byref(current), ctypes.byref(system)):
            raise ctypes.WinError()
    read()
    if mask is not None:
        require(mask > 0 and mask & current.value == mask, "CPU mask exceeds process affinity")
        if not dll.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
            raise ctypes.WinError()
        read()
        require(current.value == mask, "Requested affinity unavailable")
    return hex(current.value)


def run(sdk, out, case, workers, rep, profile):
    kind, name, frames = CASES[case]
    exe = sdk / f"SnippetDeformable{kind}AVBD_64.exe"
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
        with subprocess.Popen([exe, "--headless", "--case=" + name, f"--frames={frames}",
                               f"--dispatcher-threads={workers}"], env=env, stdout=stream, stderr=subprocess.STDOUT) as child:
            try:
                rc = child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                child.kill()
                raise
            cpu = cpu_ticks(child)
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
    cli.add_argument("--compare-sdk", type=Path, help="Interleave a second solver DLL using identical snippets and dependencies.")
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
        deps = lambda arm: {name: value for name, value in identities[arm].items() if name != "PhysX_64.dll"}
        require(deps("production") == deps("candidate"), "Unmatched runtime dependencies or snippets")
    result = dict(binary=identities["production"], runs=[], profiles=[], summary={}, target=args.target,
                  processMask=process_affinity(args.affinity_mask))
    if args.compare_sdk:
        result.update(candidateBinary=identities["candidate"], candidateSummary={})
        (out / "candidate").mkdir()
    def acquire(arm, case, workers, i, profile):
        row = run(sdks[arm], out if arm == "production" else out / arm, case, workers, i, profile)
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
