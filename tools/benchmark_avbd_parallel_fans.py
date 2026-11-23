"""Compare two Checked schedulers on 32 independent high-valence roots in a single scene."""
import argparse
import json
import os
from pathlib import Path
import statistics as st
import subprocess

from benchmark_avbd_dense_scaling import digest, require, tags
from benchmark_particle3_showcases import cpu_ticks
from run_avbd_ogc_certificate_probe import build_probe, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-kind", choices=("ready", "current"), default="ready",
                        help="With ready, additionally verify the block-ready tree's diagnostic counters.")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sdks = {"production": args.baseline.resolve(), args.candidate_kind: args.candidate.resolve()}
    binaries = {k: digest(v) for k, v in sdks.items()}
    for name, value in binaries["production"].items():
        if name.endswith(".dll") and name != "PhysX_64.dll":
            require(binaries[args.candidate_kind].get(name) == value, name)
    libraries = ["PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib", "PhysXCooking_64.lib",
                 "PhysX_64.lib", "PhysXCommon_64.lib"]
    link_sdk = sdks["production"] if (sdks["production"]/"PhysX_64.lib").is_file() else sdks[args.candidate_kind]
    require(build_probe(out, link_sdk, ROOT/"tools/avbd_parallel_fans.cpp", libraries=libraries) == 0, "build")
    records = []
    for rep in range(5):  # last pair traces actual active roots and widths
        profile = rep == 4
        for w in ((4, 8) if rep % 2 == 0 else (8, 4)):
            for arm in (list(sdks) if (rep + (w == 8)) % 2 == 0 else reversed(sdks)):
                env = {k: v for k, v in os.environ.items() if not k.startswith(("PHYSX_AVBD_", "PHYSX_NP_", "PHYSX_SNIPPET_"))}
                env.update(PHYSX_AVBD_COMMON_STEP="budget", PHYSX_AVBD_LOCAL_ITERATIONS="4",
                           PATH=str(sdks[arm]) + os.pathsep + env["PATH"])
                if profile:
                    env.update(PHYSX_AVBD_COMMON_STATS="1", PHYSX_AVBD_LOCAL_PROFILE="1")
                log = out / f"{arm}-{w}-{rep}.log"
                with log.open("wb") as f, subprocess.Popen([out/"probe.exe", str(w)], cwd=out, env=env,
                                                             stdout=f, stderr=subprocess.STDOUT) as p:
                    try:
                        rc = p.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        raise
                    cpu = cpu_ticks(p)
                lines = log.read_text(errors="replace").splitlines()
                gate, = tags(lines, "FAN_SCALING")
                require(rc == 0 and gate["status"] == "PASS" and gate["roots"] == "32" and gate["samples"] == "64", str(log))
                row = dict(rep=rep, workers=w, arm=arm, profile=profile, gate=gate, cpuProcessMsPerFrame=cpu/720000)
                if profile:
                    phases = tags(lines, "AVBD_COMMON_FRAME")
                    require(len(phases) == 72 and all(r["iterations"] == "4" and r["bdf"] == r["poseForcePublications"] == "1"
                                                      for r in phases), str(log))
                    tiles = tags(lines, "AVBD_LOCAL_TILES")
                    require(len(tiles) == 72 and all(int(r["maxIncidentRows"]) > 256 for r in tiles), str(log))
                    row.update(maxRows=max(int(r["maxIncidentRows"]) for r in tiles),
                               widePerStep=st.mean(int(r["blocks"]) for r in tiles[8:]),
                               jobs=st.mean(int(r["jobs"]) for r in tiles[8:]))
                    require(row["widePerStep"] == 4*32, str(log))
                    if arm == "ready":
                        ready = tags(lines, "AVBD_READY_TREE")
                        require(len(ready) == 72 and all(int(r["multiBlockCohorts"]) > 0 for r in ready), str(log))
                        row["multiCohorts"] = st.mean(int(r["multiBlockCohorts"]) for r in ready[8:])
                records.append(row)
                (out/"runs.json").write_text(json.dumps(records, indent=2))
    require(binaries == {k: digest(v) for k, v in sdks.items()}, "Binaries changed during the run")
    (out/"identity.json").write_text(json.dumps(binaries, indent=2))
    for arm in sdks:
        values = [[float(r["gate"]["meanMs"]) for r in records if not r["profile"] and r["arm"] == arm and r["workers"] == w]
                  for w in (4, 8)]
        print(arm, [st.median(v) for v in values], st.median(a/b for a, b in zip(*values)))


if __name__ == "__main__":
    main()
