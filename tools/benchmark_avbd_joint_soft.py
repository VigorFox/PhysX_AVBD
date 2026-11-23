#!/usr/bin/env python3
"""Serial public-API AVBD rigid-cell/FEM comparison; preserve failed samples."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from avbd_gate import parse_fields
from run_avbd_ogc_certificate_probe import ROOT, VCVARS

CASES = ("free-flight", "drop", "compression-load", "showcase")
TIMING_FIELDS = ("initMs", "averageStepMs", "averageActiveMs",
                 "averageGroundedActiveMs", "holdAverageMs", "impactWindowAverageMs",
                 "activeP95Ms", "maxStepMs")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quote(path):
    value = str(path)
    if any(char in value for char in ('"', '\n', '\r')):
        raise ValueError("Unsupported character in build path")
    return '"' + value + '"'


def build(config, output):
    work = output / ("build-" + config)
    work.mkdir(parents=True, exist_ok=True)
    binary = work / "avbd_joint_soft_benchmark.exe"
    bin_dir = ROOT / "physx/bin/win.x86_64.vc143.md" / config
    source = ROOT / "tools/avbd_joint_soft_benchmark.cpp"
    libraries = ("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
                 "PhysXCooking_64.lib", "PhysX_64.lib", "PhysXCommon_64.lib",
                 "PhysXFoundation_64.lib")
    command = work / "build.cmd"
    command.write_text(
        '@echo off\ncall %s\nif errorlevel 1 exit /b 2\n' % quote(VCVARS) +
        'cl /nologo /O2 /MD /EHsc /std:c++14 /DNDEBUG /DPX_CHECKED=%d '
        '/DPX_SUPPORT_PVD=1 /DPX_PUBLIC_RELEASE=1 /DDISABLE_CUDA_PHYSX '
        '/D_CRT_SECURE_NO_WARNINGS /D_ITERATOR_DEBUG_LEVEL=0 '
        '/I%s /I%s %s /Fe:%s /link %s\n' % (
            config == "checked", quote(ROOT / "physx/include"),
            quote(ROOT / "physx/source/geomutils/include"), quote(source), quote(binary),
            " ".join(quote(bin_dir / lib) for lib in libraries)), encoding="ascii")
    (work / source.name).write_bytes(source.read_bytes())
    proc = subprocess.run(["cmd", "/c", str(command)], cwd=work, capture_output=True)
    (work / "build.log").write_bytes(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError("Build failed: " + str(work / "build.log"))
    fingerprint = {path.name: digest(path) for path in
                   [binary] + sorted(bin_dir.glob("*.dll"))}
    return binary, bin_dir, fingerprint


def run(binary, bin_dir, config, grid, workers, case, model, repeat, args):
    argv = [str(binary), "--model=" + model, "--case=" + case,
            "--grid=" + str(grid), "--dispatcher-threads=" + str(workers),
            "--frames=" + str(args.frames)]
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("PHYSX_AVBD_") or name.startswith("PHYSX_SNIPPET_"):
            env.pop(name)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    start = time.perf_counter()
    try:
        proc = subprocess.run(argv, cwd=bin_dir, env=env, capture_output=True, timeout=args.timeout)
        output = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
        returncode = proc.returncode
    except subprocess.TimeoutExpired as error:
        output = ((error.stdout or b"") + (error.stderr or b"")).decode("utf-8", errors="replace")
        returncode = "timeout"
    stem = "%s-g%d-w%d-%s-%s-r%d" % (config, grid, workers, case, model, repeat)
    (args.output / (stem + ".log")).write_text(output, encoding="utf-8")
    lines = [line for line in output.splitlines() if line.startswith("[AVBD_GATE] ")]
    fields, errors = parse_fields(lines[0]) if len(lines) == 1 else ({}, ["Expected one benchmark record"])
    for key, value in {"model": model, "case": case, "solver": "avbd", "grid": grid,
                       "constitutiveModel": "stable_neo_hookean" if model == "fem" else "face_beam_lattice",
                       "workers": workers, "frames": args.frames, "completedFrames": args.frames}.items():
        if fields.get(key) != str(value):
            errors.append(key + " mismatch")
    for key in TIMING_FIELDS:
        try:
            if not 0 <= float(fields[key]) < float("inf"):
                errors.append(key + " non-finite or negative")
        except (KeyError, ValueError):
            errors.append(key + " missing or invalid")
    ok = returncode == 0 and fields.get("status") == "PASS" and not errors
    print("%s status=%s active=%s grounded=%s hold=%s ms" % (
        stem, fields.get("status", returncode), fields.get("averageActiveMs", "n/a"),
        fields.get("averageGroundedActiveMs", "n/a"), fields.get("holdAverageMs", "n/a")), flush=True)
    return {"argv": argv, "exit": returncode, "wallSeconds": time.perf_counter() - start,
            "fields": fields, "errors": errors, "physicsPassed": ok, "log": stem + ".log"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="checked")
    parser.add_argument("--grids", default="5")
    parser.add_argument("--workers", default="4")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs, cases = args.configs.split(","), args.cases.split(",")
    try:
        grids, workers = [int(s) for s in args.grids.split(",")], [int(s) for s in args.workers.split(",")]
    except ValueError:
        parser.error("grids/workers must be integer lists")
    if (not configs or set(configs) - {"checked", "release"} or not cases or set(cases) - set(CASES)
            or not grids or any(not 3 <= g <= 10 for g in grids)
            or not workers or any(not 0 <= w <= 16 for w in workers)
            or args.frames < 360 or args.repeats < 1 or args.timeout <= 0):
        parser.error("Invalid configuration, cases, grid, worker count, frames or repeats")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / "tools/avbd_joint_soft_benchmark.cpp",
               ROOT / "physx/snippets/snippetjointsoftbody/SnippetJointSoftBodyCommon.h",
               ROOT / "physx/snippets/snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDCommon.h"]
    sources += sorted((ROOT / "physx/source/lowleveldynamics/src/avbd").glob("*.*"))
    report = {"platform": platform.platform(), "processor": platform.processor(),
              "sources": {str(path.relative_to(ROOT)): digest(path) for path in sources if path.is_file()},
              "frames": args.frames, "repeats": args.repeats, "binaries": {}, "samples": [], "comparisons": []}
    try:
        for config in configs:
            binary, bin_dir, fingerprint = build(config, args.output)
            report["binaries"][config] = fingerprint
            for grid in grids:
                for worker_count in workers:
                    for case in cases:
                        samples = []
                        for repeat in range(args.repeats):
                            # Interleave the models serially; retain all negative results.
                            order = ("fem", "joint") if repeat % 2 == 0 else ("joint", "fem")
                            pair = {}
                            for model in order:
                                pair[model] = run(binary, bin_dir, config, grid, worker_count, case, model, repeat, args)
                                report["samples"].append(pair[model])
                                (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                            samples.append(pair)
                        qualified = all(s[model]["physicsPassed"] for s in samples for model in ("fem", "joint"))
                        medians = {model: {key: statistics.median(float(s[model]["fields"][key]) for s in samples)
                                          for key in TIMING_FIELDS} for model in ("fem", "joint")
                                   if all(key in s[model]["fields"] for s in samples for key in TIMING_FIELDS)}
                        ratios = {key: medians["fem"][key] / medians["joint"][key]
                                  for key in TIMING_FIELDS if medians.get("joint", {}).get(key, 0) > 0
                                  and key in medians.get("fem", {})}
                        report["comparisons"].append({"config": config, "grid": grid, "workers": worker_count,
                            "case": case, "physicsQualified": qualified, "medianMs": medians,
                            "rawFemOverJointRatios": ratios, "qualifiedSpeedups": ratios if qualified else {}})
            if {path.name: digest(path) for path in [binary] + sorted(bin_dir.glob("*.dll"))} != fingerprint:
                raise RuntimeError("Binaries changed during comparison")
        report["sourcesUnchanged"] = all(digest(ROOT / path) == value for path, value in report["sources"].items())
        report["physicsQualified"] = all(c["physicsQualified"] for c in report["comparisons"])
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        report["error"] = str(error)
        print(str(error), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report.get("physicsQualified") and report.get("sourcesUnchanged") else 1


if __name__ == "__main__":
    raise SystemExit(main())
