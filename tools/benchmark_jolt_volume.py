#!/usr/bin/env python3
"""Measure real PhysX snippet scenes and Jolt imports of their cooked input.

This is a workload baseline, not a claim of equivalent constitutive/contact
models. Requires numpy. Each subprocess inherits one selected logical CPU.
"""
import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIN = "a45f543e24f630b33fd3366c7c2ae241b63816b3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextlib.contextmanager
def affinity(cpu):
    # Set the parent before CreateProcess: the child starts with this affinity,
    # including its first update and any worker threads it creates.
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        old, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(old), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        if cpu < 0 or not old.value & (1 << cpu):
            raise ValueError(f"CPU {cpu} is outside the available affinity mask")
        if not kernel.SetProcessAffinityMask(handle, 1 << cpu):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not kernel.SetProcessAffinityMask(handle, old.value):
                raise ctypes.WinError(ctypes.get_last_error())
    else:
        old = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {cpu})
        try:
            yield
        finally:
            os.sched_setaffinity(0, old)


def read_scene(path):
    lines = iter(path.read_text().splitlines())
    assert next(lines) == "AVBD_JOLT_SCENE 1"
    case = next(lines).split()[1]
    dt = float(next(lines).split()[1])
    gravity = list(map(float, next(lines).split()[1:]))
    soft = next(lines).split()[1:]
    nv, nt, nc, nct = map(int, soft[:4])
    assert next(lines) == "vertices"
    vertices = np.array([list(map(float, next(lines).split())) for _ in range(nv)], dtype=np.float32)
    assert vertices.shape == (nv, 7)
    assert next(lines) == "tets"
    tets = np.array([list(map(int, next(lines).split())) for _ in range(nt)])
    nr = int(next(lines).split()[1])
    rigids = []
    for _ in range(nr):
        row = next(lines).split()
        assert len(row) == 20
        rigids.append(dict(shape=row[0], dynamic=bool(int(row[1])), values=list(map(float, row[2:]))))
    return dict(case=case, dt=dt, gravity=gravity, vertices=vertices, tets=tets,
                collisionVertices=nc, collisionTets=nct, rigids=rigids,
                materialModel="CO_ROTATIONAL" if case == "embedded-sync" else "NEO_HOOKEAN",
                material=dict(zip(["young", "poisson", "friction", "elasticityDamping", "linearDamping", "maxVelocity"], map(float, soft[4:]))))


def dets(points, tets):
    p = points[..., tets, :]
    return np.einsum("...i,...i->...", np.cross(p[..., 1, :] - p[..., 0, :],
        p[..., 2, :] - p[..., 0, :]), p[..., 3, :] - p[..., 0, :])


def analyze(path, scene, frames):
    nv, nr = len(scene["vertices"]), len(scene["rigids"])
    raw = np.fromfile(path, dtype="<f4")
    width = 2 + 3 * nv + 7 * nr
    if len(raw) != frames * width or not np.isfinite(raw).all():
        raise RuntimeError(f"Incomplete/non-finite trace: {path}")
    data = raw.reshape(frames, width).astype(np.float64)
    p = data[:, 2:2 + 3 * nv].reshape(frames, nv, 3)
    rigid = data[:, 2 + 3 * nv:].reshape(frames, nr, 7)
    rest, tets = scene["vertices"][:, :3].astype(np.float64), scene["tets"]
    det0, det = dets(rest, tets), dets(p, tets)
    jacobian = det / det0
    ratio = np.abs(det).sum(axis=1) / np.abs(det0).sum()
    edges = np.unique(np.sort(np.concatenate([tets[:, [a, b]] for a in range(4) for b in range(a + 1, 4)]), axis=1), axis=0)
    length0 = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    strain = np.abs(np.linalg.norm(p[:, edges[:, 0]] - p[:, edges[:, 1]], axis=2) / length0 - 1)
    mass = 1 / scene["vertices"][:, 3].astype(np.float64)
    center = np.einsum("fvi,v->fi", p, mass) / mass.sum()
    depth_by_rigid = []
    for i, body in enumerate(scene["rigids"]):
        # Common diagnostic: signed distance of simulation vertices to the
        # actual rigid core. This is NOT a surface-triangle nonpenetration gate.
        q = rigid[:, i, 3:7]
        q = q / np.linalg.norm(q, axis=1)[:, None]
        v = p - rigid[:, i, :3][:, None, :]
        qv = -q[:, None, :3]
        uv = 2 * np.cross(qv, v)
        local = v + q[:, None, 3:4] * uv + np.cross(qv, uv)
        if body["shape"] == "plane":
            sdf = local[:, :, 0]
        else:
            delta = np.abs(local) - np.array(body["values"][7:10])
            sdf = np.linalg.norm(np.maximum(delta, 0), axis=2) + np.minimum(delta.max(axis=2), 0)
        depth_by_rigid.append(float(np.maximum(-sdf, 0).max()))
    active = data[:, 1] != 0
    ms = data[:, 0]
    physics_hash = hashlib.sha256(raw.reshape(frames, width)[:, 2:].tobytes()).hexdigest()
    dynamic_y = [float(rigid[-1, i, 1]) for i, b in enumerate(scene["rigids"]) if b["dynamic"]]
    return dict(meanStepMs=float(ms.mean()), medianStepMs=float(np.median(ms)), p95StepMs=float(np.percentile(ms, 95)),
                maxStepMs=float(ms.max()), activeFrames=int(active.sum()),
                activeMeanStepMs=float(ms[active].mean()) if active.any() else None,
                first60MeanStepMs=float(ms[:min(frames, 60)].mean()),
                first30MeanStepMs=float(ms[:min(frames, 30)].mean()),
                minDetF=float(jacobian.min()), maxInvertedTets=int((jacobian <= 0).sum(axis=1).max()),
                minVolumeRatio=float(ratio.min()), maxVolumeRatio=float(ratio.max()), finalVolumeRatio=float(ratio[-1]),
                peakEdgeStrain=float(strain.max()), finalEdgeStrain=float(strain[-1].max()),
                minY=float(p[:, :, 1].min()), finalCenterOfMass=center[-1].tolist(),
                maxSimulationVertexCorePenetrationByRigid=depth_by_rigid,
                finalDynamicRigidY=dynamic_y,
                rigidRetentionEnvelopePassed=all(1.6 < y < 4.0 for y in dynamic_y) if dynamic_y else None,
                physicsTraceSHA256=physics_hash)


def run_one(label, case, index, args, canonical):
    prefix = args.output / f"{case}-{index}-{label}"
    trace = prefix.with_suffix(".bin")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PHYSX_AVBD_") and not k.startswith("PHYSX_SNIPPET_")}
    if label == "physx":
        scene_path = prefix.with_suffix(".scene")
        env["PHYSX_AVBD_BENCH_SCENE"] = str(scene_path)
        env["PHYSX_AVBD_BENCH_TRACE"] = str(trace)
        exe = args.physx_exe
        command = [str(exe), "--particle3", "--headless", "--solver=avbd", f"--case={case}", f"--frames={args.frames}", "--dispatcher-threads=1"]
    else:
        exe = args.jolt_exe
        command = [str(exe), str(canonical), str(trace), str(args.frames), label.split("-")[1],
                   str(args.edge_compliance), str(args.volume_compliance)]
    env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")
    start = time.perf_counter()
    with affinity(args.cpu):
        process = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, timeout=args.timeout)
    wall_seconds = time.perf_counter() - start
    output = (process.stdout + b"\n" + process.stderr).decode("utf-8", errors="replace")
    prefix.with_suffix(".log").write_text(output, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"{label} {case} failed ({process.returncode}); see {prefix}.log")
    if label == "physx":
        if "status=PASS" not in output:
            raise RuntimeError("Missing PhysX physical gate PASS")
        if canonical.exists() and canonical.read_bytes() != scene_path.read_bytes():
            raise RuntimeError("PhysX cooked inputs changed between repetitions")
        if not canonical.exists():
            canonical.write_bytes(scene_path.read_bytes())
    scene = read_scene(canonical)
    metrics = analyze(trace, scene, args.frames)
    record = dict(label=label, case=case, repeat=index, command=command,
                  processWallSeconds=wall_seconds, sceneSHA256=sha(canonical),
                  traceFile=trace.name, logFile=prefix.with_suffix(".log").name, metrics=metrics)
    print(f"{case} {label} #{index}: {metrics['meanStepMs']:.6f} ms/step, "
          f"active={metrics['activeFrames']}/{args.frames}, minJ={metrics['minDetF']:.4f}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physx-exe", type=Path, default=ROOT / "physx/bin/win.x86_64.vc143.md/release/SnippetDeformableVolumeAVBD_64.exe")
    parser.add_argument("--jolt-exe", type=Path, default=ROOT / "tools/jolt_soft_benchmark/build/Release/JoltSoftBenchmark.exe")
    parser.add_argument("--cases", default="rigid-on-soft,rigid-on-soft-separated,embedded-sync")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", default="5,16,32")
    parser.add_argument("--edge-compliance", type=float, default=0)
    parser.add_argument("--volume-compliance", type=float, default=0)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = args.cases.split(",")
    iterations = list(map(int, args.iterations.split(",")))
    if set(cases) - {"rigid-on-soft", "rigid-on-soft-separated", "embedded-sync"} or min(args.frames, args.repeats, *iterations) <= 0:
        parser.error("Invalid cases or counts")
    args.output = args.output.resolve()
    args.physx_exe, args.jolt_exe = args.physx_exe.resolve(), args.jolt_exe.resolve()
    revision = subprocess.check_output(["git", "-C", str(ROOT / "reference/JoltPhysics"), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        parser.error(f"Expected Jolt revision {PIN}, got {revision}")
    if subprocess.check_output(["git", "-C", str(ROOT / "reference/JoltPhysics"), "status", "--porcelain"]):
        parser.error("Jolt source checkout must be clean")
    args.output.mkdir(parents=True, exist_ok=False)
    labels = ["physx"] + [f"jolt-{i}" for i in iterations]
    report = dict(schema=1, platform=platform.platform(), processor=platform.processor(), joltRevision=revision,
        physxRevision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        configuration=dict(frames=args.frames, repeats=args.repeats, warmupFullRuns=1, cpu=args.cpu,
            dispatcherThreads=1, joltThreads=1, dt="exported PhysX 1/60 float", iterations=iterations,
            edgeCompliance=args.edge_compliance, volumeCompliance=args.volume_compliance,
            timing="simulate+fetchResults / PhysicsSystem::Update; diagnostics and I/O excluded",
            materialEquivalence=False, collisionRepresentationEquivalence=False),
        binaries={str(p): sha(p) for p in [args.physx_exe, args.jolt_exe] + sorted(args.physx_exe.parent.glob("*.dll"))},
        sourceFiles={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__).resolve(),
            ROOT / "tools/jolt_soft_benchmark/main.cpp", ROOT / "tools/jolt_soft_benchmark/CMakeLists.txt",
            ROOT / "physx/snippets/snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDHeadless.cpp",
            ROOT / "physx/snippets/snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDBenchmark.h",
            ROOT / "physx/source/physx/src/NpDeformableVolume.cpp"] +
            sorted((ROOT / "physx/source/lowleveldynamics/src/avbd").glob("*.cpp")) +
            sorted((ROOT / "physx/source/lowleveldynamics/src/avbd").glob("*.h"))},
        warmup=[], runs=[], scenes={}, summary={})
    for case in cases:
        canonical = args.output / f"{case}.scene"
        for label in labels:
            report["warmup"].append(run_one(label, case, "warmup", args, canonical))
        scene = read_scene(canonical)
        report["scenes"][case] = dict(vertices=len(scene["vertices"]), tets=len(scene["tets"]),
            collisionVertices=scene["collisionVertices"], collisionTets=scene["collisionTets"],
            material=scene["material"], materialModel=scene["materialModel"],
            totalMass=float((1.0 / scene["vertices"][:, 3].astype(np.float64)).sum()),
            rigids=scene["rigids"], sceneSHA256=sha(canonical))
        for repeat in range(args.repeats):
            for label in labels if repeat % 2 == 0 else labels[::-1]:
                report["runs"].append(run_one(label, case, repeat, args, canonical))
                (args.output / "report.partial.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary = report["summary"][case] = {}
        for label in labels:
            records = [r["metrics"] for r in report["runs"] if r["case"] == case and r["label"] == label]
            mean = statistics.median(r["meanStepMs"] for r in records)
            rejection = []
            if any(r["rigidRetentionEnvelopePassed"] is False for r in records):
                rejection.append("rigid_retention_envelope")
            if any(r["maxInvertedTets"] or r["minDetF"] < 0.049 or r["minY"] < -0.15 for r in records):
                rejection.append("basic_shape_or_ground_diagnostic")
            summary[label] = dict(medianOfMeanStepMs=mean, runMeanStepMs=[r["meanStepMs"] for r in records],
                medianActiveMeanStepMs=statistics.median(r["activeMeanStepMs"] for r in records),
                diagnosticRejectionReasons=rejection,
                meanStepRangeMs=[min(r["meanStepMs"] for r in records), max(r["meanStepMs"] for r in records)],
                physicsDeterministicAcrossRuns=len({r["physicsTraceSHA256"] for r in records}) == 1,
                representativeMetrics=records[0])
        for label in labels[1:]:
            if not summary[label]["diagnosticRejectionReasons"] and not summary["physx"]["diagnosticRejectionReasons"]:
                summary[label]["physxOverJoltTimeRatio"] = summary["physx"]["medianOfMeanStepMs"] / summary[label]["medianOfMeanStepMs"]
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output / "report.json")


if __name__ == "__main__":
    main()
