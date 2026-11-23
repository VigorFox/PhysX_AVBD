#!/usr/bin/env python3
"""Build an external public-API observer against a clean official PhysX 5.9 SDK.

Measurements are evidence, not an automatic calibration of AVBD tolerances.
The SDK worktree must have no tracked modifications. No AVBD SDK DLL is copied.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
TAG = "110.1-omni-and-physx-5.9.0"
COMMIT = "517a0073715120e114ee055b63b26c95e00d9039"
VCVARS = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
LIBS = ("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
        "PhysX_64.lib", "PhysXCommon_64.lib", "PhysXFoundation_64.lib", "PhysXCooking_64.lib")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def finite_vec3(value):
    return isinstance(value, list) and len(value) == 3 and all(finite_number(x) for x in value)


def check_json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    if isinstance(value, dict):
        for child in value.values():
            check_json_finite(child)
    elif isinstance(value, list):
        for child in value:
            check_json_finite(child)


def command(argv, cwd=None):
    p = subprocess.run(argv, cwd=cwd, capture_output=True)
    return p.returncode, (p.stdout + p.stderr).decode("utf-8", errors="replace")


def gpu_state():
    try:
        code, output = command(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total",
                                "--format=csv,noheader"])
        return {"exit": code, "output": output.strip()}
    except OSError as error:
        return {"error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, default=Path("D:/projects/PhysX_official_5_9_baseline"))
    parser.add_argument("--output", type=Path, default=ROOT / "tools/benchmark-results/official-physx-20261023")
    parser.add_argument("--config", choices=("checked", "release"), default="checked")
    parser.add_argument("--backend", choices=("fem-gpu", "rigid-cpu"), default="fem-gpu")
    parser.add_argument("--cases", default="drop,five-cubes,reaction")
    parser.add_argument("--solver", choices=("pgs", "tgs"), default="tgs")
    parser.add_argument("--self-collision", choices=("on", "off"), default="on")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--dt", type=float, default=1 / 60)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--voxels", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-sdk", action="store_true")
    parser.add_argument("--cuda-root", type=Path,
                        default=Path("D:/projects/PhysX_official_5_9_baseline_tools/cuda-12.8.1"))
    parser.add_argument("--sdk-build-dir", type=Path,
                        help="Existing official SDK CMake directory; default uses the isolated CUDA 12.8 build")
    args = parser.parse_args()
    cases = args.cases.split(",")
    if (args.repetitions < 1 or args.frames < 1 or not 1 <= args.iterations <= 255 or
        not 1 <= args.voxels <= 64 or not 0 <= args.workers <= 256 or not math.isfinite(args.timeout) or args.timeout <= 0 or
        not math.isfinite(args.dt) or not 0 < args.dt < 1 or
        any(c not in ("drop", "five-cubes", "reaction") for c in cases) or len(cases) != len(set(cases))):
        parser.error("Require positive runs/frames/timeout, valid solver settings and unique known cases")
    sdk = args.sdk_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code, head = command(["git", "rev-parse", "HEAD"], sdk)
    if code or head.strip() != COMMIT:
        parser.error("SDK HEAD must be the official 5.9.0 tag commit " + COMMIT)
    code, dirty = command(["git", "status", "--porcelain", "--untracked-files=no"], sdk)
    if code or dirty.strip():
        parser.error("Official SDK has tracked modifications: " + dirty)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = output / run_id
    run_dir.mkdir()
    sdk_build = (args.sdk_build_dir or sdk / "physx/compiler/baseline-gpu-cuda128").resolve()
    cuda_root = args.cuda_root.resolve()
    if args.build_sdk:
        if not (cuda_root / "bin/nvcc.exe").is_file():
            parser.error("Official PhysX 5.9 GPU build requires a CUDA 12.8 toolkit: " + str(cuda_root))
        build_env = os.environ.copy()
        build_env["CL"] = "/utf-8"
        build_env["PM_rapidjson_PATH"] = "D:/packman-repo/chk/rapidjson/1.1.0-67fac85-073453e1_public"
        configure = ["cmake", "--fresh", "-S", (sdk / "physx/compiler/public").as_posix(), "-B", sdk_build.as_posix(),
            "-G", "Visual Studio 17 2022", "-A", "x64", "-T", "cuda=" + cuda_root.as_posix(),
            "-DCUDAToolkit_ROOT=" + cuda_root.as_posix(), "-DPHYSX_ROOT_DIR=" + (sdk / "physx").as_posix(),
            "-DTARGET_BUILD_PLATFORM=windows", "-DPX_OUTPUT_ARCH=x86",
            "-DPX_OUTPUT_LIB_DIR=" + (sdk / "physx").as_posix(),
            "-DPX_OUTPUT_BIN_DIR=" + (sdk / "physx").as_posix(), "-DPX_GENERATE_GPU_PROJECTS=ON",
            "-DPX_GENERATE_GPU_REDUCED_ARCHITECTURES=ON", "-DCMAKE_CUDA_ARCHITECTURES=89",
            "-DPX_BUILDSNIPPETS=OFF", "-DPX_BUILDPVDRUNTIME=OFF",
            "-DPX_GENERATE_STATIC_LIBRARIES=OFF", "-DNV_USE_STATIC_WINCRT=OFF"]
        build_sdk = ["cmake", "--build", str(sdk_build), "--config", args.config, "--target",
                     "PhysX", "PhysXGpu", "PhysXExtensions", "PhysXCooking", "PhysXPvdSDK",
                     "--", "/m:4", "/v:minimal"]
        (run_dir / "sdk-build-commands.json").write_text(json.dumps(
            {"configure": configure, "build": build_sdk, "CL": build_env["CL"],
             "PM_rapidjson_PATH": build_env["PM_rapidjson_PATH"]}, indent=2), encoding="utf-8")
        for name, argv in (("configure", configure), ("build-sdk", build_sdk)):
            with (run_dir / (name + ".log")).open("wb") as log:
                code = subprocess.run(argv, env=build_env, stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                print("SDK build failed; see " + str(run_dir))
                return code
    bin_dir = sdk / "physx/bin/win.x86_64.vc143.md" / args.config
    missing = [str(bin_dir / lib) for lib in LIBS if not (bin_dir / lib).is_file()]
    if missing:
        parser.error("Build the official SDK first; missing: " + ", ".join(missing))
    source = ROOT / "tools/official_physx_softbody_baseline.cpp"
    build = output / ("observer-" + args.config)
    build.mkdir(exist_ok=True)
    exe = build / "official_physx_baseline.exe"
    if list(build.glob("PhysX*.dll")):
        parser.error("Observer directory must not contain SDK DLLs")
    binaries = {p.name: {"path": str(p), "sha256": digest(p)}
                for p in bin_dir.glob("*.dll") if p.name.startswith("PhysX")}
    sources = {str(p.relative_to(ROOT)): digest(p) for p in
               (source, ROOT / "tools/official_physx_baseline_geometry.h") if p.is_file()}
    library_hashes = {lib: digest(bin_dir / lib) for lib in LIBS}
    cache = sdk_build / "CMakeCache.txt"
    sdk_manifest = sdk_build / ("sdk-build-attestation-" + args.config + ".json")
    sdk_fingerprint = {"sdkCommit": COMMIT, "sdkRoot": str(sdk), "config": args.config,
                       "binaries": binaries, "libraries": library_hashes,
                       "cmakeCacheSha256": digest(cache) if cache.is_file() else None}
    if args.build_sdk:
        code, dirty_after_build = command(["git", "status", "--porcelain", "--untracked-files=no"], sdk)
        if code or dirty_after_build.strip():
            parser.error("Official SDK source changed during build")
        sdk_manifest.write_text(json.dumps({"fingerprint": sdk_fingerprint,
            "successfulBuildEvidence": str(run_dir / "sdk-build-commands.json"),
            "buildLog": str(run_dir / "build-sdk.log"), "buildExit": 0,
            "scope": "local build provenance; not a signed reproducible-build certificate"}, indent=2), encoding="utf-8")
    if not sdk_manifest.is_file():
        parser.error("No successful official SDK build attestation; run --build-sdk first")
    attestation = json.loads(sdk_manifest.read_text(encoding="utf-8"))
    if attestation.get("fingerprint") != sdk_fingerprint or attestation.get("buildExit") != 0:
        parser.error("Official SDK build attestation does not match loaded files; rebuild SDK")
    (run_dir / "sdk-build-attestation.json").write_bytes(sdk_manifest.read_bytes())
    fingerprint = {"sdkRoot": str(sdk), "sdkCommit": COMMIT, "config": args.config,
                   "sources": sources, "libraries": library_hashes,
                   "compilerOptions": "/utf-8 /EHsc /std:c++14 /O2 /MD /DNDEBUG"}
    build_manifest = build / "build-provenance.json"
    if not args.skip_build:
        script = build / "build.cmd"
        script.write_text('@echo off\ncall "' + str(VCVARS) + '"\nif errorlevel 1 exit /b 2\n' +
            'cl /nologo /utf-8 /EHsc /std:c++14 /O2 /MD /DNDEBUG /I"' + str(sdk / "physx/include") +
            '" "' + str(source) + '" /Fe:"' + str(exe) + '" /link ' +
            " ".join('"' + str(bin_dir / lib) + '"' for lib in LIBS) + '\n', encoding="ascii")
        code, log = command(["cmd", "/c", str(script)], build)
        (build / "build.log").write_text(log, encoding="utf-8")
        if code:
            print(log)
            return code
        build_manifest.write_text(json.dumps({"fingerprint": fingerprint, "exeSha256": digest(exe)}, indent=2), encoding="utf-8")
    if not exe.is_file():
        parser.error("Missing observer executable")
    if args.skip_build:
        if not build_manifest.is_file():
            parser.error("--skip-build requires recorded build provenance")
        built = json.loads(build_manifest.read_text(encoding="utf-8"))
        if built.get("fingerprint") != fingerprint or built.get("exeSha256") != digest(exe):
            parser.error("--skip-build provenance mismatch; rebuild the observer")
    provenance = {"officialTag": TAG, "officialCommit": COMMIT, "sdkPath": str(sdk),
                  "trackedSdkClean": True, "config": args.config, "sdkBinaries": binaries,
                  "observerSources": sources, "observerSha256": digest(exe),
                  "gpuBefore": gpu_state(), "settings": vars(args).copy(),
                  "scope": "public-API reference workloads; not original AVBD five-body qualification",
                  "automaticToleranceCalibration": False}
    provenance["sdkBuildAttestationSha256"] = digest(sdk_manifest)
    provenance["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in provenance["settings"].items()}
    provenance["buildFingerprint"] = fingerprint
    provenance["sdkBuildDirectory"] = str(sdk_build)
    if cache.is_file():
        cache_copy = run_dir / "CMakeCache.txt"
        cache_copy.write_bytes(cache.read_bytes())
        provenance["cmakeCacheSha256"] = digest(cache_copy)
    # The SDK binary hashes remain the authority for the loaded build. Toolchain
    # metadata explains its construction and does not replace those checks.
    nvcc = cuda_root / "bin/nvcc.exe"
    if nvcc.is_file():
        version_code, version_text = command([str(nvcc), "--version"])
        provenance["availableCudaToolchain"] = {"root": str(cuda_root), "nvccSha256": digest(nvcc),
                                               "versionExit": version_code, "version": version_text}
    toolchain_audit = cuda_root / "toolchain-audit.json"
    if toolchain_audit.is_file():
        (run_dir / "toolchain-audit.json").write_bytes(toolchain_audit.read_bytes())
        provenance["availableCudaToolchainAuditSha256"] = digest(toolchain_audit)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if args.build_only:
        print("Observer built: " + str(exe))
        return 0
    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    records = []
    all_ok = True
    for rep in range(args.repetitions):
        for case in cases:
            stem = f"{args.config}-{args.backend}-{args.solver}-{case}-self-{args.self_collision}-i{args.iterations}-r{rep+1}"
            argv = [str(exe), "--backend", args.backend, "--case", case, "--solver", args.solver,
                    "--self-collision", args.self_collision, "--iterations", str(args.iterations),
                    "--frames", str(args.frames), "--dt", str(args.dt),
                    "--voxels", str(args.voxels), "--workers", str(args.workers)]
            started = time.perf_counter()
            with (run_dir / (stem + ".jsonl")).open("wb") as stdout, (run_dir / (stem + ".stderr.log")).open("wb") as stderr:
                process = subprocess.Popen(argv, cwd=bin_dir, env=env, stdout=stdout, stderr=stderr)
                timed_out = False
                try:
                    process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    process.wait()
            entries = []
            malformed = []
            for line in (run_dir / (stem + ".jsonl")).read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    parsed = json.loads(line, parse_constant=reject_constant)
                    check_json_finite(parsed)
                    if not isinstance(parsed, dict):
                        raise ValueError("non-object output")
                    entries.append(parsed)
                except (ValueError, TypeError):
                    if line.strip():
                        malformed.append(line)
            summaries = [r for r in entries if r.get("type") == "summary"]
            configs = [r for r in entries if r.get("type") == "config"]
            frames = [r for r in entries if r.get("type") == "frame"]
            cleanups = [r for r in entries if r.get("type") == "cleanup"]
            geometries = [r for r in entries if r.get("type") == "initialGeometry"]
            validation = []
            if any(line.lstrip().startswith("{") for line in malformed):
                validation.append("malformed JSON record")
            if len(cleanups) != 1 or not cleanups[0].get("ok") or cleanups[0].get("errors") != 0:
                validation.append("missing or failed cleanup")
            if len(geometries) != 1 or geometries[0].get("valid") is not True:
                validation.append("invalid initial geometry observation")
            if len(summaries) != 1 or len(configs) != 1:
                validation.append("require exactly one config and summary")
            if summaries:
                summary = summaries[-1]
                for key in ("executionOk", "measurementOk", "completed", "finite"):
                    if summary.get(key) is not True:
                        validation.append("summary " + key + " is not true")
                if summary.get("framesDone") != args.frames or summary.get("errors") != 0:
                    validation.append("incomplete frame count or SDK errors")
                if summary.get("pairGeometryAllValid") is not True:
                    validation.append("invalid pair geometry observations")
                if any(not finite_number(summary.get(key)) for key in (
                    "stepMeanMs", "stepP95Ms", "stepMaxMs", "groundPenetrationMax", "minDet",
                    "minBodyVolumeRatio", "maxTetMtd", "momentumChangeMax", "maxKineticGravityEnergy")):
                    validation.append("missing or nonfinite summary measurements")
            if [r.get("frame") for r in frames] != list(range(1, args.frames + 1)) or any(r.get("finite") is not True for r in frames):
                validation.append("missing, reordered or nonfinite frames")
            if any(r.get("pairGeometryValid") is not True for r in frames):
                validation.append("invalid frame geometry observation")
            frame_numbers = ("time", "stepMs", "readbackMs", "observedMs", "geometryMs", "minDet",
                             "minBodyVolumeRatio", "groundPenetration", "maxVertexSpeed", "kineticEnergy",
                             "gravitationalEnergy", "maxTetMtd")
            if any(not finite_number(r.get(key)) for r in frames for key in frame_numbers):
                validation.append("missing or nonfinite frame measurements")
            body_records = [r for r in entries if r.get("type") == "body"]
            body_count = 5 if case == "five-cubes" else 2 if case == "reaction" else 1
            initial_states = [r for r in entries if r.get("type") == "initialState"]
            if len(initial_states) != 1 or len(initial_states[0].get("bodies", [])) != body_count:
                validation.append("missing initial body observations")
            for state in frames + initial_states:
                observed_bodies = state.get("bodies", [])
                if (len(observed_bodies) != body_count or any(
                    not finite_vec3(b.get("momentum")) or not finite_vec3(b.get("com")) or
                    not finite_number(b.get("minDet")) or not finite_number(b.get("volumeRatio"))
                    for b in observed_bodies)):
                    validation.append("missing or nonfinite body observations")
                    break
            if any(not finite_vec3(r.get("momentum")) for r in frames):
                validation.append("missing or nonfinite frame momentum")
            if ([r.get("index") for r in body_records] != list(range(body_count)) or
                any(not finite_number(r.get("mass")) or r["mass"] <= 0 for r in body_records)):
                validation.append("missing or invalid body metadata")
            if args.backend == "fem-gpu" and any(
                not finite_number(r.get(key)) or r[key] <= 0 for r in body_records
                for key in ("simVertices", "simTetrahedra", "collisionVertices", "collisionTetrahedra")):
                validation.append("empty soft body geometry")
            if configs:
                config = configs[0]
                expected = {"backend": args.backend, "case": case, "rigidSolver": args.solver,
                            "frames": args.frames, "iterations": args.iterations, "voxels": args.voxels,
                            "workers": args.workers, "selfCollision": args.self_collision == "on"}
                for key, value in expected.items():
                    if config.get(key) != value:
                        validation.append("config mismatch: " + key)
                if not finite_number(config.get("dt")) or not math.isclose(config["dt"], args.dt, rel_tol=1e-6):
                    validation.append("config mismatch: dt")
                for key, dll in (("physxModule", "PhysX_64.dll"), ("gpuModule", "PhysXGpu_64.dll"),
                                 ("commonModule", "PhysXCommon_64.dll"),
                                 ("foundationModule", "PhysXFoundation_64.dll"),
                                 ("cookingModule", "PhysXCooking_64.dll")):
                    if key == "gpuModule" and args.backend == "rigid-cpu":
                        continue
                    if not config.get(key) or Path(config[key]).resolve() != (bin_dir / dll).resolve():
                        validation.append("wrong loaded module: " + dll)
                    elif digest(bin_dir / dll) != binaries[dll]["sha256"]:
                        validation.append("loaded SDK changed during run: " + dll)
            okay = process.returncode == 0 and not timed_out and not validation
            all_ok &= okay
            record = {"name": stem, "argv": argv, "exit": process.returncode, "timedOut": timed_out,
                      "completedMeasurement": okay, "wallSeconds": time.perf_counter() - started,
                      "summary": summaries[-1] if summaries else None,
                      "validationErrors": validation, "nonJsonOutput": malformed, "gpuAfter": gpu_state()}
            records.append(record)
            print(json.dumps(record, ensure_ascii=True), flush=True)
    provenance["runs"] = records
    provenance["completedMeasurements"] = all_ok
    provenance["gpuAfter"] = gpu_state()
    final_clean = command(["git", "status", "--porcelain", "--untracked-files=no"], sdk)
    provenance["trackedSdkCleanAfter"] = final_clean[0] == 0 and not final_clean[1].strip()
    report = run_dir / "report.json"
    report.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return 0 if all_ok and provenance["trackedSdkCleanAfter"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
