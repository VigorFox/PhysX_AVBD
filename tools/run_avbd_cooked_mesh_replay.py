#!/usr/bin/env python3
"""Capture SDK cooked volume bytes once, then verify configuration replay.

One-frame probes validate input and IO only. Full qualification requires all
three configurations, the manifest's frames and budget, and native physics PASS.
No physics equations or tolerances are implemented here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

import avbd_gate
import run_avbd_contact_coverage as coverage

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ("debug", "checked", "release")
NAMES = ("cube", "sphere", "cone")
HEADER = struct.Struct("<8sIIQQII")
ENV_KEYS = ("PHYSX_AVBD_MESH_CAPTURE_DIR", "PHYSX_AVBD_MESH_REPLAY_DIR",
            "PHYSX_AVBD_MESH_CAPTURE_GPU_DATA")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def fixture(path):
    data = path.read_bytes()
    if len(data) < HEADER.size + 12:
        raise ValueError("empty or truncated fixture: " + str(path))
    magic, sdk, voxels, authored, digest, size, reserved = HEADER.unpack_from(data)
    payload = data[HEADER.size:]
    if (magic != b"AVBDMV1\0" or reserved or size != len(payload) or
            fnv(payload) != digest):
        raise ValueError("invalid fixture: " + str(path))
    return dict(sha256=sha(path), sdk=sdk, voxels=voxels, bytes=size,
                authoredHash="%016x" % authored, payloadHash="%016x" % digest,
                streamVersion=struct.unpack_from("<I", payload, 8)[0])


def records(text, tag, key, expected):
    result = {}
    for line in text.splitlines():
        if line.startswith(tag + " "):
            fields, errors = avbd_gate.parse_fields(line.replace(tag, "[AVBD_GATE]", 1))
            identity = fields.get(key)
            if errors or identity in result or identity not in expected:
                raise ValueError("invalid or duplicate " + tag)
            result[identity] = fields
    if set(result) != set(expected):
        raise ValueError("missing " + tag)
    return result


def input_records(text):
    ids = [str(i) for i in range(5)]
    hashes = records(text, "[AVBD_MESH_HASH]", "volume", ids)
    metrics = records(text, "[AVBD_MESH]", "volume", ids)
    for identity in ids:
        for name in ("simTopology", "collisionTopology", "simRestGeometry", "collisionRestGeometry"):
            value = hashes[identity].get(name, "")
            if len(value) != 16 or int(value, 16) == 0:
                raise ValueError("missing mesh fingerprint")
        for name in ("totalMass", "totalVolume", "minSixVolume", "minBoundaryDoubleArea"):
            value = metrics[identity].get(name)
            if not avbd_gate.is_finite_token(value) or float(value) <= 0:
                raise ValueError("invalid rest/mass geometry")
        if any(metrics[identity].get(name) != "0" for name in ("invalidTets", "invalidFaces")):
            raise ValueError("invalid rest topology")
    return dict(hashes=hashes, metrics=metrics)


def mesh_identity(inputs, include_derived_mass=True):
    # Derived boundary areas use SDK interpolated collision positions and may
    # differ by float roundoff between independent cooking configurations. Keep
    # their values in evidence; native rest/partition oracles check that error.
    keys = ("simVertices", "simTets", "collisionVertices", "collisionTets", "boundaryFaces")
    if include_derived_mass:
        keys += ("totalMass", "totalVolume")
    return dict(hashes=inputs["hashes"], metrics={identity: {
        key: values[key] for key in keys} for identity, values in inputs["metrics"].items()})


def snapshot(configs):
    paths = set((ROOT / "physx/source/lowleveldynamics/src/avbd").glob("*.h"))
    paths.update((ROOT / "physx/source/lowleveldynamics/src/avbd").glob("*.cpp"))
    paths.update((ROOT / "physx/snippets/snippetdeformablevolumeavbd").glob("*.*"))
    for name in ("GuMeshFactory.cpp", "cooking/GuCookingTetrahedronMesh.cpp", "mesh/GuMeshData.h"):
        paths.add(ROOT / "physx/source/geomutils/src" / name)
    for name in ("run_avbd_cooked_mesh_replay.py", "avbd_contact_coverage_cases.json", "avbd_gate.py"):
        paths.add(ROOT / "tools" / name)
    for config in configs:
        folder = ROOT / "physx/bin/win.x86_64.vc143.md" / config
        paths.update(folder.glob("*.dll"))
        paths.add(folder / coverage.SNIPPET_EXE["volume"])
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths) if path.is_file()}


def qualification(checks_passed, runs, configs, frames, timeout, spec):
    return (checks_passed and set(configs) == set(CONFIGS) and
            frames == spec["frames"] and timeout <= spec.get("timeout", 600) and
            len(runs) == len(CONFIGS) and all(
                run["physical_valid"] and run["observed_status"] == "pass" and
                not run["errors"] and not run["timed_out"] for run in runs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New evidence directory; never overwritten")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--capture-config", choices=CONFIGS, default="checked")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--assets", type=Path, help="Reuse existing CPU fixtures instead of capturing again")
    args = parser.parse_args()
    if args.frames < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or len(set(args.configs)) != len(args.configs):
        parser.error("positive frames/timeout and unique configurations required")
    out = args.output_dir.resolve()
    if out.exists():
        parser.error("output directory already exists")
    configs = sorted(set(args.configs + [args.capture_config]))
    for config in configs:
        if not (ROOT / "physx/bin/win.x86_64.vc143.md" / config / coverage.SNIPPET_EXE["volume"]).is_file():
            parser.error("missing executable for " + config)
    out.mkdir(parents=True)
    assets = args.assets.resolve() if args.assets else out / "assets"
    manifest = json.loads(Path(coverage.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    spec = next(item for item in manifest["cases"] if item["id"] == "volume-visual-showcase")
    report = dict(schema=1, frames=args.frames, timeout=args.timeout, configs=args.configs,
                  assets=str(assets), checks=[], runs=[], errors=[], qualification="NOT_VERIFIED",
                  before=snapshot(configs))

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def check(name, value):
        report["checks"].append(dict(name=name, passed=bool(value)))
        save()
        if not value:
            raise ValueError(name)

    def run(label, config, frames=1, mode=None, directory=None, gpu=False):
        folder = ROOT / "physx/bin/win.x86_64.vc143.md" / config
        argv = [str(folder / coverage.SNIPPET_EXE["volume"]), "--particle3", "--headless", "--solver=avbd",
                "--case=visual-showcase", "--frames=" + str(frames),
                "--dispatcher-threads=" + str(spec["dispatcher_threads"])]
        env = {key: value for key, value in os.environ.items() if key not in ENV_KEYS}
        env.update(PHYSX_SNIPPET_HEADLESS="1", PHYSX_AVBD_MESH_DIAGNOSTICS="1", PHYSX_AVBD_HEADLESS_PROGRESS="1")
        if mode:
            env["PHYSX_AVBD_MESH_" + mode.upper() + "_DIR"] = str(directory)
        if gpu:
            env["PHYSX_AVBD_MESH_CAPTURE_GPU_DATA"] = "1"
        started = time.monotonic()
        print("[COOKED_REPLAY_RUN] label=%s config=%s frames=%s" % (label, config, frames), flush=True)
        try:
            proc = subprocess.run(argv, cwd=folder, env=env, capture_output=True, text=True, timeout=args.timeout)
            result = dict(ran=True, exit=proc.returncode, timed_out=False,
                          text=coverage.process_output(proc.stdout, proc.stderr))
        except subprocess.TimeoutExpired as exc:
            result = dict(ran=True, exit=124, timed_out=True,
                          text=coverage.process_output(exc.stdout, exc.stderr))
        result.update(argv=argv, elapsed_seconds=time.monotonic() - started, timeout_seconds=args.timeout)
        (out / (label + ".log")).write_text(result["text"], encoding="utf-8")
        row = coverage.evaluate_run(dict(spec, frames=frames), result, config, "avbd")
        row.update(label=label, config=config, mode=mode, frames=frames)
        report["runs"].append(row)
        save()
        return row

    def passed(row):
        return row["physical_valid"] and row["observed_status"] == "pass" and not row["errors"]

    def payload_matches(row):
        data = records(row["process_output"], "[AVBD_COOKED_MESH]", "name", NAMES)
        return all(all(str(data[name][key]) == str(report["fixtures"][name][key])
                       for key in ("voxels", "authoredHash", "bytes", "payloadHash")) for name in NAMES)

    try:
        if not args.assets:
            assets.mkdir()
            check("capture", passed(run("capture", args.capture_config, mode="capture", directory=assets)))
        report["fixtures"] = {name: fixture(assets / (name + ".pxvol")) for name in NAMES}
        check("cpu_stream_version", all(item["streamVersion"] == 4 for item in report["fixtures"].values()))
        baseline = None
        for config in args.configs:
            regular = run("regular-" + config, config)
            replay = run("probe-" + config, config, mode="replay", directory=assets)
            check(config + "_probe_pass", passed(regular) and passed(replay))
            check(config + "_payload", payload_matches(replay))
            inputs = input_records(regular["process_output"])
            check(config + "_roundtrip_inputs", mesh_identity(inputs, False) == mesh_identity(input_records(replay["process_output"]), False))
            # Exact topology/serialized input checks remain valid. Derived
            # mass reductions and simulated trajectories need not be bitwise.
            before_hash = regular["fields"].get("trajectoryHash")
            after_hash = replay["fields"].get("trajectoryHash")
            report.setdefault("trajectoryDiagnostics", []).append(dict(
                config=config, regular=before_hash, replay=after_hash,
                equal=before_hash == after_hash))
            save()
            if baseline is None:
                baseline = inputs
            # SDK updateMass and diagnostic reductions execute in each build;
            # retain those float results without claiming bitwise identity.
            check(config + "_cross_config_topology_geometry",
                  mesh_identity(inputs, False) == mesh_identity(baseline, False))

        # Exercise the original GPU cooking/stream layout without GPU simulation.
        gpu_assets = out / "gpu-assets"
        gpu_assets.mkdir()
        check("gpu_capture", passed(run("gpu-capture", args.capture_config, mode="capture", directory=gpu_assets, gpu=True)))
        check("gpu_stream_version", all(fixture(gpu_assets / (name + ".pxvol"))["streamVersion"] == 3 for name in NAMES))
        for config in args.configs:
            check(config + "_gpu_stream_replay", passed(run("gpu-probe-" + config, config, mode="replay", directory=gpu_assets)))

        for negative, reason in (("missing", "replay_open"), ("truncated", "replay_header"),
                                 ("corrupt", "replay_checksum"), ("authored", "replay_header"),
                                 ("legacy_cpu", "deserialization")):
            folder = out / ("negative-" + negative)
            shutil.copytree(assets, folder)
            path = folder / "cube.pxvol"
            data = bytearray(path.read_bytes())
            if negative == "missing":
                path.unlink()
            else:
                if negative == "truncated":
                    data = data[:-1]
                elif negative == "corrupt":
                    data[-1] ^= 1
                elif negative == "authored":
                    data[16] ^= 1
                else:
                    # Valid envelope/checksum, unsupported CPU SDK stream v3.
                    struct.pack_into("<I", data, HEADER.size + 8, 3)
                    struct.pack_into("<Q", data, 24, fnv(data[HEADER.size:]))
                path.write_bytes(data)
            row = run("negative-" + negative, args.capture_config, mode="replay", directory=folder)
            check(negative + "_rejected", row["exit"] == 1 and not row["timed_out"] and
                  row["observed_status"] != "pass" and row["fields"].get("completedFrames") == "0" and
                  "[AVBD_MESH_IO_ERROR] name=cube reason=" + reason in row["process_output"])

        full = []
        if args.frames > 1:
            for config in args.configs:
                row = run("replay-" + config, config, args.frames, "replay", assets)
                full.append(row)
                check(config + "_full_payload", payload_matches(row))
        report["after"] = snapshot(configs)
        check("no_source_binary_drift", report["before"] == report["after"])
        check("no_fixture_drift", report["fixtures"] == {name: fixture(assets / (name + ".pxvol")) for name in NAMES})
        report["input_validation"] = "PASS"
        report["qualification"] = "PASS" if qualification(True, full, args.configs, args.frames, args.timeout, spec) else "NOT_VERIFIED"
        report["result"] = "PASS" if all(passed(row) for row in full) else "FAIL"
    except (OSError, ValueError, KeyError) as exc:
        report["errors"].append(str(exc))
        report["result"] = "FAIL"
    save()
    print("[COOKED_REPLAY] result=%s inputValidation=%s qualification=%s report=%s" %
          (report["result"], report.get("input_validation", "FAIL"), report["qualification"], out / "report.json"))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
