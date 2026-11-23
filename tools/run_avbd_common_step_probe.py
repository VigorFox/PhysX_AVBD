"""Exercise actual opt-in pose/force publication, not a frozen capture."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from run_avbd_ogc_certificate_probe import ROOT, run_probe
from avbd_gate import parse_fields

LIBRARIES = ("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
             "PhysX_64.lib", "PhysXCommon_64.lib", "PhysXCooking_64.lib")


def evaluate_output(mode, overlap, code, output, require_stationary=False):
    parsed = [parse_fields(line.replace("[AVBD_COMMON_FRAME]", "[AVBD_GATE]", 1))
              for line in output.splitlines() if line.startswith("[AVBD_COMMON_FRAME]")]
    if not parsed or any(errors for _, errors in parsed):
        return False
    frames = [fields for fields, _ in parsed]
    rejected = (len(frames) == 1 and frames[0].get("status") == "REJECTED"
                and frames[0].get("poseForcePublications") == "0" and frames[0].get("bdf") == "0")
    if mode == 7:
        # The reason/path may change (e.g. earlier infeasibility proof).
        return code == 0 and rejected and "unpublished=PASS" in output
    if overlap:
        return code in (0, 1) and rejected and "np_geometry_or_initial_recovery_uncompiled" in output
    return (code == 0 and (mode == 8 or len(frames) == 1)
            and all(f.get("status") == "ACCEPTED" and f.get("poseForcePublications") == "1"
                    and f.get("bdf") == "1" for f in frames)
            and (not require_stationary or all(f.get("stationary") == "1" for f in frames))
            and "fetchedFinite=PASS" in output and "status=FAIL" not in output
            and (mode != 6 or "[COMMON_CHAIN]" in output)
            and (mode != 8 or "[COMMON_INDEPENDENT]" in output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=int)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chain-bodies", type=int, default=12)
    parser.add_argument("--mode", choices=("strict", "local", "patch", "bounded"), default="strict")
    parser.add_argument("--require-stationary", action="store_true",
                        help="Additional research qualification; physical oracles always apply")
    parser.add_argument("--logs", type=Path,
                        default=ROOT / "tools/benchmark-results/common-step")
    args = parser.parse_args()
    if args.child is not None:
        return run_probe(ROOT / "tools/avbd_common_program_scene_probe.cpp",
                         args=[str(args.child), "overlap" if args.overlap else "separated",
                               str(args.threads), str(args.chain_bodies)],
                         libraries=LIBRARIES)
    args.logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # A frozen diagnostic call must never qualify physical publication.
    for key in list(env):
        if key.startswith("PHYSX_AVBD_COMMON_"):
            del env[key]
    env.update(PHYSX_AVBD_COMMON_STEP="1" if args.mode == "strict" else args.mode, PHYSX_AVBD_COMMON_STATS="1",
               PHYSX_NP_GEOMETRY_CERTIFICATES="2")
    failures = 0
    cases = [(i, False) for i in range(7)] + [(0, True)]
    if args.mode == 'bounded': cases.extend(((7, False), (8, False)))
    for mode, overlap in cases:
        cmd = [sys.executable, __file__, "--child", str(mode),
               "--threads", str(args.threads), "--chain-bodies", str(args.chain_bodies)]
        if overlap:
            cmd.append("--overlap")
        run = subprocess.run(cmd, env=env, capture_output=True, timeout=60)
        output = (run.stdout + run.stderr).decode("utf-8", errors="replace")
        name = "initial-recovery-rejected" if overlap else "conflicting-anchors-rejected" if mode == 7 else f"mode{mode}"
        (args.logs / (name + ".log")).write_text(output, encoding="utf-8")
        frames = [line for line in output.splitlines() if line.startswith("[AVBD_COMMON_FRAME]")]
        ok = evaluate_output(mode, overlap, run.returncode, output, args.require_stationary)
        failures += not ok
        stationary = sum("stationary=1" in line for line in frames)
        print(f"[COMMON_STEP_TEST] case={name} status={'PASS' if ok else 'FAIL'} "
              f"exit={run.returncode} stationaryFrames={stationary} frames={len(frames)} requireStationary={int(args.require_stationary)}", flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
