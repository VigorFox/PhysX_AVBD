"""Qualify finite local iterations through actual PhysX scene publication.

The scope is Rigid6, with optional NP contact mechanism qualification. Unsupported cases are separate negative tests;
they are never counted as physical success. Thread runs use independent physical
oracles, not pose equality against another schedule.
"""
import argparse
import json
import os
import subprocess
import sys

from avbd_gate import parse_fields
from run_avbd_common_step_probe import ROOT


def records(output, tag):
    parsed = [parse_fields(line.replace(tag, "[AVBD_GATE]", 1))
              for line in output.splitlines() if line.startswith(tag)]
    return [] if any(errors for _, errors in parsed) else [fields for fields, _ in parsed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default=str(ROOT / "tools/benchmark-results/local-budget"))
    parser.add_argument("--contacts", action="store_true", help="Run NP, face moment, sliding, common motion and moving boundary oracles")
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--threads", type=int, nargs="+", default=[0, 4])
    args = parser.parse_args()
    from pathlib import Path
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    base = {k: v for k, v in os.environ.items()
            if not k.startswith(("PHYSX_AVBD_", "PHYSX_NP_GEOMETRY_", "PHYSX_SNIPPET_"))}
    base.update(PHYSX_AVBD_COMMON_STEP="budget", PHYSX_AVBD_COMMON_STATS="1",
                PHYSX_NP_GEOMETRY_CERTIFICATES="2")
    results = []
    budgets = args.budgets or ([32] if args.contacts else [8, 32])
    if any(n < 1 or n > 128 for n in budgets) or any(w < 0 or w > 16 for w in args.threads):
        parser.error("budgets must be 1..128 and workers 0..16")
    modes = (0, 2, 3, 4, 5, 12, 13, 14, 15, 16) if args.contacts else (6, 8, 10, 11)
    for threads in args.threads:
        for budget in budgets:
            for mode in modes:
                env = dict(base, PHYSX_AVBD_LOCAL_ITERATIONS=str(budget))
                cmd = [sys.executable, str(ROOT / "tools/run_avbd_common_step_probe.py"),
                       "--child", str(mode), "--threads", str(threads), "--chain-bodies", "256"]
                run = subprocess.run(cmd, env=env, capture_output=True, timeout=60)
                output = (run.stdout + run.stderr).decode("utf-8", errors="replace")
                name = f"mode{mode}-n{budget}-w{threads}"
                (logs / (name + ".log")).write_text(output, encoding="utf-8")
                frames = records(output, "[AVBD_COMMON_FRAME]")
                ledgers = records(output, "[AVBD_LOCAL_BUDGET]")
                expected_frames = 30 if mode == 12 else 120 if mode >= 13 else 1
                ok = (run.returncode == 0 and len(frames) == expected_frames and bool(ledgers)
                      and "status=FAIL" not in output and "status=UNSUPPORTED" not in output
                      and all(f.get("status") == "ACCEPTED" and f.get("bdf") == "1"
                              and f.get("poseForcePublications") == "1" for f in frames)
                      and all(int(f.get("iterations", -1)) == budget
                              and f.get("strictStationarity") == "NOT_REQUIRED"
                              and 0<=int(f.get("colors", -1)) <= int(f.get("primalBlocks",0))//budget for f in ledgers))
                if args.contacts:
                    ok = ok and sum(int(f.get("geometryQueries", 0)) for f in ledgers) > 0
                if mode == 16:
                    ok = ok and any(int(f.get("geometryQueries", 0)) >= 64 for f in ledgers)
                    if threads:
                        ok = ok and any(int(f.get("peakTasks", 0)) > 1 for f in ledgers)
                results.append(dict(case=name, passed=ok, kind="physical", frames=len(frames)))
                print(f"[LOCAL_BUDGET_TEST] case={name} status={'PASS' if ok else 'FAIL'}", flush=True)
    # High incident degree is covered by run_avbd_row_tiles_probe.py using
    # analytic physical oracles; permanent rejection is no longer the contract.
    for mode, budget, reason in ((1, "8", "contact_geometry_or_policy_uncompiled"), (9, "0", "invalid_iteration_budget")):
        env = dict(base, PHYSX_AVBD_LOCAL_ITERATIONS=budget)
        run = subprocess.run([sys.executable, str(ROOT / "tools/run_avbd_common_step_probe.py"),
                              "--child", str(mode), "--threads", "4"],
                             env=env, capture_output=True, timeout=60)
        output = (run.stdout + run.stderr).decode("utf-8", errors="replace")
        name = f"unsupported-{reason}"
        (logs / (name + ".log")).write_text(output, encoding="utf-8")
        frames = records(output, "[AVBD_COMMON_FRAME]")
        ok = ("status=UNSUPPORTED" in output and reason in output
              and not any(f.get("poseForcePublications") == "1" or f.get("bdf") == "1" for f in frames)
              and (mode != 9 or (run.returncode == 0 and "unpublished=PASS" in output)))
        results.append(dict(case=name, passed=ok, kind="unsupported"))
        print(f"[LOCAL_BUDGET_TEST] case={name} status={'PASS' if ok else 'FAIL'}", flush=True)
    (logs / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return int(any(not r["passed"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
