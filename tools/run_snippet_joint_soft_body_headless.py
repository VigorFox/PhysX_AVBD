#!/usr/bin/env python3
"""Validate the experimental rigid-cell / finite-joint softbody via headless."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import avbd_gate

ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_POLICY = "soft-body-layered-20261023"
LEGACY_CASES = ("free-flight", "drop", "showcase", "compression-recovery", "reaction", "compression-load", "wake-recovery")
MULTI_CASES = ("multi-collision", "visual-showcase")
CASES = LEGACY_CASES + MULTI_CASES
MULTI_NUMERIC = ("frames", "completedFrames", "grid", "nodes", "joints", "mass", "cellSize", "dispatcherThreads", "bodies",
    "contacts", "crossContacts", "crossMask", "minCellDet", "minVolumeRatio", "maxVolumeRatio", "finalVolumeRatio",
    "maxExternalPenetration", "maxSelfPenetration", "maxJointAnchorError", "maxPostContactStep", "maxMomentumRelativeError",
    "initialMechanicalEstimate", "maxMechanicalEstimate", "finalMechanicalEstimate", "finalAwake", "skinRestError",
    "massRelativeError", "assetBytes", "skinBytes", "diagnosticBytes", "averageStepMs", "p95StepMs", "maxStepMs",
    "activeContactFrames", "averageActiveContactMs", "averageSkinMs", "averageObserverMs", "averageFrameMs",
    "nonFinite", "fetchFailures", "fatalErrors")
NUMERIC = (
    "frames", "completedFrames", "grid", "nodes", "joints", "mass", "cellSize",
    "dispatcherThreads", "compressionHoldFrames", "maxJointAngle", "minVolumeRatio", "minCellDet", "finalVolumeRatio",
    "maxShapeError", "finalShapeError", "maxGroundPenetration", "maxExternalPenetration",
    "compressionMinimum", "recoveredFraction", "contacts", "firstContactFrame",
    "maxPostContactStep", "zeroMotionFrames", "systemComDrift", "maxPostImpactVy",
    "maxMechanicalEstimate", "finalMechanicalEstimate", "averageStepMs", "maxStepMs",
    "averageFrameMs", "activeFrames", "activeContactFrames", "averageActiveStepMs",
    "averageActiveContactMs", "maxMomentumRelativeError", "maxBallisticError",
    "sleepingBeforeWake", "maxWakeMotion", "nonFinite", "fetchFailures", "fatalErrors",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_kernel_observation(output):
    """Count real main-loop execution; absence is not inferred as coverage."""
    keys = ("attempts", "unsupported", "solved", "accepted", "products")
    contact_keys = ("frictionAttempts", "frictionAccepted", "sharedContactAttempts", "sharedContactAccepted")
    contact_totals = dict.fromkeys(contact_keys, 0)
    contact_samples = 0
    totals = dict.fromkeys(keys, 0)
    samples, issues = 0, []
    for line in output.splitlines():
        if not line.startswith("[AVBD_GROUP_KERNEL] "):
            continue
        samples += 1
        fields = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
        try:
            values = {key: int(fields[key]) for key in keys}
            if fields.get("implementation") != "shared-linear-response":
                raise ValueError("unknown kernel")
            if any(value < 0 for value in values.values()):
                raise ValueError("negative counter")
            if values["solved"] + values["unsupported"] > values["attempts"] or values["accepted"] > values["solved"]:
                raise ValueError("inconsistent counters")
            if any(key in fields for key in contact_keys):
                contact = {key: int(fields[key]) for key in contact_keys}
                if any(value < 0 for value in contact.values()):
                    raise ValueError("negative contact counter")
                for prefix in ("friction", "sharedContact"):
                    if contact[prefix + "Accepted"] > min(contact[prefix + "Attempts"], values["accepted"]) or contact[prefix + "Attempts"] > values["attempts"]:
                        raise ValueError("inconsistent contact counters")
                for key in contact_keys:
                    contact_totals[key] += contact[key]
                contact_samples += 1
        except (KeyError, ValueError) as error:
            issues.append(str(error))
            continue
        for key in keys:
            totals[key] += values[key]
    return {
        "available": bool(samples), "valid": bool(samples) and not issues,
        "scope": "existing material-group primal only; no mixed contact KKT takeover inferred",
        "samples": samples, "totals": totals, "issues": issues,
        "contactSamples": contact_samples, "contactTotals": contact_totals if contact_samples else None,
        # This also includes failures during assembly; do not call it a
        # measured linear-solver failure count without launch instrumentation.
        "otherUnsolved": totals["attempts"] - totals["unsupported"] - totals["solved"],
        "candidateRejected": totals["solved"] - totals["accepted"],
    }


def layered_qualification(record, requested_frames, timed_out=False):
    """Annotate existing evidence without changing the legacy gate verdict."""
    fields = record.get("fields", {})
    status, reason = fields.get("status"), fields.get("reason")
    quality_reasons = {"shape_or_volume", "penetration", "self_penetration", "free_flight",
                       "contact_motion", "recovery", "reaction", "wake_recovery", "asset_contract",
                       "cross_contact_motion", "energy_or_reaction"}
    physical_fail = status == "FAIL" and reason in quality_reasons

    def numeric(key):
        value = fields.get(key)
        return float(value) if isinstance(value, str) and avbd_gate.is_finite_token(value) else None

    issues = []
    runtime_failure = timed_out or reason in {"incomplete", "runtime", "runtime_error", "initialization_or_fetch"}
    if timed_out:
        issues.append("process timed out")
    if record.get("gate_count") != 1:
        issues.append("missing or ambiguous GATE")
    if numeric("frames") != requested_frames or numeric("completedFrames") != requested_frames:
        issues.append("requested frames were not completely observed")
    for key in ("nonFinite", "fetchFailures", "fatalErrors"):
        value = numeric(key)
        if value != 0:
            issues.append("missing, invalid or nonzero " + key)
            runtime_failure |= value is not None and value != 0
    allowed_errors = {"status='FAIL', expected PASS", "exit=1"} if physical_fail else set()
    issues.extend(error for error in record.get("errors", []) if error not in allowed_errors)
    expected_exit = 1 if physical_fail else 0
    if record.get("exit") != expected_exit:
        issues.append("exit code does not describe a completed scene-quality observation")
        runtime_failure |= record.get("exit") not in (None, 0, 1)
    if not ((status == "PASS" and reason == "none") or physical_fail):
        issues.append("GATE is not a completed scene-quality result")
    if record.get("capability") != "EXPERIMENTAL":
        issues.append("requested experimental capability was not observed")
    runtime_status = "FAILED" if runtime_failure else "NOT_EVALUATED" if issues else "OBSERVED_COMPLETE"
    quality_status = status if runtime_status == "OBSERVED_COMPLETE" else "NOT_EVALUATED"
    source = "SnippetJointSoftBodyMultiHeadless.cpp" if fields.get("case") in MULTI_CASES else "SnippetJointSoftBodyHeadless.cpp"
    metrics = {}
    for key in ("averageStepMs", "p95StepMs", "maxStepMs", "averageActiveStepMs",
                "averageActiveContactMs", "averageFrameMs"):
        value = numeric(key)
        if value is None or value < 0:
            continue
        count_key = {"averageActiveStepMs": "activeFrames", "averageActiveContactMs": "activeContactFrames"}.get(key)
        samples = numeric(count_key) if count_key else numeric("completedFrames")
        available = samples is not None and samples > 0
        metrics[key] = {"observedMs": value, "hasSamples": available,
                        "withinReferenceBudget": value <= 16.67 if available else None}
    return {
        "runtimeObservation": {"status": runtime_status, "scope": "process completion and published finite/error counters only",
            "fullBasicContractQualification": "NOT_ASSESSED", "requestedFrames": requested_frames,
            "completedFrames": numeric("completedFrames"), "issues": issues,
            "source": "process exit and existing AVBD_GATE; no derivative/API/force-limit qualification inferred"},
        "legacySceneQuality": {"status": quality_status, "gateStatus": status, "gateReason": reason,
            "source": source, "criteria": "existing scene-specific gate, unchanged",
            "fullReplacementQualification": "NOT_ASSESSED"},
        "performance": {"status": "OBSERVATIONS_ONLY" if metrics else "NOT_EVALUATED",
            "referenceStepBudgetMs": 16.67, "metrics": metrics, "source": "existing AVBD_GATE timing fields",
            "completedRuntimeObservation": runtime_status == "OBSERVED_COMPLETE",
            "legacySceneQualityPassed": quality_status == "PASS", "isolatedBenchmark": False,
            "isolationScope": "runner does not enforce an isolated timing environment",
            "realtimeQualification": "NOT_ASSESSED"},
        "research": {"status": "not_assessed", "engineeringRequirement": "not_required_for_engineering",
            "scope": "strict global stationarity, continuous geometric guarantees and complete work ledger"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-material", action="store_true")
    parser.add_argument("--np-domain", choices=("off", "observe", "trial"), default="off",
                        help="NP support experiment; trial uses an OGC offset normal law and remains unqualified for general articulation/offset behavior")
    parser.add_argument("--disable-sleep", action="store_true", help="Separate reaction conservation diagnostic; does not replace the default sleep-enabled gate")
    parser.add_argument("--config", choices=("checked", "debug", "release"), default="checked")
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--solver", choices=("avbd", "pgs", "tgs"), default="avbd")
    parser.add_argument("--cases", default="gated")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--dispatcher-threads", type=int, default=4)
    parser.add_argument("--compression-hold-frames", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    error, cases = avbd_gate.resolve_selection(args.cases, CASES, {"gated": CASES, "all": CASES, "legacy": LEGACY_CASES, "multi": MULTI_CASES})
    if error or args.frames < 360 or not 3 <= args.grid <= 10 or not 0 <= args.dispatcher_threads <= 16 or args.timeout <= 0 or not 60 <= args.compression_hold_frames <= 10000:
        parser.error(error or "Require frames >= 360, grid 3..10, workers 0..16 and positive timeout")
    if any(case.startswith("compression-") for case in cases) and args.frames < 300 + args.compression_hold_frames:
        parser.error("Compression requires >= 180 recovery frames after the authored hold")
    if args.disable_sleep and cases != ["reaction"]:
        parser.error("--disable-sleep requires --cases reaction")
    bin_dir = (args.bin_dir or ROOT / "physx/bin/win.x86_64.vc143.md" / args.config).resolve()
    exe = bin_dir / "SnippetJointSoftBody_64.exe"
    if not exe.is_file():
        parser.error("Missing executable: " + str(exe))
    report = {"model": "rigid_cell_joint", "experimental": True, "acceptancePolicy": ACCEPTANCE_POLICY,
              "npNormalLaw": "ogc-envelope-eligible-pairs-20261026" if args.np_domain == "trial" else "position-al",
              "physxCommonSha256": sha256(exe.parent / "PhysXCommon_64.dll"),
              "npDomainMode": args.np_domain, "executable": str(exe), "exeSha256": sha256(exe),
              "physxSha256": sha256(bin_dir / "PhysX_64.dll"), "cases": []}
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    env["PHYSX_NP_GEOMETRY_CERTIFICATES"] = str({"off": 0, "observe": 1, "trial": 2}[args.np_domain])
    if args.np_domain != "off":
        env["PHYSX_AVBD_SCHEDULE_STATS"] = "1"
    for case in cases:
        argv = [str(exe), "--headless", "--case=" + case, "--solver=" + args.solver,
                "--frames=" + str(args.frames), "--grid=" + str(args.grid),
                "--dispatcher-threads=" + str(args.dispatcher_threads),
                "--compression-hold-frames=" + str(args.compression_hold_frames)]
        if args.volume_material:
            argv.append("--volume-material")
        if args.disable_sleep:
            argv.append("--disable-sleep")
        start = time.perf_counter()
        proc = subprocess.Popen(argv, cwd=bin_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timed_out = False
        try:
            out, err = proc.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            out, err = proc.communicate()
        text = out.decode("utf-8", errors="replace") + err.decode("utf-8", errors="replace")
        numeric = MULTI_NUMERIC if case in MULTI_CASES else NUMERIC
        ok, record = avbd_gate.evaluate_gate_text(text, {
            "case": case, "solver": args.solver, "snippet": "SnippetJointSoftBody",
            "frames": args.frames, "completedFrames": args.frames,
            "capability": "EXPERIMENTAL", "required_fields": numeric + ("model",),
            "numeric_fields": numeric})
        if proc.returncode != 0 or timed_out:
            record["errors"].append("timeout" if timed_out else "exit=" + str(proc.returncode))
        fields = record["fields"]
        expected = {"grid": args.grid, "dispatcherThreads": args.dispatcher_threads,
                    "fetchFailures": 0, "model": "convex_cell_joint" if case == "visual-showcase" else "rigid_cell_joint", "validation": "GATED"}
        if args.volume_material:
            expected["model"] += "_volume"
        if case in LEGACY_CASES:
            expected.update(sleepEnabled=int(not args.disable_sleep), materialElements=6*(args.grid-1)**3 if args.volume_material else 0)
            expected.update(compressionHoldFrames=args.compression_hold_frames, nodes=args.grid ** 3, joints=3 * args.grid ** 2 * (args.grid - 1))
        elif case == "multi-collision":
            expected.update(materialElements=12*(args.grid-1)**3 if args.volume_material else 0)
            expected.update(bodies=2, nodes=2 * args.grid ** 3, joints=6 * args.grid ** 2 * (args.grid - 1))
        else:
            expected.update(bodies=5)
            try:
                material_count=int(fields.get("materialElements", "-1"))
                if (args.volume_material and material_count <= 0) or (not args.volume_material and material_count != 0):
                    record["errors"].append("material element count does not match the requested model")
            except ValueError:
                record["errors"].append("invalid material element count")
        for key, value in expected.items():
            if fields.get(key) != str(value):
                record["errors"].append(key + " differs from launch/contract")
        record.update(ok=not record["errors"], argv=argv, exit=proc.returncode,
                      wallSeconds=time.perf_counter() - start, output=text)
        record["groupKernel"] = group_kernel_observation(text)
        record["qualification"] = layered_qualification(record, args.frames, timed_out)
        report["cases"].append(record)
        print("[JOINT_SOFT_HEADLESS] case=%s solver=%s result=%s" % (case, args.solver, "PASS" if record["ok"] else "FAIL"), flush=True)
        print(text.rstrip(), flush=True)
        for error in record["errors"]:
            print("[JOINT_SOFT_HEADLESS_ERROR] " + error, flush=True)
    report["passed"] = all(record["ok"] for record in report["cases"])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
