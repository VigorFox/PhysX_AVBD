#!/usr/bin/env python3
"""Summarize completed official baseline observations; assign no physical verdict."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("missing or nonfinite numeric field: " + name)
    return value


def vector(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("invalid vector: " + name)
    return [finite(x, name) for x in value]


def norm(values):
    return math.sqrt(math.fsum(x * x for x in values))


def statistics(values):
    if not values:
        return {"samples": 0, "mean": None, "p95": None, "p99": None, "max": None, "sum": 0}
    ordered = sorted(values)
    return {"samples": len(values), "mean": math.fsum(values) / len(values),
            "p95": ordered[math.ceil(.95 * (len(values) - 1))],
            "p99": ordered[math.ceil(.99 * (len(values) - 1))],
            "max": ordered[-1], "sum": math.fsum(values)}


def exceedance(values, threshold, dt):
    total = longest = current = 0
    for value in values:
        current = current + 1 if value > threshold else 0
        total += value > threshold
        longest = max(longest, current)
    return {"strictlyGreaterThanM": threshold, "frames": total, "seconds": total * dt,
            "longestConsecutiveFrames": longest, "longestConsecutiveSeconds": longest * dt}


def depth_summary(frames, key, dt):
    values = [finite(f.get(key), key) for f in frames]
    if any(v < 0 for v in values):
        raise ValueError("negative depth observation: " + key)
    stats = statistics(values)
    tail = values[-60:]
    return {"applicable": True, "units": "m", "peak": stats["max"],
            "firstPeakFrame": frames[values.index(stats["max"])]["frame"],
            "frameP95": stats["p95"], "frameP99": stats["p99"],
            "positive": exceedance(values, 0, dt),
            "historical2mmReference": exceedance(values, .002, dt),
            "tail": {"frames": len(tail), "startFrame": frames[-len(tail)]["frame"],
                     "max": max(tail), "mean": math.fsum(tail) / len(tail)}}


def read_entries(path):
    entries, ignored = [], 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        if not line.lstrip().startswith("{"):
            ignored += 1  # SDK diagnostics are retained in the original JSONL/stderr.
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            raise ValueError("non-object observer record")
        entries.append(entry)
    return entries, ignored


def only(entries, kind):
    found = [e for e in entries if e.get("type") == kind]
    if len(found) != 1:
        raise ValueError("require exactly one " + kind)
    return found[0]


def summarize_run(record, path):
    entries, ignored = read_entries(path)
    config, summary = only(entries, "config"), only(entries, "summary")
    initial, geometry, cleanup = (only(entries, k) for k in ("initialState", "initialGeometry", "cleanup"))
    frames = [e for e in entries if e.get("type") == "frame"]
    count = config.get("frames")
    if not isinstance(count, int) or count < 1 or [f.get("frame") for f in frames] != list(range(1, count + 1)):
        raise ValueError("incomplete, duplicate or reordered frames")
    if (summary != record.get("summary") or summary.get("framesDone") != count or
        any(summary.get(k) is not True for k in ("completed", "finite", "executionOk", "measurementOk", "pairGeometryAllValid")) or
        summary.get("errors") != 0 or cleanup.get("ok") is not True or cleanup.get("errors") != 0 or
        geometry.get("valid") is not True):
        raise ValueError("summary/provenance, initial geometry or cleanup is incomplete/invalid")
    if any(f.get("finite") is not True or f.get("pairGeometryValid") is not True for f in frames):
        raise ValueError("invalid frame observation")
    dt = finite(config.get("dt"), "dt")
    if dt <= 0:
        raise ValueError("nonpositive dt")
    bodies = [e for e in entries if e.get("type") == "body"]
    if not bodies or [b.get("index") for b in bodies] != list(range(len(bodies))):
        raise ValueError("missing or unordered body descriptions")
    if len(initial.get("bodies", [])) != len(bodies) or any(len(f.get("bodies", [])) != len(bodies) for f in frames):
        raise ValueError("missing per-body observations")
    for frame in frames:
        if not isinstance(frame.get("active"), bool) or not isinstance(frame.get("sleepingBodies"), int):
            raise ValueError("missing activity or sleep observation")
        sleeping = [b.get("sleeping") for b in frame["bodies"]]
        if any(not isinstance(s, bool) for s in sleeping) or sum(sleeping) != frame["sleepingBodies"]:
            raise ValueError("inconsistent per-body sleeping observations")
    case, backend = config.get("case"), config.get("backend")
    if case not in ("drop", "five-cubes", "reaction") or backend not in ("rigid-cpu", "fem-gpu"):
        raise ValueError("unsupported observation schema")
    result = {"name": record["name"], "status": "OBSERVATIONS_DERIVED", "sourceJsonl": str(path),
              "sourceSha256": hashlib.sha256(path.read_bytes()).hexdigest(), "ignoredNonJsonLines": ignored,
              "backend": backend, "case": case, "rigidSolver": config.get("rigidSolver"),
              "deformableSolver": config.get("deformableSolver"), "frames": count, "dtSeconds": dt,
              "iterations": config.get("iterations"), "selfCollision": config.get("selfCollision"),
              "assetScope": config.get("assetScope"), "physicalQualification": "NOT_ASSIGNED"}
    result["groundPenetration"] = ({"applicable": False, "reason": "reaction has no ground"} if case == "reaction"
                                   else depth_summary(frames, "groundPenetration", dt))
    result["maxTetMtd"] = ({"applicable": False, "reason": "drop has only one body"} if case == "drop"
                           else depth_summary(frames, "maxTetMtd", dt))
    result["pairGeometryScope"] = config.get("pairGeometryScope")
    result["selfPenetration"] = {"status": "NOT_MEASURED"}
    deformation = {"applicable": False, "reason": "rigid boxes do not deform"}
    if backend == "fem-gpu":
        per_body = []
        for i in range(len(bodies)):
            samples = [(0, initial["bodies"][i])] + [(f["frame"], f["bodies"][i]) for f in frames]
            det = [(frame, finite(b.get("minDet"), "minDet")) for frame, b in samples]
            volume = [(frame, finite(b.get("volumeRatio"), "volumeRatio")) for frame, b in samples]
            min_det, min_volume = min(det, key=lambda p: p[1]), min(volume, key=lambda p: p[1])
            per_body.append({"body": i, "minDet": min_det[1], "firstMinDetFrame": min_det[0],
                             "minVolumeRatio": min_volume[1], "firstMinVolumeFrame": min_volume[0],
                             "finalVolumeRatio": volume[-1][1]})
        deformation = {"applicable": True, "includesInitialState": True, "bodies": per_body,
                       "minDet": min(b["minDet"] for b in per_body)}
    result["deformation"] = deformation
    reaction = {"applicable": False, "reason": "gravity and ground invalidate a zero-external-force comparison"}
    if case == "reaction":
        denominator = math.fsum(finite(b.get("mass"), "mass") * norm(vector(b.get("initialVelocity"), "initialVelocity")) for b in bodies)
        initial_vectors = [vector(b.get("momentum"), "initial momentum") for b in initial["bodies"]]
        initial_momentum = [math.fsum(p[k] for p in initial_vectors) for k in range(3)]
        deviations = [norm([x - y for x, y in zip(vector(f.get("momentum"), "momentum"), initial_momentum)]) for f in frames]
        peak = max(deviations)
        reaction = {"applicable": denominator > 0, "initialTotalMomentumNs": initial_momentum,
                    "normalization": "sum(body mass * norm(authored initial velocity)); no max(1, ...) floor",
                    "denominatorNs": denominator, "maxAbsoluteDeviationNs": peak,
                    "firstPeakFrame": frames[deviations.index(peak)]["frame"],
                    "maxRelativeDeviation": peak / denominator if denominator > 0 else None,
                    "finalRelativeDeviation": deviations[-1] / denominator if denominator > 0 else None,
                    "interpretation": "raw momentum change includes authored damping, settling and sleep; no force-balance verdict"}
        if denominator <= 0:
            reaction["reason"] = "zero normalization; relative deviation not defined"
    result["reactionMomentum"] = reaction
    active = [f for f in frames if f["active"]]
    timing = {"units": "ms", "activeFrames": len(active), "activeFramesAreContactFrames": False,
              "contactFrames": None, "contactFramesStatus": "NOT_MEASURED", "realtimeQualification": "NOT_ASSIGNED",
              "scope": "observedMs already includes step, readback and geometry; do not add these timings again",
              "isolation": "not established by this summarizer"}
    for key in ("stepMs", "readbackMs", "geometryMs", "observedMs"):
        values = [finite(f.get(key), key) for f in frames]
        if any(v < 0 for v in values):
            raise ValueError("negative timing: " + key)
        timing[key] = statistics(values)
    timing["activeStepMs"] = statistics([f["stepMs"] for f in active])
    result["timing"] = timing
    energy = [finite(f.get("kineticEnergy"), "kineticEnergy") + finite(f.get("gravitationalEnergy"), "gravitationalEnergy") for f in frames]
    initial_energy = math.fsum(finite(b.get("kineticEnergy"), "initial kineticEnergy") + finite(b.get("gravitationalEnergy"), "initial gravitationalEnergy") for b in initial["bodies"])
    result["partialEnergy"] = {"units": "J", "scope": "kinetic plus gravitational; elastic energy omitted",
                               "initial": initial_energy, "final": energy[-1], "maxIncludingInitial": max(initial_energy, max(energy)),
                               "completeEnergyQualification": "NOT_ASSESSED"}
    result["sleepingFinal"] = {"bodies": frames[-1]["sleepingBodies"], "totalBodies": len(bodies),
                               "perBody": [b["sleeping"] for b in frames[-1]["bodies"]]}
    return result


def summarize_report(report_path):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    results = []
    for record in report.get("runs", []):
        name = record.get("name", "")
        if record.get("completedMeasurement") is not True:
            results.append({"name": name, "status": "SKIPPED_INCOMPLETE_MEASUREMENT"})
            continue
        try:
            if not name or Path(name).name != name:
                raise ValueError("invalid run filename")
            results.append(summarize_run(record, report_path.parent / (name + ".jsonl")))
        except (OSError, ValueError, TypeError, KeyError) as error:
            results.append({"name": name, "status": "INVALID_COMPLETED_RECORD", "reason": str(error)})
    return {"schema": 1, "sourceReport": str(report_path),
            "sourceReportSha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "officialCommit": report.get("officialCommit"), "settings": report.get("settings"),
            "provenanceScope": "uses the source runner's build/DLL audit; does not independently rebuild or requalify it",
            "physicalQualification": "NOT_ASSIGNED", "automaticToleranceCalibration": False,
            "historicalDepthReferenceM": .002, "historicalReferenceIsAcceptanceThreshold": False,
            "percentileConvention": "sorted value at zero-based ceil(p * (N-1)), matching the observer",
            "durationConvention": "sampled output frame count * dt; not a continuous collision duration proof",
            "depthStatisticsExcludeInitialState": True, "runs": results,
            "summarizedRuns": sum(r["status"] == "OBSERVATIONS_DERIVED" for r in results),
            "skippedRuns": sum(r["status"] == "SKIPPED_INCOMPLETE_MEASUREMENT" for r in results),
            "invalidCompletedRuns": sum(r["status"] == "INVALID_COMPLETED_RECORD" for r in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_path", type=Path)
    args = parser.parse_args()
    try:
        result = summarize_report(args.report_path)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["summarizedRuns"] and not result["invalidCompletedRuns"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
