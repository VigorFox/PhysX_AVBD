"""Audit saved common-step traces and run small mechanism counterexamples.

These are mechanism checks, not full-five or GPU qualification. No SDK state
or acceptance threshold is changed. The C++ cases use production Table kernels.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from run_avbd_ogc_certificate_probe import ROOT, run_probe


def read_text(path):
    data = path.read_bytes()
    return data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8", errors="replace")


def trace_summary(path):
    steps, after_geometry, frames, first_violators = [], [], [], {}
    saw_geometry = False
    for line in read_text(path).splitlines():
        fields = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
        if line.startswith("[AVBD_COMMON_GEOMETRY]"):
            saw_geometry = True
            key = fields["a"] + "/" + fields["b"]
            first_violators[key] = first_violators.get(key, 0) + 1
        if line.startswith("[AVBD_COMMON_STEP]") and fields.get("status") == "PASS":
            value = {k: float(fields[k]) for k in ("fraction", "requested", "initial", "residual", "blocks")}
            steps.append(value)
            if saw_geometry:
                after_geometry.append(value)
        if line.startswith("[AVBD_COMMON_FRAME]"):
            frames.append(fields)
    return dict(file=str(path.relative_to(ROOT)), acceptedIterates=len(steps),
                acceptedFrames=sum(f.get("status") == "ACCEPTED" for f in frames),
                rejectedFrames=sum(f.get("status") == "REJECTED" for f in frames),
                unfinishedAtBudget=sum(f.get("stationary") == "0" for f in frames),
                maxBlocks=max((s["blocks"] for s in steps), default=0),
                afterFirstGeometryRejection=dict(iterates=len(after_geometry),
                    minimumFraction=min((s["fraction"] for s in after_geometry), default=None),
                    medianFraction=statistics.median(s["fraction"] for s in after_geometry) if after_geometry else None,
                    fractionAtMostOneOver1024=sum(s["fraction"] <= 1/1024 for s in after_geometry)),
                firstReportedViolatorCounts=first_violators,
                scope="Observed iterates, not a count of all violating pairs or successful scene frames")


def merit_counterexample():
    # E = 1/4 (x^2-1)^2 + 0.05 (x-xhat)^2. Its GN metric is SPD,
    # but the squared gradient is not a descent merit in negative curvature.
    x = xhat = .01
    energy = lambda q: .25*(q*q-1)**2 + .05*(q-xhat)**2
    gradient = lambda q: q*(q*q-1) + .1*(q-xhat)
    gn = 2*x*x + .1
    step = -gradient(x)/gn
    samples = [dict(alpha=2.**-i, energy=energy(x+step*2.**-i),
                    gradientSquared=gradient(x+step*2.**-i)**2) for i in range(16)]
    all_energy_descent = all(s["energy"] < energy(x) for s in samples)
    all_merit_increase = all(s["gradientSquared"] > gradient(x)**2 for s in samples)
    assert all_energy_descent and all_merit_increase
    return dict(x=x, gn=gn, trueHessian=3*x*x-1+.1, step=step,
                initialEnergy=energy(x), initialGradientSquared=gradient(x)**2,
                all16StepsDecreaseEnergy=all_energy_descent,
                all16StepsIncreaseStationarityMerit=all_merit_increase, samples=samples,
                scope="Analytic counterexample to GN direction plus monotone squared-residual acceptance; not a captured material")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=("precise", "fast"))
    parser.add_argument("--report", type=Path, default=ROOT / "tools/benchmark-results/stagnation-20261029/analysis.json")
    args = parser.parse_args()
    if args.child:
        return run_probe(ROOT / "tools/avbd_stagnation_mechanism_probe.cpp",
                         compiler_options=["/fp:" + args.child],
                         extra_sources=[ROOT / "physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp"])
    checks = {}
    for mode in ("precise", "fast"):
        result = subprocess.run([sys.executable, __file__, "--child", mode], capture_output=True, timeout=90)
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        print(output, end="", flush=True)
        if result.returncode:
            return result.returncode
        checks[mode] = output.splitlines()
    traces = [ROOT / "tools/benchmark-results/common-publication-20261028" / name
              for name in ("five-first16-final.log", "five-first16-v11.log")]
    result = dict(mechanismChecksPassed=True, sceneQualified=False, gpuImplementation=False,
                  sharedKernelChecks=checks, analyticMerit=merit_counterexample(),
                  traces=[trace_summary(p) for p in traces if p.is_file()],
                  missingTraces=[str(p) for p in traces if not p.is_file()],
                  disconnectedThrottle=dict(sharedAlpha=1/16384, unconstrainedStep=1,
                      globalAcceptedStep=1/16384, independentAcceptedStep=1,
                      scope="Two uncoupled components; demonstrates lost progress, not measured runtime speedup"))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["traces"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
