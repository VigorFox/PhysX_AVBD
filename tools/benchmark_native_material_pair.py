"""Interleave a fresh Rigid6 baseline and a material candidate at four iterations.

Every run keeps the original physical oracle. Wall time is unprofiled; process
CPU work, P95 and 0/4/8 scaling are reported separately. No trajectory equality.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import statistics
from benchmark_avbd_native_parallel import run, EXES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=6)
    p.add_argument('--frames', type=int, default=360)
    p.add_argument('--scenes', nargs='+', choices=['surface', 'volume'], default=['surface', 'volume'])
    p.add_argument('--previous-report', type=Path, help='Add samples to ALL runs of an unchanged earlier batch')
    p.add_argument('--cpus', type=int, nargs='+', help='Explicit identical logical CPU set inherited by both variants')
    args = p.parse_args()
    if args.repeats < 3 or args.frames < 180:
        p.error('Use at least three pairs and 180 frames')
    if args.cpus:
        import psutil
        current = psutil.Process()
        if len(set(args.cpus)) != len(args.cpus) or not set(args.cpus).issubset(current.cpu_affinity()):
            p.error('CPU IDs must be unique and allowed in the current process')
        current.cpu_affinity(args.cpus)
        if sorted(current.cpu_affinity()) != sorted(args.cpus):
            p.error('Could not apply the requested CPU set before launching children')
    args.output.mkdir(parents=True, exist_ok=False)
    products = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    def identity():
        return {label: {x.name: hashlib.sha256(x.read_bytes()).hexdigest()
                for x in sorted(directory.iterdir()) if x.suffix in ('.exe', '.dll')}
                for label, directory in products.items()}
    report = dict(iterations=4, frames=args.frames, cpus=args.cpus, before=identity(), runs=[], summary={})
    start_repeat = 1
    if args.previous_report:
        previous = json.loads(args.previous_report.read_text())
        if (not previous['identityStable'] or not previous.get('environmentClean', True)
                or previous['before'] != report['before']
                or previous['frames'] != args.frames or previous['iterations'] != 4
                or previous.get('cpus') != args.cpus
                or set(previous['summary']) != set(args.scenes)):
            p.error('Extension must retain the same binaries, scenes, budget and frames')
        report['runs'] = previous['runs']
        report['previousReport'] = str(args.previous_report.resolve())
        report['previousSha256'] = hashlib.sha256(args.previous_report.read_bytes()).hexdigest()
        start_repeat = max(r['repeat'] for r in report['runs']) + 1
    for scene in args.scenes:
        case = 'static-box' if scene == 'surface' else 'visual-showcase'
        for repeat in range(start_repeat, start_repeat + args.repeats):
            for workers in ([0, 4, 8] if repeat % 2 else [8, 4, 0]):
                for label in (['baseline', 'candidate'] if (repeat + workers // 4) % 2 else ['candidate', 'baseline']):
                    output = args.output / scene / label
                    output.mkdir(parents=True, exist_ok=True)
                    settings = SimpleNamespace(output=output, snippet=scene, case=case, frames=args.frames,
                        budget=4, reference_bounded=False, diagnostics=False, timeout=240, grid=None, cpus=args.cpus)
                    value = run(products[label] / EXES[scene], settings, workers, repeat)
                    value.update(scene=scene, label=label)
                    report['runs'].append(value)
                    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                    print(scene, label, workers, repeat, value['physicalPass'], value['averageStepMs'], flush=True)
        physical = all(r['physicalPass'] for r in report['runs'] if r['scene'] == scene)
        if not physical:
            report['summary'][scene] = dict(physical=False, cost=False, parallel=False,
                reason='A run failed its physical contract; no performance qualification')
            continue
        by = {(label, w): [r for r in report['runs'] if r['scene'] == scene and r['label'] == label and r['workers'] == w]
              for label in products for w in (0, 4, 8)}
        stats = {}
        for label in products:
            stats[label] = {w: dict(mean=statistics.median(r['averageStepMs'] for r in by[label, w]),
                p95=statistics.median(float(r['gate']['p95StepMs']) for r in by[label, w]),
                occupiedCores=statistics.median(r['averageOccupiedCores'] for r in by[label, w]),
                cpuMsPerFrame=statistics.median(r['processCpuSeconds'] * 1000 / args.frames for r in by[label, w]))
                for w in (0, 4, 8)}
            for w in (0, 4, 8):
                stats[label][w]['speedup'] = stats[label][0]['mean'] / stats[label][w]['mean']
        pairs = {}
        for w in (0, 4, 8):
            a = sorted(by['baseline', w], key=lambda r: r['repeat'])
            b = sorted(by['candidate', w], key=lambda r: r['repeat'])
            pairs[w] = dict(meanRatio=statistics.median(y['averageStepMs'] / x['averageStepMs'] for x, y in zip(a, b)),
                p95Ratio=statistics.median(float(y['gate']['p95StepMs']) / float(x['gate']['p95StepMs']) for x, y in zip(a, b)),
                cpuRatio=statistics.median(y['processCpuSeconds'] / x['processCpuSeconds'] for x, y in zip(a, b)),
                scalingRatio=stats['candidate'][w]['speedup'] / stats['baseline'][w]['speedup'])
        cost = all(pairs[w]['meanRatio'] <= 1 and pairs[w]['p95Ratio'] <= 1 for w in (4, 8))
        parallel = all(pairs[w]['scalingRatio'] >= 1 for w in (4, 8))
        report['summary'][scene] = dict(physical=physical, cost=cost, parallel=parallel, pairs=pairs, statistics=stats)
    report['after'] = identity()
    report['identityStable'] = report['before'] == report['after']
    report['qualified'] = report['identityStable'] and all(v['physical'] and v['cost'] and v['parallel'] for v in report['summary'].values())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2), flush=True)
    return 0 if report['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
