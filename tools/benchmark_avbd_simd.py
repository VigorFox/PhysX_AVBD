"""Interleave two Checked builds and report complete-step costs by frame window."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from benchmark_particle3_showcases import cpu_ticks


def identity(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir()) if p.suffix.lower() in ('.dll', '.exe')}


def costs(values):
    ordered = sorted(values)
    return dict(meanMs=statistics.mean(values), medianMs=statistics.median(values),
                p95Ms=ordered[(len(ordered)-1)*95//100], maxMs=ordered[-1])


def summarize_comparisons(report, workers, repeats):
    """Keep whole-step results even when an event was not observed in every run."""
    report['comparisons'] = []
    report['completeWindows'] = True
    for scene in sorted({r['scene'] for r in report['runs']}):
        for worker in workers:
            selected = [r for r in report['runs'] if (r['scene'], r['workers']) == (scene, worker)]
            if not selected or not all(r['passed'] for r in selected):
                continue
            paired = {(r['repeat'], r['label']): r for r in selected}
            expected = {(i, label) for i in range(repeats) for label in ('baseline', 'candidate')}
            if set(paired) != expected:
                report['completeWindows'] = False
                continue
            windows = sorted(set().union(*(set(r['windows']) | set(r.get('unavailableWindows', {})) for r in selected)))
            for window in windows:
                missing = [dict(repeat=r['repeat'], label=r['label'],
                                reason=r.get('unavailableWindows', {}).get(window, 'windowNotObserved'))
                           for r in selected if window not in r['windows']]
                if missing:
                    report['completeWindows'] = False
                    report['comparisons'].append(dict(scene=scene, workers=worker, window=window,
                                                       complete=False, missing=missing))
                    continue
                ratios = {key: [paired[(i, 'candidate')]['windows'][window][key] /
                               paired[(i, 'baseline')]['windows'][window][key] for i in range(repeats)]
                          for key in ('meanMs', 'p95Ms')}
                report['comparisons'].append(dict(scene=scene, workers=worker, window=window,
                    complete=True, pairedRatios=ratios,
                    medianRatios={key: statistics.median(v) for key, v in ratios.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=7)
    parser.add_argument('--frames', type=int, default=360)
    parser.add_argument('--workers', type=int, nargs='+', default=[0, 4, 8])
    parser.add_argument('--scene', choices=['cloth', 'volume', 'both'], default='both')
    parser.add_argument('--cloth-case', choices=['static-box', 'static-box-25'], default='static-box')
    args = parser.parse_args()
    if args.runs < 3 or args.frames < 90 or any(w < 0 for w in args.workers):
        parser.error('Use at least three pairs, 90 frames and nonnegative worker counts.')
    if args.scene != 'cloth' and args.frames < 360:
        parser.error('The five-body physical contract requires at least 360 frames.')
    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    directories = {key: getattr(args, key).resolve() for key in ('baseline', 'candidate')}
    before = {key: identity(path) for key, path in directories.items()}
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('PHYSX_AVBD_', 'PHYSX_NP_', 'PHYSX_SNIPPET_'))}
    env.update(PHYSX_AVBD_COMMON_STEP='budget', PHYSX_AVBD_LOCAL_ITERATIONS='4')
    report = dict(clothCase=args.cloth_case, binaryDirectories={k: str(v) for k, v in directories.items()},
                  before=before, runs=[], comparisons=[])
    save = lambda: (output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    for scene, suffix, case in [('cloth', 'Surface', 'static-box'), ('volume', 'Volume', 'visual-showcase')]:
        if args.scene not in (scene, 'both'):
            continue
        if scene == 'cloth':
            case = args.cloth_case
        for repeat in range(args.runs):
            workers = args.workers[repeat % len(args.workers):] + args.workers[:repeat % len(args.workers)]
            for worker in workers:
                order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
                for label in order:
                    name = f'{scene}-{worker}-{repeat}-{label}'
                    times_path = output/(name+'.csv')
                    command = [str(directories[label]/f'SnippetDeformable{suffix}AVBD_64.exe'),
                               '--headless', '--case='+case, f'--frames={args.frames}',
                               f'--dispatcher-threads={worker}']
                    with (output/(name+'.log')).open('wb') as stream:
                        system_before = cpu_ticks()
                        with subprocess.Popen(command, cwd=root,
                                env=dict(env, PHYSX_SNIPPET_STEP_TIMES=str(times_path)),
                                stdout=stream, stderr=subprocess.STDOUT) as child:
                            try:
                                code = child.wait(timeout=600)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                                raise
                            process_ticks = cpu_ticks(child)
                        system_after = cpu_ticks()
                    text = (output/(name+'.log')).read_text(encoding='utf8', errors='replace')
                    gates = [dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)
                             for line in text.splitlines() if line.startswith('[AVBD_GATE] ')]
                    passed = code == 0 and len(gates) == 1 and gates[0].get('status') == 'PASS'
                    passed &= bool(gates) and gates[0].get('completedFrames') == str(args.frames)
                    values = []
                    if times_path.is_file():
                        with times_path.open() as source:
                            rows = list(csv.DictReader(source))
                        passed &= [int(r['frame']) for r in rows] == list(range(1, args.frames+1))
                        values = [float(r['simulateFetchMs']) for r in rows]
                    passed &= len(values) == args.frames and all(0 < x < float('inf') for x in values)
                    windows = {}
                    unavailable = {}
                    if values:
                        for first, last in [(1, args.frames), (1, 39), (40, 90), (50, 66), (60, 90)]:
                            windows[f'{first}-{last}'] = costs(values[first-1:last])
                        for line in text.splitlines():
                            if not line.startswith('[FIVE_REBOUND] '):
                                continue
                            event = dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)
                            for key, length in [('contactFrame', 60), ('departureFrame', 30)]:
                                frame = int(event.get(key, '0'))
                                name_key = event['pair']+'-'+key
                                if frame <= 0 or frame > len(values):
                                    unavailable[name_key] = 'eventNotObserved'
                                elif frame+length > len(values):
                                    unavailable[name_key] = 'insufficientPostEventFrames'
                                else:
                                    first, last = max(1, frame-5), frame+length
                                    windows[name_key] = dict(costs(values[first-1:last]),
                                        firstFrame=first, lastFrame=last)
                    other = None
                    if system_before and process_ticks is not None:
                        other = 100*max(0, system_after[1]-system_before[1]-process_ticks)/max(1, system_after[0]-system_before[0])
                    report['runs'].append(dict(scene=scene, workers=worker, repeat=repeat,
                        label=label, passed=passed, exit=code, windows=windows, unavailableWindows=unavailable, gate=gates,
                        processCpuMs=process_ticks/10000 if process_ticks is not None else None,
                        otherCpuPercent=other))
                    save()
                    print(name, 'PASS' if passed else 'FAIL', windows.get(f'1-{args.frames}'), flush=True)
    summarize_comparisons(report, args.workers, args.runs)
    report['after'] = {key: identity(path) for key, path in directories.items()}
    report['stableBinaries'] = report['before'] == report['after']
    report['allRunsPassed'] = all(r['passed'] for r in report['runs'])
    save()
    return 0 if report['stableBinaries'] and report['allRunsPassed'] and report['completeWindows'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
