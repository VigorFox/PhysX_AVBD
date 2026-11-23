"""Paired, headless, 360-frame acceptance for both authored deformable snippets."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def cpu_ticks(process=None):
    # Windows cumulative times in 100 ns units. GetSystemTimes' kernel time
    # includes idle time. A terminated child's process handle is still valid.
    if os.name != 'nt':
        return None
    times = [ctypes.c_ulonglong() for _ in range(3 if process is None else 4)]
    pointers = [ctypes.byref(t) for t in times]
    if process is None:
        if not ctypes.windll.kernel32.GetSystemTimes(*pointers):
            raise ctypes.WinError()
        idle, kernel, user = (t.value for t in times)
        return kernel + user, kernel + user - idle
    if not ctypes.windll.kernel32.GetProcessTimes(ctypes.c_void_p(int(process._handle)), *pointers):
        raise ctypes.WinError()
    return times[2].value + times[3].value


def records(text):
    return {line.split()[0]: dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)
            for line in text.splitlines() if line.startswith('[')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scene', choices=('cloth', 'volume', 'both'), default='both')
    parser.add_argument('--cloth-case', choices=('static-box', 'static-box-25'), default='static-box',
                        help='Use static-box-25 for a 4.8 m Particle3 / Rigid6 cloth comparison.')
    args = parser.parse_args()
    sdk = Path(os.environ['AVBD_PROBE_BINARY_DIR']).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    scenes = {'cloth': ('Surface', 'static-box'), 'volume': ('Volume', 'visual-showcase')}
    def identity():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sdk.iterdir()
                if p.suffix.lower() in ('.dll', '.exe')}
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PHYSX_AVBD_', 'PHYSX_NP_', 'PHYSX_SNIPPET_'))}
    env.update(PHYSX_AVBD_COMMON_STEP='budget', PHYSX_AVBD_LOCAL_ITERATIONS='4')
    report = dict(before=identity(), clothCase=args.cloth_case, runs=[], scenes={})
    def save():
        (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    for scene, (suffix, case) in scenes.items():
        if args.scene not in (scene, 'both'):
            continue
        medians = {}
        # Each repetition visits all worker counts, rotating their positions
        # in time so external load cannot systematically favor one count.
        counts=(0,4,8)
        for repeat in range(3):
            for workers in counts[repeat:]+counts[:repeat]:
                for model in (('rigid6', 'particle3') if repeat%2 == 0 else ('particle3', 'rigid6')):
                    exe = sdk/f'SnippetDeformable{suffix}AVBD_64.exe'
                    scene_case = args.cloth_case if scene == 'cloth' and model == 'particle3' else case
                    cmd = [str(exe), '--headless', '--case='+scene_case, '--frames=360', f'--dispatcher-threads={workers}']
                    if model == 'rigid6':
                        cmd.append('--rigid6')
                    path = args.output/f'{scene}-{workers}-{repeat}-{model}.log'
                    with path.open('wb') as stream:
                        start=cpu_ticks()
                        with subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT) as child:
                            try:
                                code=child.wait(timeout=600)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                raise
                            end=cpu_ticks()
                            other=None if not start else 100*max(0,end[1]-start[1]-cpu_ticks(child))/max(1,end[0]-start[0])
                    raw=path.read_bytes()
                    tags = records(raw.decode('utf8', errors='replace'))
                    gate = tags.get('[AVBD_GATE]', {})
                    route = tags.get('[AVBD_SURFACE_ROUTE]' if scene=='cloth' else '[AVBD_VOLUME_ROUTE]', {}).get('model')
                    expected = 'particle3' if model=='particle3' else ('joint_sheet' if scene=='cloth' else 'joint')
                    metrics = (gate if model=='rigid6' else tags.get('[AVBD_SURFACE_METRICS]' if scene=='cloth' else '[VOLUME_STEP_TIMING]', {}))
                    mean = float(metrics.get('averageStepMs', metrics.get('meanStepMs', metrics.get('meanMs', '0'))))
                    p95 = float(metrics.get('p95StepMs', metrics.get('p95Ms', '0')))
                    passed = route==expected and code==0 and gate.get('status')=='PASS' and gate.get('completedFrames')=='360' and mean>0 and p95>0
                    report['runs'].append(dict(scene=scene, model=model, workers=workers, repeat=repeat,
                        meanMs=mean, p95Ms=p95, otherCpuPercent=other, passed=passed, route=route, log=path.name, gate=gate))
                    print(path.name, passed, mean, p95, 'otherCpuPercent', other, flush=True)
                    save()
        for workers in counts:
            for model in ('rigid6','particle3'):
                rows = [r for r in report['runs'] if (r['scene'],r['model'],r['workers'])==(scene,model,workers)]
                medians[f'{model}-{workers}']={k:statistics.median(r[k] for r in rows) for k in ('meanMs','p95Ms')}
        correct=all(r['passed'] for r in report['runs'] if r['scene']==scene)
        cost=correct and all(medians[f'particle3-{w}'][k]<=medians[f'rigid6-{w}'][k] for w in (0,4,8) for k in ('meanMs','p95Ms'))
        speed=lambda m,w:medians[f'{m}-0']['meanMs']/medians[f'{m}-{w}']['meanMs']
        scaling={w:speed('particle3',w)/speed('rigid6',w) for w in (4,8)} if correct else {}
        report['scenes'][scene]=dict(medians=medians,correct=correct,costPassed=cost,relativeScaling=scaling,
            passed=cost and min(scaling.values())>=1)
    report['stableBinaries']=report['before']==identity()
    save()
    return 0 if report['stableBinaries'] and all(v['passed'] for v in report['scenes'].values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
