"""Real Particle3/Rigid6 four-iteration flight comparison, not contact acceptance.

Same 4.8 m sheet, approximately 625 kg, dt=1/60 and translation velocity.
Rigid6 has 576 cell frames; Particle3 has 625 nodes. Constitutive models differ.
Keep all pairs, cold first steps, failed gates, separate instrumented DOF audit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import psutil
from benchmark_avbd_native_parallel import process_cpu_seconds, telemetry_fields


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rigid6', type=Path, required=True)
    p.add_argument('--particle3', type=Path, required=True)
    p.add_argument('--legacy-rigid6-default', action='store_true', help='Use an older Rigid6 build that predates --rigid6')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=180)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if a.frames < 60 or a.repeats < 3:
        p.error('Use at least 60 frames and three pairs')
    a.output.mkdir(parents=True, exist_ok=False)
    products = dict(rigid6=a.rigid6.resolve(), particle3=a.particle3.resolve())
    def identity():
        return {k: {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in sorted(exe.parent.glob('*')) if f.suffix in ('.exe', '.dll')}
                for k, exe in products.items()}
    report = dict(scope='contact-free translation only; NOT production acceptance',
                  frames=a.frames, iterations=4, before=identity(), runs=[], audits=[])
    def compilers():
        return [{'pid': proc.info['pid'], 'name': proc.info['name']} for proc in
                psutil.process_iter(['pid','name']) if (proc.info['name'] or '').lower() in
                ('cl.exe','clang-cl.exe','unrealbuildtool.exe','link.exe')]
    def run(model, workers, repeat, diagnostic=False):
        env = {k: v for k, v in os.environ.items() if not k.startswith(
            ('PHYSX_AVBD_', 'PHYSX_SNIPPET_', 'PHYSX_NP_GEOMETRY_'))}
        env.update(PHYSX_AVBD_COMMON_STEP='budget', PHYSX_AVBD_LOCAL_ITERATIONS='4')
        frames = 60 if diagnostic else a.frames
        if diagnostic:
            env['PHYSX_AVBD_COMMON_STATS'] = '1'
        command = [str(products[model]), '--headless', '--case=free-flight',
                   f'--frames={frames}', f'--dispatcher-threads={workers}']
        if model == 'particle3':
            command.append('--particle3')
        elif not a.legacy_rigid6_default:
            command.append('--rigid6')
        log = a.output / f'{model}-{workers}-{repeat}-{int(diagnostic)}.log'
        start = time.perf_counter()
        competing_before=compilers()
        with log.open('wb') as stream:
            child = subprocess.Popen(command, cwd=products[model].parent, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(); code = 124
            cpu = process_cpu_seconds(child)
        wall = time.perf_counter()-start
        lines = log.read_text(encoding='utf8', errors='replace').splitlines()
        def records(tag):
            return [telemetry_fields(line) for line in lines if line.startswith('['+tag+'] ')]
        gates, metrics, dofs, budgets = records('AVBD_GATE'), records('AVBD_SURFACE_METRICS'), records('AVBD_BLOCK_DOF'), records('AVBD_LOCAL_BUDGET')
        gate = gates[0] if len(gates)==1 else {}
        valid = code==0 and gate.get('status')=='PASS' and gate.get('completedFrames')==str(frames)
        routes = records('AVBD_SURFACE_ROUTE')
        valid &= len(routes)==1 and routes[0].get('model')==('particle3' if model=='particle3' else 'joint_sheet')
        if diagnostic:
            valid &= len(dofs)==frames and all(d.get('rigid6')=='0' and d.get('particle3')=='625'
                and d.get('physicalDof')=='1875' and (d.get('nodalSolves')=='2500' or
                (d.get('nodalMaterialRows')=='1152' and d.get('membranePrimal')=='4608')) for d in dofs)
            valid &= len(budgets)==frames and all(b.get('budget')=='4' and b.get('iterations')=='4'
                and b.get('workers')==str(workers) and b.get('status')=='ACCEPTED' for b in budgets)
        metric = metrics[0] if len(metrics)==1 else gate
        result = dict(model=model, workers=workers, repeat=repeat, diagnostic=diagnostic, passed=bool(valid),
                      exit=code, command=command, gate=gate, log=str(log),
                      meanMs=float(metric.get('meanStepMs', metric.get('averageStepMs','nan'))),
                      p95Ms=float(metric.get('p95StepMs','nan')), cpuMsPerFrame=1000*cpu/frames,
                      processCpuSeconds=cpu, processWallSeconds=wall)
        result['compilersBefore']=competing_before
        result['compilersAfter']=compilers()
        return result
    for workers in (0,4,8):
        report['audits'].append(run('particle3',workers,0,True))
    for repeat in range(1,a.repeats+1):
        for workers in ((0,4,8) if repeat%2 else (8,4,0)):
            for model in (('rigid6','particle3') if repeat%2 else ('particle3','rigid6')):
                r=run(model,workers,repeat);report['runs'].append(r)
                print(model,workers,repeat,r['passed'],r['meanMs'],flush=True)
                (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['summary']={}
    for workers in (0,4,8):
        s={}
        for model in products:
            runs=[r for r in report['runs'] if r['model']==model and r['workers']==workers]
            s[model]={key:statistics.median(r[key] for r in runs) for key in ('meanMs','p95Ms','cpuMsPerFrame')}
        s['meanRatio']=s['particle3']['meanMs']/s['rigid6']['meanMs']
        s['p95Ratio']=s['particle3']['p95Ms']/s['rigid6']['p95Ms']
        report['summary'][workers]=s
    report['identityStable']=identity()==report['before']
    report['physicalPassed']=all(r['passed'] for r in report['runs']+report['audits'])
    report['measuredCostWithinBaseline']=all(s['meanRatio']<=1 and s['p95Ratio']<=1 for s in report['summary'].values())
    report['parallelScaling']={model:{w:report['summary'][0][model]['meanMs']/report['summary'][w][model]['meanMs']
                                      for w in (4,8)} for model in products}
    report['relativeScaling']={w:report['parallelScaling']['particle3'][w]/report['parallelScaling']['rigid6'][w] for w in (4,8)}
    report['environmentClean']=not any(r['compilersBefore'] or r['compilersAfter'] for r in report['runs'])
    report['performanceQualified']=report['environmentClean'] and report['physicalPassed'] and report['identityStable'] and report['measuredCostWithinBaseline'] and min(report['relativeScaling'].values())>=1
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('before','runs','audits')},indent=2))
    return 0 if report['performanceQualified'] else 1


if __name__=='__main__':
    raise SystemExit(main())
