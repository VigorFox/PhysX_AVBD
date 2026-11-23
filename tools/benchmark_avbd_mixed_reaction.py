"""Alternating mechanism comparison; not cloth/five-body production qualification."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from benchmark_particle3_showcases import cpu_ticks
from run_avbd_ogc_certificate_probe import build_probe, retire_probe

ROOT = Path(__file__).resolve().parent.parent

def compare(work):
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sdk = Path(os.environ['AVBD_PROBE_BINARY_DIR']).resolve()
    soft_pair = '--soft-soft' in sys.argv[2:]
    sheet = next((name for name in ('reaction', 'strain-release') if '--sheet-'+name in sys.argv[2:]), None)
    identity = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(sdk.iterdir()) if p.suffix.lower() in ('.dll','.exe')}
    report = dict(scope=(('625-node sheet / 576-cell lattice '+sheet) if sheet else
                         'two 8-node actors' if soft_pair else '8-node actor and dynamic box') +
                  '; not default cloth/five-body acceptance',
                  sdk=str(sdk), before=identity(), runs=[])
    env = os.environ.copy()
    env["PATH"] = str(sdk) + os.pathsep + env.get("PATH", "")
    if sheet:
        code = build_probe(work, sdk, ROOT / 'tools/avbd_nodal_release_probe.cpp',
                           libraries=['PhysXExtensions_static_64.lib', 'PhysXPvdSDK_static_64.lib',
                                      'PhysXCooking_64.lib', 'PhysX_64.lib', 'PhysXCommon_64.lib'])
        if code:
            return code
        report['probeSha256'] = hashlib.sha256((work / 'probe.exe').read_bytes()).hexdigest()
    counts = (0, 4, 8)
    for repeat in range(3):
        for workers in counts[repeat:] + counts[:repeat]:
            for model in (('rigid6', 'particle3') if repeat % 2 == 0 else ('particle3', 'rigid6')):
                if sheet and model == 'rigid6':
                    command = [str(sdk / 'SnippetDeformableSurfaceAVBD_64.exe'), '--rigid6', '--headless', '--case='+sheet,
                               '--frames=180', f'--dispatcher-threads={workers}']
                elif sheet:
                    command = [str(work / 'probe.exe'), str(workers),
                               'impact' if sheet == 'reaction' else 'release']
                elif soft_pair and model == 'particle3':
                    command = [sys.executable, str(ROOT / 'tools/run_avbd_nodal_soft_contact_probe.py'), str(workers)]
                else:
                    command = [sys.executable, str(ROOT / 'tools/run_avbd_mixed_reaction_probe.py'),
                               'rigid6-soft' if soft_pair else model, str(workers)]
                start = cpu_ticks() if sheet else None
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as child:
                    try:
                        raw, _ = child.communicate(timeout=120)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.communicate()
                        raise
                    end = cpu_ticks() if start else None
                    other = 100 * max(0, end[1]-start[1]-cpu_ticks(child)) / max(1, end[0]-start[0]) if end else None
                    code = child.returncode
                name = f'{workers}-{repeat}-{model}.log'
                (output / name).write_bytes(raw)
                lines = (raw).decode('utf8', errors='replace').splitlines()
                record = next((dict(t.split('=', 1) for t in line.split()[1:] if '=' in t) for line in reversed(lines)
                               if line.startswith(('[MIXED_REACTION] ', '[NODAL_SOFT_CONTACT] ', '[NODAL_RELEASE] ', '[AVBD_GATE] '))), {})
                if sheet:
                    for dest, src in [('meanMs','averageStepMs'),('p95Ms','p95StepMs'),('maxComError','maxBallisticError')]:
                        if src in record: record[dest] = record[src]
                report['runs'].append(dict(model=model, workers=workers, repeat=repeat,
                                           exit=code, log=name, observation=record, otherCpuPercent=other))
                print(name, record or {'exit': code}, flush=True)
                (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    report['medians'] = {}
    for workers in (0, 4, 8):
        for model in ('rigid6', 'particle3'):
            records = [r['observation'] for r in report['runs'] if r['model'] == model and r['workers'] == workers]
            report['medians'][f'{model}-{workers}'] = {
                field: statistics.median(float(r[field]) for r in records)
                for field in ('meanMs', 'p95Ms', 'cpuMs', 'maxMomentumError', 'maxComError')
                if all(field in r for r in records)}
    report['after'] = identity()
    report['binariesUnchanged'] = report['before'] == report['after']
    report['allPhysicalGatesPassed'] = all(
        r['exit'] == 0 and r['observation'].get('status') == 'PASS' and
        (not sheet or r['model'] != 'rigid6' or r['observation'].get('model') == 'rigid_patch_joint_sheet')
        for r in report['runs'])
    medians = report['medians']
    complete = all('meanMs' in value and 'p95Ms' in value for value in medians.values())
    report['withinWholeStepCost'] = complete and all(medians[f'particle3-{w}'][field] <= medians[f'rigid6-{w}'][field]
                                        for w in (0, 4, 8) for field in ('meanMs', 'p95Ms'))
    report['relativeScaling'] = {
        w: (medians['particle3-0']['meanMs'] / medians[f'particle3-{w}']['meanMs']) /
           (medians['rigid6-0']['meanMs'] / medians[f'rigid6-{w}']['meanMs'])
        for w in (4, 8) if complete and all(medians[f'{model}-{i}']['meanMs']>0 for i in (0,w) for model in ('rigid6','particle3'))}
    report['scalingPassed'] = len(report['relativeScaling'])==2 and min(report['relativeScaling'].values()) >= 1
    report['mechanismPassed'] = all(report[key] for key in
                                    ('allPhysicalGatesPassed', 'binariesUnchanged', 'withinWholeStepCost', 'scalingPassed'))
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    return 0 if report['mechanismPassed'] else 1

if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='avbd_paired_') as directory:
        try:
            raise SystemExit(compare(Path(directory)))
        finally:
            retire_probe(Path(directory) / 'probe.exe')
