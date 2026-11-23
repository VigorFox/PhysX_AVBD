"""Verify the NP input stage; deliberately does NOT grant contact/production qualification."""
import json
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', help='Comma-separated case names to recheck, retaining other results')
    options = parser.parse_args()
    selected = set(options.only.split(',')) if options.only else None
    binary = Path(os.environ['AVBD_PROBE_BINARY_DIR']).resolve()
    output = ROOT / 'tools/benchmark-results/particle3-np-20261118'
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'verification.json'
    records = json.loads(result_path.read_text(encoding='utf-8'))['cases'] if selected and result_path.exists() else []
    executed = set()

    def run(name, command, expected=0, required=()):
        if selected and name not in selected:
            return True
        executed.add(name)
        with (output / (name + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        text = (output / (name + '.log')).read_text(encoding='utf-8', errors='replace')
        passed = result.returncode == expected and all(re.search(pattern, text) for pattern in required)
        record = dict(name=name, command=list(map(str, command)), exit=result.returncode,
                      expectedExit=expected, passed=bool(passed))
        previous = [row for row in records if row['name'] == name]
        if previous:
            record['previousAttempts'] = previous
        records[:] = [row for row in records if row['name'] != name]
        records.append(record)
        print(json.dumps(record), flush=True)
        (output / 'verification.json').write_text(json.dumps(dict(
            qualification='NP input/geometry mechanisms only; contact response UNCOMPILED',
            binary=str(binary), cases=records), indent=2), encoding='utf-8')
        return passed

    for workers in (0, 4, 8):
        run('mesh-f32-' + str(workers), [sys.executable, 'tools/run_np_deformable_mesh_probe.py', str(workers), '24'],
            required=[r'\[NP_MESH\] status=PASS'])
    for options, suffix in ((['--fast'], 'f32-fast'), (['--double'], 'f64'), (['--double', '--fast'], 'f64-fast')):
        run('mesh-' + suffix, [sys.executable, 'tools/run_np_deformable_mesh_probe.py', '4', '24'] + options,
            required=[r'\[NP_MESH\] status=PASS'])
    run('mesh-large', [sys.executable, 'tools/run_np_deformable_mesh_probe.py', '8', '128'],
        required=[r'\[NP_MESH\] status=PASS'])
    for kind in ('plane', 'sphere'):
        run('volume-' + kind + '-input', [sys.executable, 'tools/run_avbd_nodal_volume_step_probe.py', '4', kind + '-rejection'],
            expected=2, required=[r'views=1 vertices=8 invalid=0 missing=0 pending=0', r'response=UNCOMPILED', r'step not published'])
    for kind in ('plane', 'box'):
        run('surface-' + kind + '-input', [sys.executable, 'tools/run_np_deformable_surface_input_probe.py', kind],
            expected=2, required=[r'views=1 vertices=4 invalid=0 missing=0 pending=0', r'response=UNCOMPILED', r'step not published'])
    run('volume-flight', [sys.executable, 'tools/run_avbd_nodal_volume_step_probe.py', '4', 'flight'],
        required=[r'status=PASS'])
    for solver in ('avbd', 'pgs', 'tgs'):
        run('hello-' + solver, [sys.executable, 'tools/run_snippet_hello_world_headless.py', '--bin-dir', str(binary),
            '--solver', solver, '--cases', 'gated'])
    for case in ('membrane', 'bending', 'free-flight'):
        run('cloth-' + case, [str(binary / 'SnippetDeformableSurfaceAVBD_64.exe'), '--particle3', '--headless',
            '--case=' + case, '--frames=180', '--dispatcher-threads=4'], required=[r'PASS'])
    run('rigid6-cloth', [str(binary / 'SnippetDeformableSurfaceAVBD_64.exe'), '--rigid6', '--headless', '--case=static-box',
        '--frames=360', '--dispatcher-threads=4'], required=[r'PASS'])
    run('rigid6-volume', [str(binary / 'SnippetDeformableVolumeAVBD_64.exe'), '--rigid6', '--headless', '--case=visual-showcase',
        '--frames=360', '--dispatcher-threads=4'], required=[r'PASS'])
    if selected and selected != executed:
        parser.error('Unknown cases: ' + ', '.join(sorted(selected - executed)))
    return 0 if all(row['passed'] for row in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
