"""Replay the f9abbe0 checkpoint against an explicit, freshly built SDK.

The original validators remain authoritative. This driver only selects products,
isolates logs/environment, and records source/product provenance. It never
compares floating-point trajectories between worker counts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def fingerprints(binary_dir):
    paths = subprocess.check_output(
        ['git', 'ls-files', '-z', 'physx'], cwd=ROOT).decode().split('\0')
    files = [ROOT / p for p in paths if p and (ROOT / p).is_file()]
    files += sorted(p for p in binary_dir.iterdir()
                    if p.suffix.lower() in ('.dll', '.exe', '.lib'))
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--retry-failed', type=Path,
                        help='Retry failed jobs after a dependency fix; prior product hashes must match')
    args = parser.parse_args()
    binary_dir = args.bin_dir.resolve()
    args.logs.mkdir(parents=True, exist_ok=False)
    assert (binary_dir / 'avbd_smoke.exe').is_file(), 'Build smoke in --bin-dir first'
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PHYSX_AVBD_', 'PHYSX_NP_GEOMETRY_', 'PHYSX_SNIPPET_'))}
    env['AVBD_PROBE_BINARY_DIR'] = str(binary_dir)
    env['PATH'] = str(binary_dir) + os.pathsep + env.get('PATH', '')
    env['PYTHONIOENCODING'] = 'utf-8'
    source = ROOT / 'tools/fixtures/native-checkpoint-20261116/final-validation.json'
    jobs = json.loads(source.read_text(encoding='utf-8'))
    jobs += [dict(name='extra-hello-' + solver,
                  command=[sys.executable, 'tools/run_snippet_hello_world_headless.py',
                           '--cases', 'gated', '--solver', solver]) for solver in ('pgs', 'tgs')]
    jobs += [dict(name='extra-bounded-reference', command=[sys.executable,
                'tools/run_avbd_common_step_probe.py', '--mode', 'bounded',
                '--logs', str(args.logs.resolve() / 'bounded')])]
    before = fingerprints(binary_dir)
    previous = None
    if args.retry_failed:
        previous = json.loads(args.retry_failed.read_text(encoding='utf-8'))
        assert previous['identityStable'], 'Previous execution had changing inputs'
        assert all(before.get(p) == h for p, h in previous['fingerprints'].items()), \
            'Source/products changed; a selective retry would mix different checkpoints'
        failed = {r['name'] for r in previous['results'] if not r['passed']}
        jobs = [j for j in jobs if j['name'] in failed]
    report = dict(base='f9abbe0cd869136d20cba6462551bbb141a75cd0',
                  head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                  binaryDir=str(binary_dir), results=[])
    if previous:
        report['retryOf'] = str(args.retry_failed.resolve())
        report['previousPassedUnchanged'] = [r['name'] for r in previous['results'] if r['passed']]
    for job in jobs:
        command = list(job['command'])
        if command[0].lower().endswith('.exe') and 'python' not in Path(command[0]).name.lower():
            command[0] = str(binary_dir / Path(command[0]).name)
        else:
            command[0] = sys.executable
            if Path(command[1]).name.startswith('run_snippet_'):
                command += ['--bin-dir', str(binary_dir)]
            if '--logs' in command:
                command[command.index('--logs') + 1] = str(args.logs.resolve() / job['name'])
        start = time.perf_counter()
        try:
            run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=600)
            code, output = run.returncode, run.stdout + run.stderr
        except subprocess.TimeoutExpired as error:
            code, output = 124, (error.stdout or b'') + (error.stderr or b'')
        text = output.decode('utf-8', errors='replace')
        (args.logs / (job['name'] + '.log')).write_text(text, encoding='utf-8')
        gates = [line for line in text.splitlines() if line.startswith('[AVBD_GATE]')]
        direct_scene = Path(command[0]).name.startswith('Snippet')
        passed = code == 0 and (not direct_scene or
                  bool(gates) and all('status=PASS' in line for line in gates))
        report['results'].append(dict(name=job['name'], command=command, exit=code,
            passed=passed, seconds=time.perf_counter() - start, gates=gates))
        (args.logs / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(job['name'], 'PASS' if passed else 'FAIL', code, flush=True)
    after = fingerprints(binary_dir)
    report['identityStable'] = before == after
    report['fingerprints'] = before
    report['passed'] = report['identityStable'] and all(r['passed'] for r in report['results'])
    (args.logs / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('RESULT', report['passed'], sum(r['passed'] for r in report['results']), '/', len(jobs))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
