"""Audit f9abbe0's actual four-iteration dispatch and reaction coverage.

This checks work/coverage of the baseline numerical program, not equality of
colors, trajectories or group membership between different executions.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def records(text, tag):
    return [dict(re.findall(r'(\w+)=([^\s]+)', line))
            for line in text.splitlines() if line.startswith(tag)]


def coverage(text):
    result, parent = [], {}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def finish():
        if parent:
            result.append(dict(blocks=len(parent), components=len({root(x) for x in parent})))

    for line in text.splitlines():
        if not line.startswith('[AVBD_LOCAL_PARTITION]'):
            continue
        match = re.search(r'iteration=(\d+) closed=(\S*) groups=(.*)', line)
        if int(match[1]) == 0:
            finish()
            parent = {int(x): int(x) for x in match[2].split(',') if x}
        for group in match[3].split(';'):
            members = [int(x) for x in group.split(',') if x and int(x) in parent]
            for x in members[1:]:
                parent[root(x)] = root(members[0])
    finish()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PHYSX_AVBD_', 'PHYSX_NP_', 'PHYSX_SNIPPET_'))}
    env.update(PHYSX_AVBD_GROUP_TRACE='1', PHYSX_AVBD_COMMON_STATS='summary',
               PHYSX_AVBD_BOUNDED_PROFILE='1')
    bounds = dict(queuePeak=16, primalMaxBlocks=32, primalMaxRows=256,
                  dualMaxRows=256, queryMaxRows=32)
    report = []
    for scene, case in [('Surface', 'static-box'), ('Volume', 'visual-showcase'),
                        ('Surface', 'reaction'), ('Volume', 'reaction')]:
        command = [str(args.bin_dir.resolve() / f'SnippetDeformable{scene}AVBD_64.exe'),
                   '--headless', '--case=' + case, '--frames=360', '--dispatcher-threads=4']
        run = subprocess.run(command, env=env, capture_output=True, timeout=600)
        text = (run.stdout + run.stderr).decode('utf-8', errors='replace')
        name = scene + '-' + case
        (args.logs / (name + '.log')).write_text(text, encoding='utf-8')
        gates = records(text, '[AVBD_GATE]')
        ledgers = records(text, '[AVBD_LOCAL_BUDGET]')
        dispatch = records(text, '[AVBD_LOCAL_DISPATCH]')
        groups = records(text, '[AVBD_LOCAL_GROUPS]')
        passed = (run.returncode == 0 and len(gates) == 1 and gates[0]['status'] == 'PASS'
                  and gates[0]['completedFrames'] == '360' and len(ledgers) == len(dispatch) == 360
                  and all(0 < int(r['iterations']) <= 4 and int(r['budget']) == 4 for r in ledgers)
                  and all(all(int(r[k]) <= v for k, v in bounds.items()) for r in dispatch)
                  and len(groups) == 360 and all(int(r['colorBuilds']) <= 4 for r in groups))
        span = coverage(text) if case == 'reaction' else []
        if case == 'reaction':
            blocks = int(gates[0]['cells' if scene == 'Surface' else 'nodes']) + 1 if gates else 0
            passed = passed and len(span) == 360 and all(r['components'] == 1 for r in span)
            passed = passed and all(int(r['primalBlocks']) == int(r['iterations']) * blocks for r in ledgers)
        report.append(dict(name=name, passed=passed, command=command, exit=run.returncode,
                           frames=len(ledgers), bounds=bounds,
                           observedMax={k: max((int(r[k]) for r in dispatch), default=0) for k in bounds},
                           coverage=span, gate=gates))
        (args.logs / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(name, 'PASS' if passed else 'FAIL', flush=True)
    return int(any(not r['passed'] for r in report))


if __name__ == '__main__':
    raise SystemExit(main())
