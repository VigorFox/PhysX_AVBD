"""Attribute preparation/cooking cost separately from unprofiled qualification."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from collections import defaultdict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('PHYSX_AVBD_', 'PHYSX_NP_', 'PHYSX_SNIPPET_'))}
    env.update(PHYSX_AVBD_LOCAL_PROFILE='1', PHYSX_AVBD_COMMON_STATS='summary')
    report = []
    for scene, case in [('Surface', 'static-box'), ('Volume', 'visual-showcase')]:
        for workers in (0, 4, 8):
            for label in ('baseline', 'candidate'):
                binary = getattr(args, label).resolve()
                command = [str(binary / f'SnippetDeformable{scene}AVBD_64.exe'),
                    '--headless', '--case=' + case, '--frames=360',
                    '--dispatcher-threads=' + str(workers)]
                proc = subprocess.run(command, cwd=binary, env=env, capture_output=True, timeout=300)
                text = (proc.stdout + proc.stderr).decode('utf-8', errors='replace')
                (args.output / f'{scene}-{label}-{workers}.log').write_text(text, encoding='utf-8')
                data = defaultdict(list)
                for line in text.splitlines():
                    m = re.match(r'\[(AVBD_[A-Z_]+)\] (.*)', line)
                    if m:
                        data[m[1]].append(dict(re.findall(r'(\w+)=([^\s]+)', m[2])))
                cook = data['AVBD_MATERIAL_COOK']
                prepare = data['AVBD_LOCAL_PREPARE']
                result = dict(scene=scene, label=label, workers=workers, exit=proc.returncode,
                    cookFrames=len(cook), rebuilds=sum(int(x['rebuilt']) for x in cook),
                    maxCookedBytes=max((int(x['cookedBytes']) for x in cook), default=0),
                    prepareMeanMs={i: sum(float(x['ms']) for x in prepare if x['part'] == str(i)) / 360 for i in range(8)},
                    prepareColdMs={i: next((float(x['ms']) for x in prepare if x['part'] == str(i)), 0) for i in range(8)},
                    gate=data['AVBD_GATE'], raw=data)
                report.append(result)
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(scene, label, workers, proc.returncode, result['rebuilds'], result['prepareMeanMs'], flush=True)
    return int(any(r['exit'] or not r['gate'] or r['gate'][0]['status'] != 'PASS' for r in report))


if __name__ == '__main__':
    raise SystemExit(main())
