"""Check joint/rigid gravity trajectories with an unrelated impact contact."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_avbd_local_budget_probe import ROOT, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, nargs='+', default=[0, 4, 8])
    parser.add_argument('--logs', type=Path, default=ROOT / 'tools/benchmark-results/gravity')
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PHYSX_AVBD_', 'PHYSX_NP_GEOMETRY_', 'PHYSX_SNIPPET_'))}
    env.update(PHYSX_AVBD_COMMON_STEP='budget', PHYSX_AVBD_LOCAL_ITERATIONS='4',
               PHYSX_AVBD_COMMON_STATS='summary', PHYSX_NP_GEOMETRY_CERTIFICATES='2')
    results = []
    for threads in args.threads:
        for mode in (17, 18):
            name = f'mode{mode}-w{threads}'
            run = subprocess.run([sys.executable, str(ROOT / 'tools/run_avbd_common_step_probe.py'),
                                  '--child', str(mode), '--threads', str(threads)],
                                 env=env, capture_output=True, timeout=90)
            output = (run.stdout + run.stderr).decode('utf8', errors='replace')
            (args.logs / (name + '.log')).write_text(output, encoding='utf8')
            frames = records(output, '[AVBD_COMMON_FRAME]')
            ledger = records(output, '[AVBD_LOCAL_BUDGET]')
            quality = records(output, '[GRAVITY_FLIGHT]')
            passed = (run.returncode == 0 and len(quality) == 1 and quality[0].get('status') == 'PASS'
                      and len(frames) == 120 and len(ledger) == 120
                      and all(f.get('status') == 'ACCEPTED' and f.get('bdf') == '1' for f in frames)
                      and all(f.get('iterations') == '4' for f in ledger))
            results.append(dict(name=name, passed=passed, exit=run.returncode, quality=quality))
            (args.logs / 'report.json').write_text(json.dumps(results, indent=2), encoding='utf8')
            print(name, 'PASS' if passed else 'FAIL', quality, flush=True)
    return int(any(not r['passed'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
