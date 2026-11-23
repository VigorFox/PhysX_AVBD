"""Validate high-degree Rigid6 rows against an analytic implicit solution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_avbd_local_budget_probe import ROOT, records

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,nargs='+',default=[0,4,8])
    parser.add_argument('--logs',type=Path,default=ROOT/'tools/benchmark-results/row-tiles')
    args=parser.parse_args();args.logs.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('PHYSX_AVBD_','PHYSX_NP_GEOMETRY_','PHYSX_SNIPPET_'))}
    env.update(PHYSX_AVBD_COMMON_STEP='budget',PHYSX_AVBD_LOCAL_ITERATIONS='4',PHYSX_AVBD_COMMON_STATS='summary',PHYSX_NP_GEOMETRY_CERTIFICATES='2')
    results=[]
    for workers in args.threads:
        for mode in (9,19,20,21):
            name=f'mode{mode}-w{workers}'
            run=subprocess.run([sys.executable,str(ROOT/'tools/run_avbd_common_step_probe.py'),'--child',str(mode),'--threads',str(workers)],env=env,capture_output=True,timeout=120)
            output=(run.stdout+run.stderr).decode('utf8',errors='replace');(args.logs/(name+'.log')).write_text(output,encoding='utf8')
            frames=records(output,'[AVBD_COMMON_FRAME]');ledger=records(output,'[AVBD_LOCAL_BUDGET]');tiles=records(output,'[AVBD_LOCAL_TILES]');quality=records(output,'[LOCAL_COLOR_CLIQUE]' if mode==21 else '[LOCAL_HIGH_DEGREE]')
            ok=(run.returncode==0 and len(quality)==1 and quality[0].get('status')=='PASS'
                and len(frames)==1 and frames[0].get('status')=='ACCEPTED' and frames[0].get('bdf')=='1'
                and len(ledger)==1 and ledger[0].get('iterations')=='4')
            if mode==21:
                ok=ok and ledger[0].get('primalBlocks')=='160' and ledger[0].get('colors')=='40'
            else:
                blocks='4' if mode==9 else '32'
                ok=(ok and ledger[0].get('primalBlocks')==blocks
                    and len(tiles)==1 and tiles[0].get('blocks')==blocks and tiles[0].get('rowLimit')=='256'
                    and tiles[0].get('reductionInputs')=='2' and int(tiles[0].get('maxIncidentRows',0))>=int(quality[0]['rows'])
                    and (not workers or (int(ledger[0].get('workers',0))==workers and int(ledger[0].get('tasks',0))>0)))
            # A short leaf may finish before another OS worker starts. Measure
            # effective overlap/scaling in the full native benchmark, not as a
            # scheduling-luck veto on this independent physical oracle.
            results.append(dict(name=name,passed=ok,quality=quality,tiles=tiles,
                                peakTasks=int(ledger[0].get('peakTasks',0)) if ledger else 0));print(name,'PASS' if ok else 'FAIL',quality,flush=True)
            (args.logs/'report.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    return int(any(not r['passed'] for r in results))
if __name__=='__main__':raise SystemExit(main())
