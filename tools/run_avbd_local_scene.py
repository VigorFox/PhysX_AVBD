"""Run actual SDK scenes with explicit local common-step execution and raw logs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import json

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='visual-showcase')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--mode', choices=('bounded','patch','local','strict','old'), default='local')
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--capture-failure',type=Path)
    parser.add_argument('--capture-candidate',type=Path)
    parser.add_argument('--capture-np',type=Path)
    parser.add_argument('--stop-on-rejection',action='store_true',help='stop diagnostic run after first explicitly rejected frame; never a completed gate')
    args=parser.parse_args()
    outputs=[args.log,args.log.with_suffix('.json')]+[p for p in (args.capture_failure,args.capture_candidate,args.capture_np) if p]
    if len({p.resolve() for p in outputs})!=len(outputs):parser.error('log, summary, and capture paths must be distinct')
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('PHYSX_AVBD_COMMON_'):del env[key]
    env['PHYSX_NP_GEOMETRY_CERTIFICATES']='2'
    if args.mode!='old':env['PHYSX_AVBD_COMMON_STEP']='1' if args.mode=='strict' else args.mode
    if args.stats or args.verbose:env['PHYSX_AVBD_COMMON_STATS']='1' if args.verbose else 'summary'
    if args.capture_failure:env['PHYSX_AVBD_COMMON_FAILED_NCP']=str(args.capture_failure.resolve())
    if args.capture_candidate:env['PHYSX_AVBD_COMMON_FAILED_CANDIDATE']=str(args.capture_candidate.resolve())
    if args.capture_np:env['PHYSX_AVBD_COMMON_FAILED_NP']=str(args.capture_np.resolve())
    cmd=[str(ROOT/'physx/bin/win.x86_64.vc143.md/checked/SnippetJointSoftBody_64.exe'),
         '--headless','--case='+args.case,'--solver=avbd','--frames='+str(args.frames),
         '--grid=5','--dispatcher-threads='+str(args.threads),'--volume-material']
    args.log.parent.mkdir(parents=True,exist_ok=True);begin=time.perf_counter();timed_out=False;stopped=False
    with args.log.open('wb') as stream:
        process=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
        with args.log.open('rb') as monitor:
            tail=b''
            while process.poll() is None:
                if time.perf_counter()-begin>args.timeout:timed_out=True;break
                if args.stop_on_rejection:
                    tail=(tail+monitor.read())[-8192:]
                    if b'[AVBD_COMMON_FRAME] status=REJECTED' in tail:stopped=True;break
                time.sleep(.1)
        if timed_out or stopped:process.terminate()
        code=process.wait()
    lines=args.log.read_text(encoding='utf8',errors='replace').splitlines()
    result=dict(command=cmd,mode=args.mode,elapsed=time.perf_counter()-begin,exit=code,timedOut=timed_out,stoppedOnRejection=stopped,
        accepted=sum(line.startswith('[AVBD_COMMON_FRAME] status=ACCEPTED') for line in lines),
        rejected=sum(line.startswith('[AVBD_COMMON_FRAME] status=REJECTED') for line in lines),
        nonstationary=sum(line.startswith('[AVBD_COMMON_FRAME] status=ACCEPTED') and 'stationary=0' in line for line in lines),
        inexact=sum(line.startswith('[AVBD_LOCAL_RESPONSE]') and 'status=INEXACT' in line for line in lines),
        certifiedInexact=sum(line.startswith('[AVBD_COMMON_CERTIFIED_INEXACT]') for line in lines),
        certifiedDirections=sum(line.startswith('[AVBD_COMMON_DIRECTION_CERTIFICATE]') for line in lines),
        strictBudgetExhaustions=sum(line.startswith('[AVBD_BOUNDED_RESPONSE]') and 'status=budget_exhausted' in line for line in lines),
        gate=[line for line in lines if line.startswith('[AVBD_GATE]')])
    args.log.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2));return code if code is not None else 2
if __name__=='__main__':sys.exit(main())
