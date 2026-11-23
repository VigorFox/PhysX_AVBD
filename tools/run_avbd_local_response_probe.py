"""Replay captured H/J through the shared colored local numerical solver."""
import argparse
import gzip
import json
import struct
import tempfile
from pathlib import Path
from run_avbd_ogc_certificate_probe import ROOT, run_probe

def pack(path, p, warm=False):
    with path.open('wb') as f:
        def write(kind, values):
            f.write(struct.pack('<' + kind * len(values), *values))
        write('I', [len(p['dimensions']), len(p['mu']), len(p['H']), len(p['J'])])
        write('I', p['dimensions'])
        write('d', p['base']); write('d', p['mu']); write('i', p['coulomb'])
        for key in ('offset', 'lower', 'upper', 'compliance'):
            write('d', [float(v) for v in p[key]])
        for key in ('H', 'J'):
            for row in p[key]:
                write('I', row[:2]); write('d', row[2:])
        write('d', p['initialForce'] if warm else [0.] * len(p['offset']))
        write('d', p.get('motionTolerance') or [1.e-5] * len(p['base']))
        metric = p.get('metricH', [])
        write('I', [len(metric)])
        for row in metric:
            write('I', row[:2]); write('d', row[2:])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--engineering',action='store_true')
    parser.add_argument('--strict',action='store_true')
    parser.add_argument('--grouped',action='store_true')
    parser.add_argument('--bounded', action='store_true')
    parser.add_argument('--warm-start',action='store_true')
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--threads', type=int, choices=(1, 4), default=1, help='bounded executor test pool only')
    args=parser.parse_args()
    if (args.strict and args.grouped) or (args.engineering and args.strict):
        parser.error('choose one executor; engineering applies only to local/grouped')
    if args.bounded and (args.strict or args.grouped or args.engineering):
        parser.error('bounded has its own fixed budget and strict residual check')
    if args.threads != 1 and not args.bounded: parser.error('--threads applies to --bounded only')
    data=args.capture.read_bytes()
    if args.capture.suffix=='.gz': data=gzip.decompress(data)
    p=json.loads(data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='avbd_local_') as directory:
        path=Path(directory)/'program.bin';pack(path,p,args.warm_start)
        mode='strict' if args.strict else 'grouped-engineering' if args.grouped and args.engineering else 'grouped' if args.grouped else 'engineering' if args.engineering else 'local'
        if args.bounded: mode='bounded'
        code=run_probe(ROOT/'tools/avbd_local_response_probe.cpp', args=(str(path),str(args.report.resolve()),mode,str(args.repeat),str(args.threads)),
            include_dirs=(ROOT/'tools',), compiler_options=('/fp:precise',),
            extra_sources=(ROOT/'physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp',))
    if args.report.is_file():
        from validate_avbd_common_capture import validate
        result=json.loads(args.report.read_text())
        p['replay']=dict(solved=result['converged'],step=result['step'],force=result['force'],source='strict full response' if args.strict else 'grouped augmented primal/dual' if args.grouped else 'local block primal/dual')
        if args.bounded: p['replay']['source']='bounded colored AL; no global inner solver'
        p['tableIterations']=result['iterations'];p['materialProducts']=result['materialProducts']
        full=args.report.with_suffix('.program.json');full.write_text(json.dumps(p),encoding='utf8')
        validation=validate(full);validation['solverConverged']=result['converged'];validation['engineering']=args.engineering
        validation['initialForceMode']='captured' if args.warm_start else 'zero'
        # Independent full-H check for bounded diagnostic packets. This is
        # deliberately not a geometry or complete force-law qualification.
        if args.engineering and len(p['base'])<=192:
            import numpy as np
            starts=np.r_[0,np.cumsum(p['dimensions'])];n=int(starts[-1]);h=np.zeros((n,n));reaction=np.zeros(n)
            for a,b,*v in p.get('metricH') or p['H']:
                h[starts[a]:starts[a+1],starts[b]:starts[b+1]]+=np.array(v).reshape(6,6)[:p['dimensions'][a],:p['dimensions'][b]]
            for a,b,*v in p['J']:
                reaction[starts[b]:starts[b+1]]+=np.array(v).reshape(3,6)[:,:p['dimensions'][b]].T@np.array(result['force'][3*a:3*a+3])
            correction=np.linalg.solve(h,h@result['step']-p['base']-reaction)
            tolerance=.1*np.array(p.get('motionTolerance') or [1.e-5]*n)
            ratio=float(np.max(np.abs(correction)/tolerance,initial=0))
            validation['engineeringStationarity']=dict(maximumRatio=ratio,passed=ratio<=1,scope='full-H correction only')
            if ratio>1:code=1
        args.report.with_suffix('.validation.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
        print(json.dumps(validation,indent=2))
        if not args.engineering and not validation['passed']:code=1
    return code

if __name__=='__main__': raise SystemExit(main())
