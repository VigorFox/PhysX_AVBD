"""Independent sparse products and original force-law check on SDK Table output.

Checks the captured linearization, not nonlinear geometry or physical publication.
No H inverse or dense island matrix is constructed here. Unit step in the root
map is arbitrary; its zeros are the original bounded-normal/Coulomb equations.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def read_capture(path):
    """Read plain diagnostic output or a lossless checked-in fixture."""
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def validate(path):
    p=read_capture(path)
    replay=p.get('replay');solved=replay['solved'] if replay else p.get('tableSolved',False)
    source=replay['source'] if replay else 'production Table diagnostic'
    if not p['complete'] or not solved:
        return dict(file=str(path),passed=False,complete=p['complete'],solved=solved,executionSource=source,reason=p['reason'],
                    materialProducts=p.get('materialProducts'),tableIterations=p.get('tableIterations'))
    dims=np.array(p['dimensions']);starts=np.r_[0,np.cumsum(dims)];n=starts[-1];m=len(p['mu'])
    x=np.array(replay['step'] if replay else p['tableStep']);f=np.array(replay['force'] if replay else p['tableForce']).reshape(m,3);base=np.array(p['base'])
    hx=np.zeros(n);jtf=np.zeros(n);jx=np.zeros((m,3));h={}
    for row,col,*values in p['H']:
        B=np.array(values).reshape(6,6)[:dims[row],:dims[col]];h[row,col]=B
        hx[starts[row]:starts[row+1]]+=B@x[starts[col]:starts[col+1]]
    symmetry=max((float(np.max(abs(B-h.get((c,r),np.zeros(B.T.shape)).T))) for (r,c),B in h.items()),default=0.)
    for row,col,*values in p['J']:
        B=np.array(values).reshape(3,6)[:,:dims[col]]
        jx[row]+=B@x[starts[col]:starts[col+1]];jtf[starts[col]:starts[col+1]]+=B.T@f[row]
    station=float(np.max(abs(hx-base-jtf),initial=0))
    station_scale=max(1.,np.max(abs(base),initial=0),np.max(abs(jtf),initial=0))
    lo=np.array(p['lower'],dtype=float).reshape(m,3);hi=np.array(p['upper'],dtype=float).reshape(m,3)
    u=np.array(p['offset']).reshape(m,3)+jx+np.array(p['compliance']).reshape(m,3)*f
    v=f-u;projected=np.clip(v,lo,hi)
    for c in np.flatnonzero(p['coulomb']):
        radius=p['mu'][c]*max(0.,f[c,0]);length=np.linalg.norm(v[c,1:])
        projected[c,1:]=v[c,1:]*(radius/length if length>radius else 1.)
    root=float(np.max(abs(f-projected),initial=0));scale=max(1.,np.max(abs(f),initial=0),np.max(abs(u),initial=0))
    passed=bool(symmetry<1e-7 and station/station_scale<1e-10 and root/scale<1e-10)
    return dict(file=str(path),passed=passed,complete=True,solved=True,executionSource=source,dofs=int(n),groups=m,
                activeNormalForces=int(np.count_nonzero(f[:,0]>1e-10)),maximumForce=float(np.max(abs(f),initial=0)),
                symmetry=symmetry,stationarity=station,relativeStationarity=float(station/station_scale),
                unitStepRoot=root,relativeUnitStepRoot=float(root/scale),minimumNormalResidual=float(np.min(u[:,0],initial=0)),
                tableIterations=p['tableIterations'],materialProducts=p['materialProducts'],publication='none')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('capture',type=Path);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();result=validate(args.capture);args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
