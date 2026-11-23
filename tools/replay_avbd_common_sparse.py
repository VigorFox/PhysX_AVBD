"""Replay SDK input through the experimental sparse adapter and shared NCP.

This is offline numerical evidence, never a physical state publication.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
lab = ROOT / 'experiments/avbd_contact_lab'
if not (lab / 'split.py').is_file():
    raise SystemExit(f"Offline replay requires the local AVBD contact lab at {lab}")
sys.path.insert(0,str(lab))
import split
from validate_avbd_common_capture import read_capture, validate

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('capture',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=('mixed','cold'),default='mixed');args=parser.parse_args()
    p=read_capture(args.capture);assert p['complete'] and all(d==6 for d in p['dimensions'])
    m=len(p['mu']);blocks=[{} for _ in p['dimensions']];jac=[{} for _ in range(m)]
    for row,col,*values in p['H']:blocks[row][col]=np.array(values).reshape(6,6)
    for row,col,*values in p['J']:jac[row][col]=np.array(values).reshape(3,6)
    stencil=[(np.array(list(row),dtype=int),np.stack(list(row.values()),axis=1)) for row in jac]
    program=split.Program(blocks,stencil,np.array(p['offset']).reshape(m,3),np.array(p['base']))
    print('compiled',len(p['base']),m,'method',args.method,flush=True)
    if args.method=='cold':
        lo=np.array(p['lower'],dtype=float).reshape(m,3);hi=np.array(p['upper'],dtype=float).reshape(m,3)
        assert np.all(lo[:,0]==0) and np.all(hi[:,0]>1e30) and not np.any(p['compliance'])
        x,f,stats=program.solve_coulomb(p['mu'])
    else:
        x,f,stats=program.solve_mixed_coulomb(p['mu'],p['coulomb'],
            np.array(p['lower'],dtype=float).reshape(m,3),np.array(p['upper'],dtype=float).reshape(m,3),
            np.array(p['compliance']).reshape(m,3))
    p['replay']=dict(solved=stats['converged'],step=x.tolist(),force=f.ravel().tolist(),statistics=stats,
                     source='experimental sparse adapter; shared production NCP',provenance=split.provenance())
    args.output.write_text(json.dumps(p),encoding='utf-8')
    result=validate(args.output);result['execution']=stats
    args.output.with_suffix('.validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
