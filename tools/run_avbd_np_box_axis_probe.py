"""Check a captured false NP separation failure against complete world vertices."""
import argparse
import json
from pathlib import Path
import struct
import tempfile
import numpy as np
from run_avbd_ogc_certificate_probe import ROOT, run_probe

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    args=parser.parse_args();data=json.loads(args.capture.read_text())
    with tempfile.TemporaryDirectory(prefix='np_box_axis_') as directory:
        path=Path(directory)/'geometry.bin'
        with path.open('wb') as stream:
            for side in data['sides']:
                points=np.array(side['points']);assert side['kind']==2 and points.shape==(8,3) and side['radius']==0
                edges=np.array([points[i]-points[0] for i in (1,2,4)]).T
                lengths=np.linalg.norm(edges,axis=0)
                # Recover the authored rigid box from its captured corners;
                # original world vertices remain the independent gap oracle.
                m=edges/lengths
                k=np.array([[m[0,0]-m[1,1]-m[2,2],m[0,1]+m[1,0],m[0,2]+m[2,0],m[2,1]-m[1,2]],
                    [m[0,1]+m[1,0],m[1,1]-m[0,0]-m[2,2],m[1,2]+m[2,1],m[0,2]-m[2,0]],
                    [m[0,2]+m[2,0],m[1,2]+m[2,1],m[2,2]-m[0,0]-m[1,1],m[1,0]-m[0,1]],
                    [m[2,1]-m[1,2],m[0,2]-m[2,0],m[1,0]-m[0,1],np.trace(m)]])
                quat=np.linalg.eigh(k)[1][:,-1]
                stream.write(struct.pack('<10f',*points.mean(axis=0),*quat,*(lengths*.5)))
                stream.write(struct.pack('<24d',*points.flatten()))
            stream.write(struct.pack('<3d',*data['axis']))
        for mode in ('precise','fast'):
            print('NP box axis /fp:'+mode,flush=True)
            code=run_probe(ROOT/'tools/avbd_np_box_axis_probe.cpp',args=(str(path),),
                include_dirs=(ROOT/'physx/source/lowlevel/software/include',),compiler_options=('/fp:'+mode,))
            if code:return code
    return 0

if __name__=='__main__':raise SystemExit(main())
