from pathlib import Path
p=Path('physx/source/lowleveldynamics/src/avbd/DyAvbdSolve.cpp');b=p.read_bytes()
old=b'#include "avbd/DyAvbdCommonResidual.h"';assert b.count(old)==1;b=b.replace(old,old+b'\r\n#include "avbd/DyAvbdCurvatureMetric.h"')
old=b'    if(!p.finish()){reason="numeric_program_invalid";return false;}'
extra='''
    if(preserveCurvature)for(PxU32 e=0;e<p.H.size();++e)if(p.H[e].row==p.H[e].column)
        if(!avbdRegularizeLocalCurvature(p.H[e].values,p.metricBlock(e).values,p.dimensions[p.H[e].row]))
        {reason="invalid_local_curvature";return false;}
'''.replace('\n','\r\n').encode()
assert b.count(old)==1;p.write_bytes(b.replace(old,old+extra))
