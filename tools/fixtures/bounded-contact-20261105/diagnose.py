import gzip,json,numpy as np
from pathlib import Path
out=Path('tools/benchmark-results/bounded-contact-20261105'); results=[]
for source,response in [(Path('tools/fixtures/bounded-host-20261105/reaction-failure.json.gz'),out/'cost-before-t1.json'),(out/'reaction-eq-failure.json',out/'reaction-eq-replay.json')]:
 data=source.read_bytes();p=json.loads(gzip.decompress(data) if source.suffix=='.gz' else data);r=json.loads(response.read_text())
 n=len(p['base']);m=len(p['mu']);dims=np.array(p['dimensions']);starts=np.r_[0,np.cumsum(dims)];h=np.zeros((n,n));reaction=np.zeros(n);f=np.array(r['force']).reshape(m,3)
 for a,b,*v in p['H']:h[starts[a]:starts[a+1],starts[b]:starts[b+1]]+=np.array(v).reshape(6,6)[:dims[a],:dims[b]]
 for a,b,*v in p['J']:reaction[starts[b]:starts[b+1]]+=np.array(v).reshape(3,6)[:,:dims[b]].T@f[a]
 x=np.array(r['step']);base=np.array(p['base']);dx=np.linalg.solve(h,h@x-base-reaction)
 result=dict(source=str(source),solverConverged=r['converged'],scope='independent dense diagnostic only; no publication or production fallback',fullHCorrection=float(max(abs(dx))),states=[])
 for name,y in [('original',x),('fixedForceEquilibrium',x-dx)]:
  jx=np.zeros((m,3))
  for a,b,*v in p['J']:jx[a]+=np.array(v).reshape(3,6)[:,:dims[b]]@y[starts[b]:starts[b+1]]
  u=np.array(p['offset']).reshape(m,3)+jx+np.array(p['compliance']).reshape(m,3)*f
  v=f-u;proj=np.clip(v,np.array(p['lower'],float).reshape(m,3),np.array(p['upper'],float).reshape(m,3))
  for a in np.flatnonzero(p['coulomb']):
   length=np.linalg.norm(v[a,1:]);radius=p['mu'][a]*max(0,f[a,0]);proj[a,1:]=v[a,1:]*(radius/length if length>radius else 1)
  result['states'].append(dict(name=name,station=float(max(abs(h@y-base-reaction))),unitRoot=float(np.max(abs(f-proj)))))
 results.append(result)
(out/'equilibrium-analysis.json').write_text(json.dumps(results,indent=2),encoding='utf8');print(json.dumps(results,indent=2))
