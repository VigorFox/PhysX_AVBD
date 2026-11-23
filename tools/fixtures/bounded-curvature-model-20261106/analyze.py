from pathlib import Path
import json,numpy as np
out=Path(__file__).parent;p=json.loads((out/'original-candidate.json').read_text());t=json.loads((out/'original-candidate.json.trial.json').read_text());curv=json.loads((out/'original-candidate.json.curvature.json').read_text())
n=len(p['base']);H=np.zeros((n,n));raw=np.zeros_like(H);projected=np.zeros_like(H);index={b:i for i,b in enumerate(p['sourceBlocks'])}
for a,b,*v in p['H']:H[6*a:6*a+6,6*b:6*b+6]+=np.array(v).reshape(6,6)
before=np.array(t['from']).reshape(-1,7);after=np.array(t['to']).reshape(-1,7);d=np.zeros((len(before),6));d[:,:3]=after[:,:3]-before[:,:3]
for i,(a,b) in enumerate(zip(before[:,3:],after[:,3:])):
 a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);v=a[3]*b[:3]-b[3]*a[:3]-np.cross(b[:3],a[:3]);w=a@b
 if w<0:v=-v;w=-w
 s=np.linalg.norm(v)
 if s:d[i,3:]=v*(2*np.arctan2(s,w)/s)
d=d.ravel();rows=[]
for f in curv['frames']:
 assert f['valid'];S=np.zeros((9,12));S[:3,:3]=np.eye(3);S[:3,6:9]=-np.eye(3);S[3:6,3:6]=np.eye(3);S[6:,9:]=np.eye(3)
 r=np.array(f['raw']).reshape(9,9);q=np.array(f['projected']).reshape(9,9);movement=np.zeros(12)
 for a,big in enumerate((raw,projected)):
  block=S.T@(r if a==0 else q)@S
  for s,body in enumerate((f['a'],f['b'])):
   if body not in index:continue
   i=6*index[body];movement[6*s:6*s+6]=d[i:i+6]
   for z,other in enumerate((f['a'],f['b'])):
    if other in index:
     j=6*index[other];big[i:i+6,j:j+6]+=block[6*s:6*s+6,6*z:6*z+6]
 y=S@movement;rows.append(dict(row=f['row'],a=f['a'],b=f['b'],defect=float(y@(q-r)@y),raw=float(y@r@y),projected=float(y@q@y)))
model=H-projected+raw;scale=1/np.sqrt(np.diag(H));scaled=lambda a: a*scale[:,None]*scale[None,:]
L=np.linalg.cholesky(scaled(H));A=np.linalg.solve(L,scaled(model));A=np.linalg.solve(L,A.T).T
ev=np.linalg.eigvalsh((A+A.T)*.5);Delta=np.array(t['actual']['base'])-p['base'];tol=np.array(p['motionTolerance'])
residuals=np.column_stack([Delta+H@d,Delta+model@d,(projected-raw)@d]);corrections=np.linalg.solve(H,residuals)
summary=dict(blocks=len(index),frames=len(rows),modelDirection=float(d@H@d),trueFrameDirection=float(d@raw@d),projectedFrameDirection=float(d@projected@d),otherDirection=float(d@(H-projected)@d),rawModelDirection=float(d@model@d),actualDirection=float(-d@Delta),
 generalizedSpectrum=dict(minimum=float(ev.min()),maximum=float(ev.max()),negative=int(np.sum(ev<0)),smallest=ev[:12].tolist()),
 correctionRatios=dict(zip(['oldModel','rawFrameModel','projectionDefect'],np.max(np.abs(corrections)/tol[:,None],axis=0).tolist())),
 worstFrames=sorted(rows,key=lambda r:r['defect'],reverse=True)[:12])
np.savez_compressed(out/'operators.npz',H=H,exact=model,frame=raw,projected=projected,delta=d)
(out/'analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
