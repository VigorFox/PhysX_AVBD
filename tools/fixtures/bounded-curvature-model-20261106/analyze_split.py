from pathlib import Path
import json,numpy as np
out=Path(__file__).parent;p=json.loads((out/'split-candidate.json').read_text());t=json.loads((out/'split-candidate.json.trial.json').read_text());a=t['actual'];n=len(p['base'])
def hessian(entries):
 H=np.zeros((n,n))
 for r,c,*v in entries:H[6*r:6*r+6,6*c:6*c+6]+=np.array(v).reshape(6,6)
 return H
K=hessian(p['H']);M=hessian(p['metricH']);L=np.linalg.cholesky(M);A=np.linalg.solve(L,K);A=np.linalg.solve(L,A.T).T;ev=np.linalg.eigvalsh(.5*(A+A.T))
x=np.array(t['step']);f=np.array(t['force']);gx=np.array(p['base']);g1=np.array(a['base']);contact=np.zeros(n)
for r,c,*v in p['J']:gx[6*c:6*c+6]+=np.array(v).reshape(3,6).T@f[3*r:3*r+3];contact[6*c:6*c+6]-=np.array(v).reshape(3,6).T@f[3*r:3*r+3]
for r,c,*v in a['J']:g1[6*c:6*c+6]+=np.array(v).reshape(3,6).T@f[3*r:3*r+3];contact[6*c:6*c+6]+=np.array(v).reshape(3,6).T@f[3*r:3*r+3]
result=dict(spectrum=dict(minimum=float(ev.min()),negative=int((ev<0).sum()),maximum=float(ev.max())),requestedMetric=float(x@M@x),requestedCurvature=float(x@K@x),requestedSlope=-float(x@gx),stepMax=float(abs(x).max()),
 fullFrozenResidual=float(abs(K@x-gx).max()),actualCorrection=float(np.max(abs(np.linalg.solve(M,g1))/np.array(p['motionTolerance']))),contactGradientChange=float(abs(contact).max()),maxStoredDifference=float(np.max(abs(np.array(t['to'])-t['from']))))
(out/'split-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
diagonal=[]
for b,source in enumerate(p['sourceBlocks']):
 sl=slice(6*b,6*b+6);chol=np.linalg.cholesky(M[sl,sl]);v=np.linalg.solve(chol,K[sl,sl]);v=np.linalg.solve(chol,v.T).T
 diagonal.append((float(np.linalg.eigvalsh((v+v.T)*.5).min()),source))
values,vectors=np.linalg.eigh((A+A.T)*.5);mode=np.linalg.solve(L.T,vectors[:,0]);magnitudes=np.linalg.norm(mode.reshape(-1,6),axis=1)
detail=dict(smallestDiagonal=sorted(diagonal)[:8],negativeModeBodies=sorted(zip(magnitudes.tolist(),p['sourceBlocks']),reverse=True)[:8])
(out/'split-mode.json').write_text(json.dumps(detail,indent=2));print(json.dumps(detail,indent=2))
