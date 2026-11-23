from pathlib import Path
import gzip,json
import numpy as np

root=Path.cwd();out=Path(__file__).parent;fixture=root/'tools/fixtures/bounded-root-20261106'
p=json.loads(gzip.decompress((fixture/'terminal-candidate.json.gz').read_bytes()))
t=json.loads(gzip.decompress((fixture/'terminal-candidate.json.trial.json.gz').read_bytes()))
a=t['actual'];n=len(p['base']);starts=np.r_[0,np.cumsum(p['dimensions'])]
H=np.zeros((n,n));J=np.zeros((len(p['offset']),n));A=np.zeros_like(J)
for r,c,*v in p['H']:H[starts[r]:starts[r+1],starts[c]:starts[c+1]]+=np.array(v).reshape(6,6)[:p['dimensions'][r],:p['dimensions'][c]]
for program,matrix in [(p,J),(a,A)]:
 for r,c,*v in program['J']:matrix[3*r:3*r+3,starts[c]:starts[c+1]]+=np.array(v).reshape(3,6)[:,:p['dimensions'][c]]
x=np.array(t['step']);f=np.array(t['force']);tol=np.array(p['motionTolerance']);hx=H@x
parts=dict(energy=np.array(a['base'])-p['base']+hx,contact=(A-J).T@f,
           frozen=np.array(p['base'])+J.T@f-hx,rounding=A.T@(np.array(t['storedForce'])-f))
result={}
for name,c in zip(parts,np.linalg.solve(H,np.array(list(parts.values())).T).T):
 i=int(np.argmax(np.abs(c)/tol));result[name]=dict(ratio=float(np.max(np.abs(c)/tol)),body=p['sourceBlocks'][i//6],coordinate=i%6,
                                                atWorst=float(c[6*p['sourceBlocks'].index(7)+5]/tol[6*p['sourceBlocks'].index(7)+5]))
result['rayleighEnergy']=float(-x@(np.array(a['base'])-p['base'])/(x@hx))
result['rayleighContact']=float(x@((A-J).T@f)/(x@hx))
result['stepMetric']=float(x@hx)
before=np.array(t['from']).reshape(-1,7);after=np.array(t['to']).reshape(-1,7)
d=np.zeros((len(before),6));d[:,:3]=after[:,:3]-before[:,:3]
for i in range(len(before)):
 q=before[i,3:]/np.linalg.norm(before[i,3:]);z=after[i,3:]/np.linalg.norm(after[i,3:])
 v=q[3]*z[:3]-z[3]*q[:3]-np.cross(z[:3],q[:3]);w=np.dot(q,z)
 if w<0:v=-v;w=-w
 length=np.linalg.norm(v)
 if length:d[i,3:]=v*(2*np.arctan2(length,w)/length)
d=d.ravel();actualH=H@d
result['actualStepMetric']=float(d@actualH)
result['actualRayleighEnergy']=float(-d@(np.array(a['base'])-p['base'])/(d@actualH))
result['actualEnergyModelCorrection']=float(np.max(np.abs(np.linalg.solve(H,np.array(a['base'])-p['base']+actualH))/tol))
worst=6*p['sourceBlocks'].index(7)+5
result['worstStep']=dict(proposed=float(x[worst]),actual=float(d[worst]),tolerance=float(tol[worst]))
previous=np.array(p['initialForce']);mu=np.array(p['mu']);lo=np.array(p['lower'],dtype=float);hi=np.array(p['upper'],dtype=float)
for alpha in (1.,1.5,2.,4.):
 proposed=previous+alpha*(f-previous);viol=max(float(np.max(lo-proposed)),float(np.max(proposed-hi)),0.)
 cones=[np.hypot(proposed[3*r+1],proposed[3*r+2])-mu[r]*max(0,proposed[3*r]) for r in range(len(mu)) if p['coulomb'][r]]
 result['force'+str(alpha)]=dict(boundViolation=viol,coneViolation=max(cones,default=0.))
reaction0=J.T@f+np.array(p['base']);reaction1=A.T@f+np.array(a['base'])
slope0=-float(d@reaction0);slope1=-float(d@reaction1)
result['secant']=dict(slope0=slope0,slope1=slope1,alpha=slope0/(slope0-slope1))
(out/'decomposition.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
