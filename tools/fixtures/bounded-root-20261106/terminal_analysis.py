from pathlib import Path
import json
import numpy as np

root=Path(__file__).parent
p=json.loads((root/'terminal-candidate.json').read_text())
trial=json.loads((root/'terminal-candidate.json.trial.json').read_text())
actual=trial['actual'];n=len(p['base']);starts=np.r_[0,np.cumsum(p['dimensions'])]
H=np.zeros((n,n));J=np.zeros((len(p['offset']),n));A=np.zeros_like(J)
for a,b,*values in p['H']:
    H[starts[a]:starts[a+1],starts[b]:starts[b+1]]+=np.array(values).reshape(6,6)[:p['dimensions'][a],:p['dimensions'][b]]
for program,matrix in [(p,J),(actual,A)]:
    for a,b,*values in program['J']:
        matrix[3*a:3*a+3,starts[b]:starts[b+1]]+=np.array(values).reshape(3,6)[:,:p['dimensions'][b]]
step=np.array(trial['step']);force=np.array(trial['force']);stored=np.array(trial['storedForce'])
rhs=np.array(actual['base'])+A.T@stored
unrounded=np.array(actual['base'])+A.T@force
frozen=np.array(p['base'])+J.T@force-H@step
correction=np.linalg.solve(H,np.array([rhs,unrounded,frozen]).T)
tolerance=np.array(p['motionTolerance'])
ratios=np.abs(correction)/tolerance[:,None]
index=int(np.argmax(ratios[:,0]));rounding=correction[:,0]-correction[:,1]
before=np.array(trial['from']).reshape(-1,7);after=np.array(trial['to']).reshape(-1,7)
delta=np.zeros((len(before),6));delta[:,:3]=after[:,:3]-before[:,:3]
for i in range(len(before)):
    a=before[i,3:]/np.linalg.norm(before[i,3:]);b=after[i,3:]/np.linalg.norm(after[i,3:])
    v=a[3]*b[:3]-b[3]*a[:3]-np.cross(b[:3],a[:3]);w=np.dot(a,b)
    if w<0:v=-v;w=-w
    length=np.linalg.norm(v)
    if length:delta[i,3:]=v*(2*np.arctan2(length,w)/length)
result=dict(blocks=len(p['dimensions']),rows=len(p['mu']),
            body=p['sourceBlocks'][index//6],coordinate=index%6,
            actualCorrectionRatio=float(ratios[:,0].max()),
            unroundedForceCorrectionRatio=float(ratios[:,1].max()),
            frozenCorrectionRatio=float(ratios[:,2].max()),
            forceRoundingCorrectionRatio=float(np.max(np.abs(rounding)/tolerance)),
            storedMotionDifferenceRatio=float(np.max(np.abs(delta.reshape(-1)-step)/tolerance)),
            solveResidual=float(np.max(np.abs(H@correction[:,0]-rhs))),
            scope='offline full-H diagnostic only; not a production solve or physical qualification')
(root/'terminal-analysis.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
