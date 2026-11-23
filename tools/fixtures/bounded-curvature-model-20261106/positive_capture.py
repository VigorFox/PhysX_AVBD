from pathlib import Path
import json,numpy as np
out=Path(__file__).parent;p=json.loads((out/'original-candidate.json').read_text());data=np.load(out/'operators.npz')
p['metricH']=p['H'];K=data['exact']
p['H']=[[a,b,*K[6*a:6*a+6,6*b:6*b+6].ravel().tolist()] for a,b,*v in p['metricH']]
(out/'positive-frame-program.json').write_text(json.dumps(p))
