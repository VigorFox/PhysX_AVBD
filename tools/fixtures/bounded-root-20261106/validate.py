from pathlib import Path
import subprocess,sys,json,os
from concurrent.futures import ThreadPoolExecutor
root=Path.cwd(); out=root/'tools/benchmark-results/bounded-root-20261106';jobs=[]
for name in ('hello_world','joint','articulation_rc'):
 for solver in ('avbd','pgs','tgs'):
  jobs.append((f'public-{name}-{solver}',[sys.executable,f'tools/run_snippet_{name}_headless.py','--solver',solver,'--cases','gated'],{}))
for t in (0,1,4):
 jobs.append((f'publication-t{t}',[sys.executable,'tools/run_avbd_common_step_probe.py','--mode','bounded','--threads',str(t),'--chain-bodies','129','--logs',str(out/f'publication-t{t}')],{}))
jobs.extend([
 ('np-final',[sys.executable,'tools/run_avbd_np_geometry_scene_probe.py'],{'PHYSX_AVBD_COMMON_STEP':'bounded'}),
 ('smoke-final',[str(root/'tools/avbd_smoke.exe')],{}),
])
def run(job):
 name,args,extra=job;env=os.environ.copy();env.update(extra);env['PATH']=str(root/'physx/bin/win.x86_64.vc143.md/checked')+os.pathsep+env.get('PATH','')
 with (out/f'{name}.log').open('wb') as log:
  r=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
 result=dict(name=name,exit=r.returncode,passed=r.returncode==0);print(json.dumps(result),flush=True);return result
with ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(run,jobs))
(out/'regressions.json').write_text(json.dumps(results,indent=2),encoding='utf8')
sys.exit(any(not r['passed'] for r in results))
