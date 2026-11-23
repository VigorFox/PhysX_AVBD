from pathlib import Path
import json,os,re,subprocess,sys
p=Path(__file__).resolve().parent;repo=p.parents[2];sys.path.insert(0,str(repo/'tools'))
from run_avbd_ogc_certificate_probe import build_probe
tmp=p/'rigid';tmp.mkdir(exist_ok=True)
os.environ['PATH']=r'C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem'
libs=['PhysXExtensions_static_64.lib','PhysXPvdSDK_static_64.lib','PhysXCooking_64.lib','PhysX_64.lib','PhysXCommon_64.lib']
assert build_probe(tmp,p/'candidate',p/'wide-rigid.cpp',include_dirs=[repo/'tools'],libraries=libs)==0
results=[]
for name,sdk in [('base',p.parent/'color-overlap-20261123/baseline'),('ready',p/'candidate')]:
    for w in (0,8):
        env=dict(os.environ,PATH=str(sdk)+';'+os.environ['PATH'],PHYSX_AVBD_COMMON_STEP='budget',PHYSX_AVBD_LOCAL_ITERATIONS='4',PHYSX_AVBD_COMMON_STATS='1')
        result=subprocess.run([tmp/'probe.exe',str(w)],cwd=tmp,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
        log=result.stdout.decode(errors='replace');(tmp/f'{name}-{w}.log').write_text(log)
        peak=max(map(int,re.findall(r'maxIncidentRows=(\d+)',log)),default=0)
        print(name,w,peak,log.splitlines()[-1],flush=True)
        assert result.returncode==0 and 'status=PASS' in log and peak>256
        results.append(dict(arm=name,w=w,rows=peak,result=log.splitlines()[-1]))
(tmp/'summary.json').write_text(json.dumps(results,indent=2))

