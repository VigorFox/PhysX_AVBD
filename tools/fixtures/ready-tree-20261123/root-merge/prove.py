"""Test-only root completion events. No timestamps from this probe enter the benchmark."""
from pathlib import Path
import json,os,shutil,subprocess,sys
p=Path(__file__).resolve().parent
out=p/'progress';out.mkdir()
src=out/'src/avbd';shutil.copytree(p/'src/avbd',src)
sdk=out/'candidate';shutil.copytree(p.parent/'color-overlap-20261123/baseline',sdk)
cpp=src/'DyAvbdDynamics.cpp';s=cpp.read_text()
before='        // Offer every independent block before consuming its caller leaf.'
assert s.count(before)==1
s=s.replace(before,r'''        std::fprintf(stderr,"[PROGRESS_COHORT] blocks=%u\n",active);
'''+before)
before='            block.jobs+=block.count;'
assert s.count(before)==1
s=s.replace(before,r'''            std::fprintf(stderr,"[PROGRESS_ROOT] owner=%u attempt=%u\n",owner,block.attempt);
'''+before)
cpp.write_bytes(s.encode())
(out/'build.py').write_bytes((p/'build.py').read_text().replace('repo=p.parents[2]','repo=p.parents[3]').encode())
subprocess.run([sys.executable,out/'build.py'],check=True)
summary=[]
for w,sync in [(2,False),(4,False),(8,False),(8,True)]:
    cmd=[p/'stress/probe.exe','bending',str(w)]+(['inline'] if sync else [])
    env=dict(os.environ,PATH=str(sdk)+';'+os.environ['PATH'])
    proc=subprocess.run(cmd,cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True,timeout=60)
    log=proc.stdout.decode(errors='replace');(out/f'order-{w}-{int(sync)}.log').write_text(log)
    assert '[WIDE_STAR]' in log and 'status=PASS' in log
    groups=[]
    for line in log.splitlines():
        if line.startswith('[PROGRESS_COHORT]'):groups.append([])
        elif line.startswith('[PROGRESS_ROOT]'):
            row=dict(t.split('=') for t in line.split()[1:]);groups[-1].append((int(row['owner']),int(row['attempt'])))
    evidence=[g for g in groups if any(a>0 and any(other!=owner and phase==0 for other,phase in g[i+1:])
        for i,(owner,a) in enumerate(g))]
    if sync:assert evidence,groups
    summary.append(dict(workers=w,synchronous=sync,groups=len(groups),trialBeforePeerAssembly=len(evidence),example=evidence[0] if evidence else []))
print(summary)
(out/'summary.json').write_text(json.dumps(summary,indent=2))

