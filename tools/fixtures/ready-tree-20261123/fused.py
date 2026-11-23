"""Ablate only the final binary join counter, using the block's window release."""
from pathlib import Path
import json,shutil,statistics as st,subprocess,sys,time
import study
p=Path(__file__).resolve().parent
def wait_json(path,field=None):
    while True:
        try:data=json.loads(path.read_bytes())
        except (ValueError,OSError):data={}
        if (field and isinstance(data,dict) and data.get(field)) or (not field and data):return
        time.sleep(2)
def change(source):
    cpp=source/'DyAvbdDynamics.cpp';header=source/'DyAvbdDynamics.h'
    s=cpp.read_text()
    s=s.replace('PxU32 begin=0,end=0,count=0,scratch=0,slot=0,slots=0;', 'PxU32 begin=0,end=0,count=0,rootStride=0,scratch=0,slot=0,slots=0;')
    s=s.replace('block.scratch=leaves;leaves+=block.count;', 'block.rootStride=1u<<PxHighestSetBit(block.count-1);\n        block.scratch=leaves;leaves+=block.count;')
    a=s.index('        if(!nodal)\n        {',s.index('void AvbdDynamicsContext::runReadyLeaf'))
    b=s.index('        work.pending[b]=2;',a)
    merge=s[a:b];s=s[:a]+'        mergeReadyChildren(owner,a,b);\n'+s[b:]
    merge='\n'.join(l[4:] for l in merge.rstrip().splitlines())
    method='''void AvbdDynamicsContext::mergeReadyChildren(PxU32 owner,PxU32 a,PxU32 b)
{
    auto& work=*mLocalWork;
    const auto& block=work.blocks[owner];
    const bool nodal=block.state.body>=mBodies.size(),trial=block.attempt!=0;
'''+merge+'\n}\n\n'
    s=s.replace('// The right child',method+'// The right child',1)
    s=s.replace('for(PxU32 stride=1;stride<block.count;stride*=2)', 'for(PxU32 stride=1;stride<block.rootStride;stride*=2)')
    s=s.replace('            block.jobs+=block.count;', '''            block.jobs+=block.count;
            // The last leaf release already joined both root subtrees.
            mergeReadyChildren(owner,block.scratch,block.scratch+block.rootStride);''')
    cpp.write_bytes(s.encode())
    h=header.read_text();key='    void runReadyLeaf(PxU32 owner,PxU32 leaf);'
    assert h.count(key)==1
    header.write_bytes(h.replace(key,'    void mergeReadyChildren(PxU32 owner,PxU32 a,PxU32 b);\n'+key).encode())
if __name__=='__main__':
    wait_json(p/'runs.json','complete')
    wait_json(p/'long/summary.json')
    out=p/'fused';out.mkdir(exist_ok=True)
    src=out/'src/avbd';shutil.copytree(p/'diag-ready/src/avbd',src)
    base=p.parent/'color-overlap-20261123/baseline'
    shutil.copytree(base,out/'candidate');change(src)
    (out/'build.py').write_text((p/'build.py').read_text().replace('repo=p.parents[2]','repo=p.parents[3]'))
    subprocess.run([sys.executable,out/'build.py'],check=True)
    # Keep the independent reread of the baseline and the new candidate in their own output directory.
    study.p=out
    study.bins={'base':base,'ready':out/'candidate'}
    for arm,target in [('base',p/'diag-base/candidate'),('ready',out/'candidate')]:
        folder=out/('diag-'+arm)/'candidate';folder.mkdir(parents=True)
        for file in target.iterdir():
            if file.suffix.lower() in ('.dll','.exe'):shutil.copyfile(file,folder/file.name)
    rows=[]
    for rep in range(6):
        for w in ([4,8] if rep%2==0 else [8,4]):
            arms=['base','ready'] if (rep+(w==8))%2==0 else ['ready','base']
            for arm in arms:
                rows.append(study.acquire('diagnostic','five',w,rep,arm))
    summary={}
    for w in [4,8]:
        by={(r['rep'],r['arm']):r for r in rows if r['w']==w}
        ratios=[by[i,'ready']['treeMs']/by[i,'base']['treeMs'] for i in range(6)]
        summary[w]=dict(ratios=ratios,meanHalves=[st.median(ratios[:3]),st.median(ratios[3:])],ratio=st.median(ratios),
            base=st.median(by[i,'base']['treeMs'] for i in range(6)),ready=st.median(by[i,'ready']['treeMs'] for i in range(6)))
    passed=max(summary[8]['meanHalves'])<1 and min(summary[4]['meanHalves'])<=1
    (out/'summary.json').write_text(json.dumps(dict(passed=passed,workers=summary,runs=rows),indent=2))
    print(summary,passed)
    
