from pathlib import Path
import json, os, re, subprocess, sys
p=Path(__file__).resolve().parent
repo=p.parents[2];sys.path.insert(0,str(repo/'tools'))
from run_avbd_ogc_certificate_probe import build_probe
tmp=p/'stress';tmp.mkdir(exist_ok=True)
s=(repo/'tools/avbd_nodal_wide_star_probe.cpp').read_text().replace('#include <cstring>','#include <cstring>\n#include <vector>')
a=s.index('    const PxU32 faces=288;');b=s.index('    PxCookingParams cooking',a)
s=s[:a]+'''    const PxU32 counts[]={288,513,1297};
    const PxU32 stars=3,faces=2098;
    std::vector<PxVec3> vertices;
    std::vector<PxU32> indices;
    for(PxU32 j=0;j<stars;++j)
    {
        const PxU32 first=PxU32(vertices.size()),count=counts[j];
        const PxVec3 origin(6.f*j,2,0);vertices.push_back(origin);
        for(PxU32 i=0;i<count;++i)
        {
            const PxReal a=PxTwoPi*PxReal(i)/count;
            vertices.push_back(origin+1.5f*PxVec3(PxCos(a),0,PxSin(a)));
            indices.push_back(first);indices.push_back(first+1+(i+1)%count);indices.push_back(first+1+i);
        }
    }
    PxTriangleMeshDesc meshDesc;
    meshDesc.points.count=PxU32(vertices.size());meshDesc.points.data=vertices.data();meshDesc.points.stride=sizeof(PxVec3);
    meshDesc.triangles.count=faces;meshDesc.triangles.data=indices.data();meshDesc.triangles.stride=3*sizeof(PxU32);
'''+s[b:]
s=s.replace('PxU32 center=PX_MAX_U32;','PxU32 centers[stars]={PX_MAX_U32,PX_MAX_U32,PX_MAX_U32};')
s=s.replace('const bool free=(x-vertices[0]).magnitudeSquared()<1.e-8f;', 'const PxU32 j=PxU32((x.x+3)/6);const bool free=(x-PxVec3(6.f*j,2,0)).magnitudeSquared()<1.e-8f;')
s=s.replace('center=i;','centers[j]=i;').replace('center==PX_MAX_U32','centers[0]==PX_MAX_U32 || centers[1]==PX_MAX_U32 || centers[2]==PX_MAX_U32')
s=s.replace('PxReal best=.02f,','PxReal best[stars]={.02f,.02f,.02f},')
s=s.replace('if(i==center){best=PxMin(best,error);peak=PxMax(peak,error);}', 'const PxU32 j=PxU32((rest[i].x+3)/6);if(i==centers[j]){best[j]=PxMin(best[j],error);peak=PxMax(peak,error);}')
s=s.replace('best<.019f','best[0]<.019f && best[1]<.019f && best[2]<.019f').replace('double(best)','double(PxMax(best[0],PxMax(best[1],best[2])))')
s=s.replace('int main(int argc,char** argv)', '''class InlineDispatcher : public PxCpuDispatcher
{
    PxU32 width;
public:
    explicit InlineDispatcher(PxU32 workers):width(workers) {}
    void submitTask(PxBaseTask& task) override {task.run();task.release();}
    PxU32 getWorkerCount() const override {return width;}
};
int main(int argc,char** argv)''')
s=s.replace('    PxSceneDesc desc(', '    InlineDispatcher inlineDispatcher(workers);\n    PxSceneDesc desc(')
s=s.replace('desc.cpuDispatcher=dispatcher;', 'desc.cpuDispatcher=argc>3?static_cast<PxCpuDispatcher*>(&inlineDispatcher):dispatcher;')
src=tmp/'stars.cpp';src.write_bytes(s.encode())
base=p.parent/'color-overlap-20261123/baseline'
os.environ['PATH']=r'C:\MinGW\bin;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem'
assert build_probe(tmp,p/'candidate',src,include_dirs=[repo/'tools'],libraries=['PhysXExtensions_static_64.lib','PhysXPvdSDK_static_64.lib','PhysXCooking_64.lib','PhysX_64.lib','PhysXCommon_64.lib'])==0
rows=[]
for w,inline in [(0,False),(2,False),(4,False),(8,False),(8,True)]:
    for arm,d in [('base',base),('ready',p/'candidate')]:
        env=dict(os.environ,PATH=str(d)+';'+os.environ['PATH'],PHYSX_AVBD_COMMON_STATS='1')
        out=subprocess.run([tmp/'probe.exe','bending',str(w)]+(['inline'] if inline else []),cwd=tmp,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
        log=out.stdout.decode(errors='replace');(tmp/f'{arm}-{w}-{int(inline)}.log').write_text(log)
        multi=[int(k) for k in re.findall(r'multiBlockCohorts=(\d+)',log)]
        maxrows=max(map(int,re.findall(r'maxIncidentRows=(\d+)',log)),default=0)
        assert out.returncode==0 and re.search(r'\[WIDE_STAR\].*status=PASS',log),(arm,w,log[-1000:])
        if arm=='ready' and w>=4:assert max(multi)>0,(arm,w,multi)
        rows.append(dict(arm=arm,w=w,inline=inline,multiCohorts=sum(multi),maxRows=maxrows,gate=next(l for l in log.splitlines() if l.startswith('[WIDE_STAR]'))))
print(rows);(tmp/'summary.json').write_text(json.dumps(rows,indent=2))

