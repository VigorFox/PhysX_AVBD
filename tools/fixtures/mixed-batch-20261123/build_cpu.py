from pathlib import Path
import re, shutil, subprocess, os
p=Path(__file__).resolve().parent
repo=p.parents[2]
build=repo/'physx/compiler/vc17win64-cpu-only/sdk_snippets_bin'
source=repo/'physx/snippets/snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDHeadless.cpp'
old=source.read_text()
def replace(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
s=replace(old,'using namespace physx;',r'''#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
struct ProcessSample { unsigned long long ticks, cycles; bool valid; };
static ProcessSample sampleProcess()
{
    FILETIME creation, end, kernel, user;
    ULONG64 cycles=0;
    const HANDLE process=GetCurrentProcess();
    if(!GetProcessTimes(process,&creation,&end,&kernel,&user) || !QueryProcessCycleTime(process,&cycles))return {0,0,false};
    const auto ticks=[](FILETIME v){return (static_cast<unsigned long long>(v.dwHighDateTime)<<32) | v.dwLowDateTime;};
    return {ticks(kernel)+ticks(user),cycles,true};
}
using namespace physx;''')
s=replace(s,'\t\tPxTime timer;\n\t\tok = gVolumeAvbdScene', '''        const PxU32 f=frame+1;
        const bool measure=(f>=91 && f<=97) || (f>=163 && f<=167) || (f>=274 && f<=279);
        const ProcessSample before=measure?sampleProcess():ProcessSample{};
        PxTime timer;
        ok = gVolumeAvbdScene''')
s=replace(s,'\t\tif(!ok) break;\n\t\ttotalStepMs += stepMs;\n\t\tmaximumStepMs = PxMax(maximumStepMs, stepMs);\n        stepTimes.push_back(stepMs);',r'''        if(!ok) break;
        if(measure)
        {
            const ProcessSample after=sampleProcess();
            if(!before.valid || !after.valid || after.ticks<before.ticks || after.cycles<before.cycles){ok=false;break;}
            std::printf("[PROCESS_FRAME] frame=%u ticks=%llu cycles=%llu\n",f,after.ticks-before.ticks,after.cycles-before.cycles);
        }
        totalStepMs += stepMs;
        maximumStepMs = PxMax(maximumStepMs, stepMs);
        stepTimes.push_back(stepMs);''')
out=p/'cpu-build';out.mkdir(exist_ok=True)
f=out/source.name;f.write_bytes(s.encode())
def lines(tool):
    return next((build/'SnippetDeformableVolumeAVBD.dir/checked').glob('*tlog/'+tool+'.command.1.tlog')).read_text(encoding='utf-16').splitlines()
cl=next(l for l in lines('CL') if l.startswith('/c ') and l.upper().endswith('SNIPPETDEFORMABLEVOLUMEAVBDHEADLESS.CPP'))
cl=cl[:cl.upper().rindex(' D:')]+' "'+str(f)+'"'
cl=re.sub(r'/Fo"[^"]+"',lambda m:'/Fo"'+str(out/'headless.obj')+'"',cl)
cl=re.sub(r'/Fd"[^"]+"',lambda m:'/Fd"'+str(out/'compiler.pdb')+'"',cl)
cl='/I"'+str(source.parent)+'" '+cl
linklines=lines('link'); i=next(i for i,l in enumerate(linklines) if l.startswith('/OUT:'))
link='\n'.join(linklines[i:]).split('\n^')[0]
for option,filename in [('OUT','CpuWindow.exe'),('PDB','CpuWindow.pdb'),('IMPLIB','CpuWindow.lib')]:
    link=re.sub('/'+option+r':"[^"]+"',lambda _: '/'+option+':"'+str(out/filename)+'"',link,flags=re.I)
link=re.sub(r'/ILK:"[^"]+"','',link,flags=re.I).replace(' /INCREMENTAL ',' /INCREMENTAL:NO ')
link=re.sub(re.escape(r'SNIPPETDEFORMABLEVOLUMEAVBD.DIR\CHECKED\SNIPPETDEFORMABLEVOLUMEAVBDHEADLESS.OBJ'),lambda _: '"'+str(out/'headless.obj')+'"',link,flags=re.I)
assert str(out/'headless.obj') in cl and str(out/'headless.obj') in link
for name,raw in [('cl',cl),('link',link)]: (out/(name+'.rsp')).write_text(raw)
batch=out/'build.cmd'
batch.write_text('@echo off\ncall "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul\nif errorlevel 1 exit /b 1\n'+'\n'.join(t+' @"'+str(out/(t+'.rsp'))+'"\nif errorlevel 1 exit /b 1' for t in ('cl','link')))
with (out/'build.log').open('wb') as stream:
    result=subprocess.run(['cmd','/d','/c',batch],cwd=build,env=dict(os.environ,PATH=r'C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem'),stdout=stream,stderr=subprocess.STDOUT)
if result.returncode:raise RuntimeError((out/'build.log').read_text(errors='replace')[-3500:])
base=p.parent/'color-overlap-20261123/baseline'
for name,run in [('base',base),('atomic',p.parent/'mixed-claim-20261123/candidate'),('batch',p/'batch')]:
    dest=p/('cpu-'+name);dest.mkdir(exist_ok=True)
    for dll in run.glob('*.dll'):shutil.copyfile(dll,dest/dll.name)
    shutil.copyfile(out/'CpuWindow.exe',dest/'CpuWindow.exe')

