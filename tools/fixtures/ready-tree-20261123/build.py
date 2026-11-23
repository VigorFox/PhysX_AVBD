from pathlib import Path
import re,subprocess,sys,os
p=Path(__file__).resolve().parent
repo=p.parents[2]
build=repo/'physx/compiler/vc17win64-cpu-only/sdk_source_bin'
base=repo/'physx/bin/win.x86_64.vc143.md/checked'
dst=p/'candidate'
src=p/'src/avbd'
def log(target,tool):
    f=next((build/(target+'.dir/checked')).glob('*tlog/'+tool+'.command.1.tlog'))
    return f.read_text(encoding='utf-16').splitlines()
def rsp(target,tool):
    lines=log(target,tool);idx=next(i for i,l in enumerate(lines) if l.startswith('/OUT:'))
    return '\n'.join(lines[idx:]).split('\n^')[0]
def opt(s,flag,path):
    return re.sub('/'+flag+r'"[^"]+"',lambda _:'/'+flag+'"'+str(path)+'"',s,flags=re.I)
lib=rsp('LowLevelDynamics','Lib')
link=rsp('PhysX','link')
cmds=[]
for tag in ['DyAvbdDynamics','DyAvbdSolve']:
    cl=next(l for l in log('LowLevelDynamics','CL') if l.startswith('/c ') and l.upper().endswith(tag.upper()+'.CPP'))
    cl=cl[:cl.upper().rindex(' D:')]+' "'+str(src/(tag+'.cpp'))+'"'
    cl=opt(opt(cl,'Fo',p/(tag+'.obj')),'Fd',p/(tag+'.pdb'))
    cl='/I"'+str(src.parent)+'" /I"'+str(repo/'physx/source/lowleveldynamics/src/avbd')+'" '+cl
    target='LOWLEVELDYNAMICS.DIR\\CHECKED\\'+tag.upper()+'.OBJ'
    assert target in lib
    lib=lib.replace(target,'"'+str(p/(tag+'.obj'))+'"')
    cmds.append(('cl',tag,cl))
for path in [str(base),base.as_posix()]:
    lib=re.sub(re.escape(path),lambda _:str(dst),lib,flags=re.I)
    link=re.sub(re.escape(path),lambda _:str(dst),link,flags=re.I)
link=link.replace(' /MAP ',' /MAP:"'+str(dst/'PhysX_64.map')+'" ')
cmds.extend([('lib','lib',lib),('link','link',link)])
batch=['@echo off',r'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul','if errorlevel 1 exit /b 1']
for exe,name,args in cmds:
    f=p/(name+'.rsp');f.write_text(args)
    batch.extend([f'{exe} @"{f}"','if errorlevel 1 exit /b 1'])
script=p/'build.cmd';script.write_text('\n'.join(batch))
with (p/'build.log').open('wb') as out:
    code=subprocess.run(['cmd','/d','/c',script],cwd=build,env=dict(os.environ,PATH=r'C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem'),stdout=out,stderr=subprocess.STDOUT).returncode
if code:print((p/'build.log').read_text(errors='replace')[-3500:])
sys.exit(code)

