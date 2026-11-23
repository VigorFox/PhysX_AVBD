from pathlib import Path
import re,shutil,os,subprocess,sys
p=Path(__file__).resolve().parent
ws=p.parents[2]
b=ws/'physx/compiler/vc17win64-cpu-only/sdk_source_bin'
variant=sys.argv[1] if len(sys.argv)>1 else 'candidate'
out=p/variant
src=p/('src' if variant=='candidate' else variant+'-src')
tag='DyAvbdDynamics'
def lines(t,k):
 a=next((b/(t+'.dir/checked')).glob('*tlog/'+k+'.command.1.tlog')).read_text(encoding='utf-16').splitlines();return a
cl=next(l for l in lines('LowLevelDynamics','CL') if l.startswith('/c ') and l.upper().endswith('DYAVBDDYNAMICS.CPP'))
cl=re.sub(r'/Fo"[^"]+"',lambda m:'/Fo"'+str(p/(tag+'.obj'))+'"',cl)
cl=re.sub(r'/Fd"[^"]+"',lambda m:'/Fd"'+str(p/(tag+'.pdb'))+'"',cl)
cl=cl[:cl.upper().rindex(' D:')]+' "'+str(src/'avbd/DyAvbdDynamics.cpp')+'"'
cl='/I"'+str(src)+'" /I"'+str(ws/'physx/source/lowleveldynamics/src/avbd')+'" '+cl
def rsp(t,k):
 a=lines(t,k);i=next(i for i,l in enumerate(a) if l.startswith('/OUT:'));return '\n'.join(a[i:]).split('\n^')[0]
lib=rsp('LowLevelDynamics','Lib'); link=rsp('PhysX','link')
orig=ws/'physx/bin/win.x86_64.vc143.md/checked'
for pre in [str(orig),orig.as_posix()]:
 lib=re.sub(re.escape(pre),lambda m:str(out),lib,flags=re.I)
 link=re.sub(re.escape(pre),lambda m:str(out),link,flags=re.I)
lib=re.sub(re.escape('LOWLEVELDYNAMICS.DIR\\CHECKED\\DYAVBDDYNAMICS.OBJ'),lambda m:str(p/(tag+'.obj')),lib,flags=re.I)
link=link.replace(' /MAP ', ' /MAP:"'+str(out/'PhysX_64.map')+'" ')
assert str(p/(tag+'.obj')) in lib and str(out).upper() in link.upper()
for name,raw in [('cl',cl),('lib',lib),('link',link)]: (p/(name+'.rsp')).write_text(raw,encoding='utf-8')
batch=p/'build.cmd'
batch.write_text('@echo off\ncall "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul\nif errorlevel 1 exit /b 1\n'+'\n'.join(tool+' @"'+str(p/(tool+'.rsp'))+'"\nif errorlevel 1 exit /b 1' for tool in ('cl','lib','link'))+'\n',encoding='ascii')
env=dict(os.environ,PATH=r'C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem')
with (p/(variant+'-build.log')).open('wb') as f:
 r=subprocess.run(['cmd','/d','/c',str(batch)],cwd=b,stdout=f,stderr=subprocess.STDOUT,env=env)
if r.returncode:print((p/(variant+'-build.log')).read_text(errors='replace')[-2000:])
sys.exit(r.returncode)

