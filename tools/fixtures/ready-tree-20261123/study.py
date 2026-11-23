"""Checked paired measurements. Supply the local run directory; created runs are kept individually."""
from pathlib import Path
import csv,json,os,shutil,statistics as st,subprocess,sys,time
p=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
repo=p.parents[2];sys.path.insert(0,str(repo/'tools'))
from benchmark_particle3_showcases import cpu_ticks
from benchmark_avbd_simd import costs,identity
reg=json.loads((p/'registration.json').read_bytes());fixed=reg['fixed']
base=p.parent/'color-overlap-20261123/baseline'
bins={'base':base,'ready':p/'candidate'}
prev=p.parent/'mixed-batch-20261123/cpu-base'
env={k:v for k,v in os.environ.items() if not k.startswith(('PHYSX_AVBD_','PHYSX_NP_','PHYSX_SNIPPET_'))}
env.update(PHYSX_AVBD_COMMON_STEP='budget',PHYSX_AVBD_LOCAL_ITERATIONS='4')
def fields(lines,tag):
    return [dict(t.split('=',1) for t in l.split()[1:] if '=' in t) for l in lines if l.startswith('['+tag+'] ')]
def acquire(mode,scene,w,i,arm):
    folder=p/mode;folder.mkdir(exist_ok=True)
    name=f'{scene}-{w}-{i}-{arm}';csvpath=folder/(name+'.csv')
    sdk=bins[arm];options=env.copy()
    exe='SnippetDeformable'+('Volume' if scene=='five' else 'Surface')+'AVBD_64.exe'
    if mode=='diagnostic':
        sdk=p/('diag-'+arm)/'candidate'
        options.update(PHYSX_AVBD_COMMON_STATS='1',PHYSX_AVBD_LOCAL_PROFILE='1',PHYSX_AVBD_LOCAL_TIMELINE='1')
    elif mode=='cpu':sdk=p/('cpu-'+arm);exe='CpuWindow.exe'
    else:options['PHYSX_SNIPPET_STEP_TIMES']=str(csvpath)
    cmd=[sdk/exe,'--headless','--case='+('visual-showcase' if scene=='five' else 'static-box'),'--frames=360',f'--dispatcher-threads={w}']
    logfile=folder/(name+'.log')
    with logfile.open('wb') as log:
        before=cpu_ticks()
        with subprocess.Popen(cmd,cwd=repo,env=options,stdout=log,stderr=subprocess.STDOUT) as proc:
            try:rc=proc.wait(timeout=180)
            except subprocess.TimeoutExpired:proc.kill();proc.wait();raise
            cpu=cpu_ticks(proc)
        after=cpu_ticks()
    lines=logfile.read_text(errors='replace').splitlines();gate=fields(lines,'AVBD_GATE')
    assert rc==0 and len(gate)==1 and gate[0]['status']=='PASS' and gate[0]['completedFrames']=='360',(name,mode,rc)
    if scene=='five':
        events={r['pair']:r for r in fields(lines,'FIVE_REBOUND')}
        for pair in ['cone-ground','prism-sphere']:
            assert 0<int(events[pair]['contactFrame'])<int(events[pair]['departureFrame'])<=330
    metrics={'cpu360':cpu/10000/360,'otherCPU':100*max(0,after[1]-before[1]-cpu)/max(1,after[0]-before[0])}
    def select(rows):assert len(rows)==360;return [rows[j-1] for j in fixed]
    if mode=='diagnostic':
        phase=fields(lines,'AVBD_COMMON_FRAME')
        assert len(phase)==360 and all(r['iterations']=='4' and r['bdf']==r['poseForcePublications']=='1' for r in phase)
        caps=fields(lines,'AVBD_LOCAL_DISPATCH')
        assert len(caps)==360 and all(int(r['queuePeak'])<=4*w and int(r['primalMaxRows'])<=256 for r in caps)
        intervals=select(fields(lines,'AVBD_WIDE_INTERVAL'));tiles=select(fields(lines,'AVBD_LOCAL_TILES'))
        budget=select(fields(lines,'AVBD_LOCAL_BUDGET'))
        stage=select([r for r in fields(lines,'AVBD_LOCAL_STAGE') if r['stage']=='3'])
        mean=lambda rs,k:st.mean(float(r[k]) for r in rs)
        metrics.update(waves=mean(tiles,'waves'),treeMs=mean(intervals,'ms'),wideColors=mean(intervals,'colors'),
            stageSpan=mean(stage,'spanMs'),cohortWaves=mean(stage,'waves'),jobs=mean(tiles,'jobs'),
            tasks=mean(budget,'tasks'),leaves=mean(tiles,'leaves'),scratchPeak=max(int(r['scratchBytes']) for r in tiles))
        if arm=='ready':
            ready=select(fields(lines,'AVBD_READY_TREE'))
            assert all(a['leafWindows']==b['waves'] and a['cohorts']==c['waves'] for a,b,c in zip(ready,tiles,stage))
            metrics['multiCohorts']=mean(ready,'multiBlockCohorts')
    elif mode=='cpu':
        procframes=fields(lines,'PROCESS_FRAME')
        assert [int(r['frame']) for r in procframes]==fixed and all(int(r['cycles'])>0 for r in procframes)
        metrics.update(cpu18=st.mean(int(r['ticks']) for r in procframes)/10000,mCycles18=st.mean(int(r['cycles']) for r in procframes)/1e6)
    else:
        with csvpath.open() as stream:rows=list(csv.DictReader(stream))
        assert [int(r['frame']) for r in rows]==list(range(1,361))
        times=[float(r['simulateFetchMs']) for r in rows];assert all(0<x<float('inf') for x in times)
        metrics.update(mean=st.mean(times),p95=costs(times)['p95Ms'])
        if scene=='five':
            slow=select(times)
            metrics.update(mean18=st.mean(slow),p9518=costs(slow)['p95Ms'])
    return dict(mode=mode,scene=scene,w=w,rep=i,arm=arm,**metrics)

if __name__=='__main__':
    for arm,sdk in bins.items():
        dest=p/('cpu-'+arm);dest.mkdir(exist_ok=True)
        for f in sdk.glob('*.dll'):shutil.copyfile(f,dest/f.name)
        shutil.copyfile(prev/'CpuWindow.exe',dest/'CpuWindow.exe')
    before={arm:identity(sdk) for arm,sdk in bins.items()}
    assert before['base']['PhysX_64.dll'].startswith('6b3badbf') and before['ready']['PhysX_64.dll']!=before['base']['PhysX_64.dll']
    assert {k:v for k,v in before['base'].items() if k!='PhysX_64.dll'}=={k:v for k,v in before['ready'].items() if k!='PhysX_64.dll'}
    qfile=p/'qa/summary.json'
    while True:
        try:q=json.loads(qfile.read_bytes())
        except (ValueError,OSError):q={}
        if 'allPassed' in q:break
        time.sleep(3)
    assert len(q['checks'])==109 and q['binariesUnchanged'] and q['allPassed']
    assert q['after']['PhysX_64.dll']==before['ready']['PhysX_64.dll']
    result={'before':before,'runs':[]}
    for mode,n in reg['repeats'].items():
        for i in range(n):
            scenes=['five','cloth'] if mode=='timing' else ['five']
            if i%2:scenes.reverse()
            for scene in scenes:
                workers=[4,8] if (i//2+i)%2==0 else [8,4]
                for w in workers:
                    arms=list(bins)
                    if (i+(w==8)+(scene=='cloth'))%2:arms.reverse()
                    for arm in arms:
                        result['runs'].append(acquire(mode,scene,w,i,arm));print(mode,scene,w,i,arm,flush=True)
                        (p/'runs.json').write_text(json.dumps(result))
    result['after']={arm:identity(sdk) for arm,sdk in bins.items()}
    assert result['before']==result['after'] and len(result['runs'])==100
    result['complete']=True
    (p/'runs.json').write_text(json.dumps(result))

