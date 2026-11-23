"""Run the fixed-order three-way Checked study; each process completes before the next starts."""
import csv, itertools, hashlib, json, os, statistics, subprocess, sys, time
from pathlib import Path

p=Path(__file__).resolve().parent
repo=p.parents[2]
sys.path.insert(0,str(repo/'tools'))
from benchmark_avbd_simd import costs,identity
from benchmark_particle3_showcases import cpu_ticks

reg=json.loads((p/'registration.json').read_text())
fixed=set(reg['fixed'])
runtimes={'base':p.parent/'color-overlap-20261123/baseline','atomic':p.parent/'mixed-claim-20261123/candidate','batch':p/'batch'}
env={k:v for k,v in os.environ.items() if not k.startswith(('PHYSX_AVBD_','PHYSX_NP_','PHYSX_SNIPPET_'))}
env.update(PHYSX_AVBD_COMMON_STEP='budget',PHYSX_AVBD_LOCAL_ITERATIONS='4')
orders=list(itertools.permutations(runtimes))
report={'before':{s:identity(path) for s,path in runtimes.items()},'runs':[]}
expected=json.loads((repo/'tools/fixtures/mixed-claim-20261123/manifest.json').read_text())['runtime']
assert all(report['before'][s]['PhysX_64.dll']==expected[k] for s,k in [('base','baseline'),('atomic','candidate')])
assert all({k:v for k,v in vals.items() if k!='PhysX_64.dll'}=={k:v for k,v in report['before']['base'].items() if k!='PhysX_64.dll'} for vals in report['before'].values())
def save():
    (p/'study.json').write_text(json.dumps(report,indent=1))
def fields(line):
    return dict(token.split('=',1) for token in line.split()[1:] if '=' in token)
def rows(lines,tag):
    return [fields(line) for line in lines if line.startswith('['+tag+'] ')]
def acquire(mode,scene,w,rep,arm):
    folder=p/mode;folder.mkdir(exist_ok=True)
    name=f'{scene}-{w}-{rep}-{arm}'; log=folder/(name+'.log')
    path=runtimes[arm]
    exe='SnippetDeformable'+('Volume' if scene=='volume' else 'Surface')+'AVBD_64.exe'
    options=env.copy()
    csvfile=folder/(name+'.csv')
    if mode=='timing':options['PHYSX_SNIPPET_STEP_TIMES']=str(csvfile)
    elif mode=='cpu':
        path=p/('cpu-'+arm);exe='CpuWindow.exe'
    else:
        path=p/(arm+'-tail')
        options.update(PHYSX_AVBD_COMMON_STATS='1',PHYSX_AVBD_LOCAL_TIMELINE='1',PHYSX_AVBD_LOCAL_PROFILE='1')
    cmd=[str(path/exe),'--headless','--case='+('visual-showcase' if scene=='volume' else 'static-box'),'--frames=360',f'--dispatcher-threads={w}']
    with log.open('wb') as stream:
        system=cpu_ticks()
        with subprocess.Popen(cmd,cwd=repo,env=options,stdout=stream,stderr=subprocess.STDOUT) as proc:
            try: code=proc.wait(timeout=150)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait();raise
            cpu=cpu_ticks(proc)
        finish=cpu_ticks()
    lines=log.read_text(errors='replace').splitlines()
    gate=rows(lines,'AVBD_GATE')
    assert code==0 and len(gate)==1 and gate[0]['status']=='PASS' and gate[0]['completedFrames']=='360',(name,mode,code)
    if scene=='volume':
        events={r['pair']:r for r in rows(lines,'FIVE_REBOUND')}
        assert all(0<int(events[k]['contactFrame'])<int(events[k]['departureFrame'])<=330 for k in ('cone-ground','prism-sphere'))
    rec=dict(mode=mode,scene=scene,w=w,rep=rep,arm=arm,cpu360=cpu/10000/360,
             otherCPU=100*max(0,finish[1]-system[1]-cpu)/max(1,finish[0]-system[0]))
    if mode=='timing':
        with csvfile.open() as f:data=list(csv.DictReader(f))
        assert [int(x['frame']) for x in data]==list(range(1,361))
        vals=[float(x['simulateFetchMs']) for x in data];assert all(0<v<float('inf') for v in vals)
        rec.update(mean=statistics.mean(vals),p95=costs(vals)['p95Ms'])
        if scene=='volume':
            slow=[vals[i-1] for i in sorted(fixed)]
            rec.update(mean18=statistics.mean(slow),p9518=costs(slow)['p95Ms'])
    elif mode=='cpu':
        data=rows(lines,'PROCESS_FRAME')
        assert [int(x['frame']) for x in data]==sorted(fixed) and all(int(x['cycles'])>0 for x in data)
        rec.update(cpu18=sum(int(x['ticks']) for x in data)/10000/18,
                   mCycles18=sum(int(x['cycles']) for x in data)/1e6/18)
    else:
        steps=rows(lines,'AVBD_COMMON_FRAME'); claims=rows(lines,'AVBD_MIXED_CLAIMS'); disp=rows(lines,'AVBD_LOCAL_DISPATCH')
        assert len(steps)==len(claims)==len(disp)==360
        frame=0;counts=[];n=0;tail=[]
        for l in lines:
            if l.startswith('[AVBD_JOIN_FRAME] '):
                if frame:assert n==counts[-1]
                frame+=1;n=0;counts.append(int(fields(l)['waves']))
            elif l.startswith('[AVBD_JOIN] '):
                r={k:float(v) for k,v in fields(l).items()}
                assert 0<=r['alone']<=r['gap']+.001 and r['alone']<=r['service']+.001 and 1<=r['claims']<=4 and 2<=r['lanes']<=w
                n+=1
                if frame in fixed:tail.append(dict(frame=frame,**r))
        assert frame==360 and n==counts[-1]
        for i,(s,c,ds) in enumerate(zip(steps,claims,disp)):
            assert int(s['iterations'])==4 and int(s['bdf'])==int(s['poseForcePublications'])==1
            assert counts[i]==int(c['waves']) and int(c['tiles'])+int(c['leaves'])==int(c['claimed'])
            assert int(c['maxPerLane'])<=4 and int(ds['queuePeak'])<=4*w and int(ds['primalMaxRows'])<=256
        (folder/(name+'.json')).write_text(json.dumps(tail))
        rec.update(mixedFixed=len(tail),mixedAll=sum(counts))
    report['runs'].append(rec);save()
    print(mode,name,flush=True)
if __name__=='__main__':
    while True:
        try:q=json.loads((p/'qa/summary.json').read_text())
        except (ValueError,OSError):q={}
        if 'allPassed' in q:break
        time.sleep(2)
    assert q['allPassed'] and q['binariesUnchanged'] and len(q['checks'])==109
    for mode,count in [('tail',3),('cpu',6),('timing',12)]:
        for rep in range(count):
            workers=[4,8] if (rep+rep//6)%2==0 else [8,4]
            scenes=(['volume','cloth'] if rep%2==0 else ['cloth','volume']) if mode=='timing' else ['volume']
            for scene in scenes:
                for w in workers:
                    arms=orders[(rep+(3 if w==8 else 0)+(1 if scene=='cloth' else 0))%6]
                    if mode=='tail':arms=(['atomic','batch'] if (rep+(w==8))%2==0 else ['batch','atomic'])
                    for arm in arms:acquire(mode,scene,w,rep,arm)
    report['after']={s:identity(path) for s,path in runtimes.items()}
    assert report['before']==report['after']
    assert len(report['runs'])==192
    report['complete']=True;save()

