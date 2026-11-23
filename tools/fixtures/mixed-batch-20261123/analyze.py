"""Analyze complete matched triples; use per-repeat ratios and keep both halves."""
from pathlib import Path
import collections, json, statistics as st, sys
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
def load(f):return json.loads((p/f).read_text())
def q95(vals):return sorted(vals)[(len(vals)-1)*95//100]
reg=load('registration.json')
data=load('study.json')
def cost_summary(rows,scene,mode):
    runs={(r['rep'],r['w'],r['arm']):r for r in rows if r['scene']==scene and r['mode']==mode}
    n=reg['cpuProbeRepeats'] if mode=='cpu' else reg['repetitions']
    assert len(runs)==6*n
    metrics=['cpu18','mCycles18'] if mode=='cpu' else ['mean','p95','cpu360']+(['mean18','p9518'] if scene=='volume' else [])
    output={}
    for metric in metrics:
        for w in [4,8,'scale']:
            def sequence(model):
                if w=='scale':return [runs[i,4,model][metric]/runs[i,8,model][metric] for i in range(n)]
                return [runs[i,w,model][metric] for i in range(n)]
            values={key:sequence(key) for key in reg['models']}
            cell={'median':{key:st.median(v) for key,v in values.items()},'comparisons':{}}
            for new,old in [('atomic','base'),('batch','atomic'),('batch','base')]:
                ratios=[a/b for a,b in zip(values[new],values[old])]
                cell['comparisons'][new+'/'+old]=dict(ratio=st.median(ratios),
                    halves=[st.median(ratios[:n//2]),st.median(ratios[n//2:])],
                    favorable=sum((r>1 if w=='scale' else r<1) for r in ratios),n=n,ratios=ratios)
            output[metric+'/'+str(w)]=cell
    return output
def tail_summary():
    result={}
    for w in [4,8]:
        for arm in ['atomic','batch']:
            samples=[]
            for rep in range(3):
                samples += [dict(rep=rep,**r) for r in load(f'tail/volume-{w}-{rep}-{arm}.json')]
            assert all(r['frame'] in reg['fixed'] for r in samples)
            means={k:st.mean(r[k] for r in samples) for k in ('service','queue','offer','gap','alone','closed','hottest','claims','leaf','tile')}
            byframe={k:sum(r[k] for r in samples)/(3*len(reg['fixed']))/1000 for k in ('service','queue','gap','alone','closed','tile','leaf')}
            counts=collections.Counter(('leaf' if r['hotLeaf'] else 'tile')+'-'+str(int(r['hotKind'])) for r in samples)
            gaps={}
            for r in samples:
                k=('leaf' if r['hotLeaf'] else 'tile')+'-'+str(int(r['hotKind']))
                gaps[k]=gaps.get(k,0)+r['gap']
            def proportions(rows):
                return dict(ordinary=sum(not x['hotLeaf'] for x in rows)/len(rows),
                   inline=sum(x['winner']==0 for x in rows)/len(rows),longest=st.mean(x['longest'] for x in rows),
                   multi=sum(x['claims']>1 for x in rows)/len(rows))
            tail=sorted(samples,key=lambda r:r['gap'],reverse=True)[:max(1,len(samples)//10)]
            group=collections.defaultdict(list)
            for r in samples:group[r['frame']].append(r)
            by_repeat=[]
            for rep in range(3):
                waves=[r for r in samples if r['rep']==rep]
                by_repeat.append(dict(waves=len(waves),
                    gapMsPerFixedFrame=sum(r['gap'] for r in waves)/(1000*len(reg['fixed'])),
                    singleItemWaves=sum(r['claims']==1 for r in waves)))
            leaf_rows=collections.Counter(int(r['hotRows']) for r in samples if r['hotLeaf'])
            result[f'{arm}/{w}']=dict(waves=len(samples),perWaveUs=means,
                byRepeat=by_repeat,leafRows=leaf_rows,
                p95Us={k:q95([r[k] for r in samples]) for k in ('service','offer','queue','gap','alone','closed','hottest')},
                perFixedFrameMs=byframe,types=counts,gapsByTypeUs=gaps,proportions=proportions(samples),
                upperDecile=proportions(tail),upperDecileMeanUs={k:st.mean(r[k] for r in tail) for k in ('offer','queue','service','gap','alone','hottest')},
                concentration=sorted(((f, sum(x['gap'] for x in g)/3000) for f,g in group.items()),key=lambda pair:-pair[1])[:6],
                representative=sorted(tail,key=lambda r:r['gap'])[len(tail)//2],
                worst=max(samples,key=lambda r:r['gap']))
    return result
if __name__=='__main__':
    assert data['complete'] and len(data['runs'])==192
    result={f'{mode}/{scene}':cost_summary(data['runs'],scene,mode) for mode,scene in [('timing','volume'),('timing','cloth'),('cpu','volume')]}
    result['tail']=tail_summary()
    (p/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for section,cells in result.items():
        if section=='tail':continue
        for name,c in cells.items():
            print(section,name,[round(v,3) for v in c['median'].values()],[round(100*(v['ratio']-1),2) for v in c['comparisons'].values()])

