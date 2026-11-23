"""Summarize complete matched runs. Absolute samples and paired changes use separate medians."""
import json,statistics as st,sys
from pathlib import Path
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
runs=json.loads((p/'runs.json').read_bytes())
assert runs['complete'] and runs['before']==runs['after']
reg=json.loads((p/'registration.json').read_bytes())
out={}
for mode,n in reg['repeats'].items():
    for scene in (['five','cloth'] if mode=='timing' else ['five']):
        sel={(r['rep'],r['w'],r['arm']):r for r in runs['runs'] if r['mode']==mode and r['scene']==scene}
        assert len(sel)==n*4
        metrics={'diagnostic':['waves','cohortWaves','treeMs','stageSpan','wideColors','jobs','tasks','leaves','scratchPeak'],
                 'cpu':['cpu18','mCycles18'],'timing':['mean','p95','cpu360']+(['mean18','p9518'] if scene=='five' else [])}[mode]
        cells=out[mode+'/'+scene]={}
        for metric in metrics:
            for workers in [4,8,'scale']:
                def val(rep,arm):
                    if workers=='scale':return sel[rep,4,arm][metric]/sel[rep,8,arm][metric]
                    return sel[rep,workers,arm][metric]
                a,b=([val(i,arm) for i in range(n)] for arm in ['base','ready'])
                ratios=[y/x for x,y in zip(a,b)]
                cells[f'{metric}/{workers}']={'base':st.median(a),'ready':st.median(b),'ratio':st.median(ratios),
                    'halves':[st.median(ratios[:n//2]),st.median(ratios[n//2:])],
                    'favorable':sum(r>1 if workers=='scale' else r<1 for r in ratios),'n':n,'pairedRatios':ratios}
                if workers!='scale' or metric in ['mean','mean18','p9518']:
                    print(mode,scene,metric,workers,*[round(v,3) for v in [st.median(a),st.median(b),(st.median(ratios)-1)*100]])
cpu=[r['otherCPU'] for r in runs['runs'] if r['mode']=='timing']
out['externalCpuPercent']=[min(cpu),st.median(cpu),max(cpu)]
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')

