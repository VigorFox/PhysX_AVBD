from collections import Counter
from pathlib import Path
import json, re
root=Path(__file__).parent
result={}
for name in ['five-root','five-components','five-final-t1','reaction-final-t4']:
    phases=Counter(); sweeps=Counter(); blocks=Counter();frames=[]
    counts=Counter(); total=Counter(); current=None
    for line in (root/(name+'.txt')).read_text(errors='replace').splitlines():
        values=dict(re.findall(r'(\w+)=([^ ]+)',line))
        if line.startswith('[AVBD_BOUNDED_RESPONSE]'):
            p=values['phase'];phases[p]+=1;sweeps[p]+=int(values['sweeps']);blocks[values['blocks']]+=1
            counts[p]+=1
        if line.startswith('[AVBD_COMMON_FRAME]'):
            current=dict(verdict=values, responses=dict(counts));counts.clear();frames.append(current)
        if line.startswith('[AVBD_BOUNDED_SCHEDULE]') and current is not None:
            current['schedule']={k:int(v) for k,v in values.items()}
            for k in ('waves','batches','sweeps','tasks'):total[k]+=int(values[k])
            current=None
    result[name]=dict(phases=dict(phases),sweeps=dict(sweeps),blockSizes=dict(blocks),schedule=dict(total),frames=frames)
(root/'accounting.json').write_text(json.dumps(result,indent=2),encoding='utf8')
for name, data in result.items():print(name,{k:v for k,v in data.items() if k!='frames'})
