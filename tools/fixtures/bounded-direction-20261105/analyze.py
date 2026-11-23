from pathlib import Path
from collections import Counter
import json, re

out = Path(__file__).parent
results = {}
for name in ('reaction-final', 'five-qualified', 'five-final'):
    text = (out / (name + '.txt')).read_text(errors='replace')
    phase = Counter(); sweeps = Counter(); status = Counter(); frames = []; current = Counter()
    for line in text.splitlines():
        if line.startswith('[AVBD_BOUNDED_RESPONSE]'):
            values = dict(re.findall(r'(\w+)=([^ ]+)', line))
            key = values['phase']; phase[key] += 1; sweeps[key] += int(values['sweeps'])
            status[values['status']] += 1; current[key] += 1
        if line.startswith('[AVBD_COMMON_FRAME]'):
            frames.append(dict(frame=len(frames)+1, responses=dict(current), verdict=line)); current.clear()
    results[name] = dict(responses=dict(phase), sweeps=dict(sweeps), status=dict(status), frames=frames,
                         unfinishedResponses=dict(current))
(out / 'response-accounting.json').write_text(json.dumps(results, indent=2), encoding='utf8')
for name, data in results.items():
    print(name, {k: v for k, v in data.items() if k != 'frames'})
