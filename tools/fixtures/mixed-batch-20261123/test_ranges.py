"""Check every mixed-wave cardinality accepted by the Checked CPU planner."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent
shapes=0
for workers in range(2,33):
    for tiles in range(1,3*workers+1):
        for leaves in range(1,4*workers-tiles+1):
            items=tiles+leaves
            lanes=min(workers,max(2,(3*items+3)//4))
            base,extra=divmod(items,lanes)
            spans=[]
            for slot in range(lanes):
                rank=slot-1 if slot else lanes-1
                begin=rank*base+min(rank,extra)
                size=base+int(rank<extra)
                assert 1<=size<=4
                if not slot:assert size==base
                spans.append((begin,begin+size))
            ordered=sorted(spans)
            assert ordered[0][0]==0 and ordered[-1][1]==items
            assert all(left[1]==right[0] for left,right in zip(ordered,ordered[1:]))
            shapes+=1
(p/'ranges.json').write_text(json.dumps(dict(shapes=shapes,maxItems=4,workers=[2,32],fullyCovered=True)))
print(shapes)

