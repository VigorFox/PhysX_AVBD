"""Summarize the preregistered 18 frames and paired full-step measurements."""
import csv
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
def load(name):
    return json.loads((root / name).read_text(encoding="utf8"))
ids = load("registration.json")["indices"]
summary = {"fixedFrames": ids, "timing": {}, "trace": {}}
for label in ("ab", "aa"):
    report = load(label + "/summary.json")
    assert report["allRunsPassed"] and report["completeWindows"] and report["stableBinaries"]
    sample = {}
    for run in report["runs"]:
        scene, w, i, side = (run[k] for k in ("scene", "workers", "repeat", "label"))
        values = run["windows"]["1-360"].copy()
        if scene == "volume":
            filename = f"{scene}-{w}-{i}-{side}.csv"
            with (root / label / filename).open() as stream:
                frames = {int(row["frame"]): float(row["simulateFetchMs"]) for row in csv.DictReader(stream)}
            values["fixed18Ms"] = st.mean(frames[n] for n in ids)
        sample[scene,w,i,side] = values
    for scene in ("volume", "cloth"):
        result = {}
        count = len([k for k in sample if k[0]==scene and k[1]==4 and k[3]=="candidate"])
        assert count == (7 if label == "ab" else 3)
        for metric in ["meanMs","p95Ms"] + (["fixed18Ms"] if scene=="volume" else []):
            cells = {w: {side: [sample[scene,w,i,side][metric] for i in range(count)]
                         for side in ("baseline","candidate")} for w in (4,8)}
            cells["4to8"] = {side: [a/b for a,b in zip(cells[4][side],cells[8][side])]
                              for side in ("baseline","candidate")}
            for runs in cells.values():
                ratios = [c/b for c,b in zip(runs["candidate"],runs["baseline"])]
                runs.update(medianBaseline=st.median(runs["baseline"]),medianCandidate=st.median(runs["candidate"]),
                            pairedRatios=ratios,medianRatio=st.median(ratios))
            result[metric] = cells
        summary["timing"][label+"-"+scene] = result
rows = load("overlap-by-frame.json")
for w in (4,8):
    s = [r for r in rows if r["workers"]==w and r["frame"] in ids]
    assert len(s)==3*len(ids)
    summary["trace"][w] = dict(means={k:st.mean(r[k] for r in s) for k in
            ("bothMs","wideMs","ordinaryMs","waves","issued","totalLeaves")},
        framesWithOverlap=sum(r["bothMs"]>0 for r in s),samples=len(s),
        byFrame={n:{k:st.median(r[k] for r in s if r["frame"]==n) for k in ("bothMs","wideMs","issued","totalLeaves")} for n in ids})
(root/"analysis.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
for k,v in summary["timing"].items():
    for metric,cells in v.items():
        print(k,metric,[(w,round(s["medianBaseline"],4),round(s["medianCandidate"],4),round(s["medianRatio"],5)) for w,s in cells.items()])

# Every ratio below is experiment/reference, including the reversed-order confirmation.
summary["confirmation"] = {}
for scene, folder, workers in (("cloth","cloth-check",[8]), ("volume","five-check",[4,8])):
    report = load(folder+"/summary.json")
    assert report["allRunsPassed"] and report["completeWindows"] and report["stableBinaries"]
    reverse = scene=="volume"
    original=load("ab/summary.json")
    reference="candidate" if reverse else "baseline"
    assert report["before"][reference]==original["before"]["baseline"]
    assert report["before"]["baseline" if reverse else "candidate"]==original["before"]["candidate"]
    section = summary["confirmation"][scene] = {}
    vals={}
    for run in report["runs"]:
        i,w,label=(run[k] for k in ("repeat","workers","label"))
        side="baseline" if label==reference else "candidate"
        vals[w,i,side]=run["windows"]["1-360"].copy()
        if scene=="volume":
            with (root/folder/f"{scene}-{w}-{i}-{label}.csv").open() as stream:
                fr={int(r["frame"]):float(r["simulateFetchMs"]) for r in csv.DictReader(stream)}
            vals[w,i,side]["fixed18Ms"]=st.mean(fr[n] for n in ids)
    for metric in ("meanMs","p95Ms","fixed18Ms") if scene=="volume" else ("meanMs","p95Ms"):
        section[metric]={}
        for w in workers+(["4to8"] if len(workers)==2 else []):
            first=summary["timing"]["ab-"+scene][metric][w]
            def score(i,side):
                if w=="4to8":return vals[4,i,side][metric]/vals[8,i,side][metric]
                return vals[w,i,side][metric]
            second=[score(i,"candidate")/score(i,"baseline") for i in range(7)]
            allratios=first["pairedRatios"]+second
            section[metric][w]=dict(first=first["medianRatio"],confirmation=st.median(second),
                 overall=st.median(allratios),ratios=allratios,
                 favorable=sum(r>1 if w=="4to8" else r<1 for r in allratios),
                 baselineMs=st.median(first["baseline"]+[score(i,"baseline") for i in range(7)]),
                 candidateMs=st.median(first["candidate"]+[score(i,"candidate") for i in range(7)]))
(root/"analysis.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
for scene,data in summary["confirmation"].items():
    print(scene, {m:{w:round(stat["overall"],5) for w,stat in bins.items()} for m,bins in data.items()})
