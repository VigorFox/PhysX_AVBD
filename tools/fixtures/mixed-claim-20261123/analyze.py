"""Aggregate per-run timing pairs and the preregistered, frozen five-body frames."""
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
def load(name):
    return json.loads((root / name).read_text(encoding="utf-8"))

def percentile95(data):
    return sorted(data)[(len(data) - 1) * 95 // 100]

def paired(pairs, speed=False):
    baseline = [b for b, _ in pairs]
    candidate = [c for _, c in pairs]
    ratios = [c/b for b, c in pairs]
    return dict(baseline=statistics.median(baseline), candidate=statistics.median(candidate),
                ratio=statistics.median(ratios), favorable=sum(r>1 if speed else r<1 for r in ratios),
                pairs=len(pairs), ratios=ratios)

ids = set(load("registration.json")["ids"])
trace = load("trace.json")
result = {"frames": sorted(ids), "trace": {}, "timing": {}}
for w in (4, 8):
    result["trace"][w] = {}
    for side in ("probe", "candidate"):
        group = [r for r in trace if r["scene"] == "volume" and r["workers"] == w
                 and r["side"] == side and r["frame"] in ids]
        assert len(group) == len(ids) * 3
        assert all(r["external"] == r["submits"] for r in group)
        values = {k: statistics.mean(r[k] for r in group)
                  for k in ("submits", "sumWait", "readyIdle", "jobs", "maxWait")}
        values["meanWaitUs"] = 1000 * values["sumWait"] / values["submits"]
        if side == "candidate":
            values.update({k: statistics.mean(r[k] for r in group)
                           for k in ("waves", "claimed", "lanes", "tiles", "leaves", "itemWaitMs")})
            values["dispatchesAvoidedByClaim"] = values["claimed"] - values["lanes"]
            values["meanItemResidenceUs"] = 1000 * values["itemWaitMs"] / values["claimed"]
            values["maxClaimed"] = max(r["maxPerLane"] for r in group)
        result["trace"][w][side] = values

entries = load("entry.json")
assert len(entries) == 3 * 2 * 2 * 360
for w in (4, 8):
    for side in ("probe", "candidate"):
        group = [r for r in entries if r["workers"] == w and r["side"] == side and r["frame"] in ids]
        assert len(group) == 54
        waits = [float(r["sumMs"]) for r in group]
        callbacks = sum(int(r["external"]) for r in group)
        result["trace"][w][side].update(entryWaitMs=statistics.mean(waits),
            entryWaitUsPerCallback=1000 * sum(waits) / callbacks,
            entryMaxWaitMs=statistics.mean(float(r["maxMs"]) for r in group),
            entryWaitMsByRepeat=[statistics.mean(float(r["sumMs"]) for r in group if r["repeat"]==i) for i in range(3)])

original = load("ab/summary.json")["before"]
allpairs = {}
for batch, reverse, expected in (("aa", False, 4), ("ab", False, 8), ("confirm", True, 6)):
    report = load(batch + "/summary.json")
    assert report["allRunsPassed"] and report["completeWindows"] and report["stableBinaries"]
    reference = "candidate" if reverse else "baseline"
    variant = "baseline" if reverse else "candidate"
    assert report["before"][reference] == original["baseline"]
    assert report["before"][variant] == original["baseline" if batch == "aa" else "candidate"]
    cells = {}
    for run in report["runs"]:
        scene, w, i, label = (run[k] for k in ("scene", "workers", "repeat", "label"))
        values = dict(run["windows"]["1-360"], cpu=run["processCpuMs"] / 360)
        if scene == "volume":
            with (root / batch / f"{scene}-{w}-{i}-{label}.csv").open() as source:
                fixed = [float(r["simulateFetchMs"]) for r in csv.DictReader(source) if int(r["frame"]) in ids]
            assert len(fixed) == len(ids)
            values.update(fixedMean=statistics.mean(fixed), fixedP95=percentile95(fixed))
        cells[scene, w, i, label] = values
    assert len(cells) == 2 * 2 * expected * 2
    for scene in ("cloth", "volume"):
        metrics = ["meanMs", "p95Ms", "cpu"]
        if scene == "volume":
            metrics += ["fixedMean", "fixedP95"]
        for metric in metrics:
            for w in (4, 8, "4to8"):
                def value(i, side):
                    if w == "4to8":
                        return cells[scene, 4, i, side][metric] / cells[scene, 8, i, side][metric]
                    return cells[scene, w, i, side][metric]
                pairs = [(value(i, reference), value(i, variant)) for i in range(expected)]
                key = f"{scene}/{metric}/{w}"
                result["timing"].setdefault(batch, {})[key] = paired(pairs, w == "4to8")
                if batch != "aa":
                    allpairs.setdefault(key, []).extend(pairs)
result["timing"]["all"] = {k: paired(pairs, k.endswith("/4to8")) for k, pairs in allpairs.items()}
(root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
for key, data in result["timing"]["all"].items():
    print(key, round(data["baseline"], 3), round(data["candidate"], 3),
          round(100*(data["ratio"]-1), 2), data["favorable"])

