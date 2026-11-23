"""Matched mechanistic screen of an isolated 256-row first leaf."""
import json
import statistics
from pathlib import Path

import study

work = Path(__file__).resolve().parent
screen = work / "screen"

def check_ranges():
    trials = 0
    for workers in range(2, 33):
        for tiles in range(1, 3 * workers + 1):
            for leaves in range(1, 4 * workers - tiles + 1):
                count = tiles + leaves
                lanes = min(workers, max(2, (3 * count + 3) // 4))
                for full in (False, True):
                    reserved = int(full and count - 1 <= (lanes - 1) * 4)
                    size, extra = divmod(count - reserved, lanes - reserved)
                    spans = []
                    for slot in range(lanes):
                        if reserved and slot == 0:
                            spans.append((0, 1))
                        else:
                            rank = slot - 1 if slot else lanes - 1
                            begin = reserved + rank * size + min(rank, extra)
                            spans.append((begin, begin + size + int(rank < extra)))
                    assert all(1 <= b - a <= 4 for a, b in spans)
                    ordered = sorted(spans)
                    assert ordered[0][0] == 0 and ordered[-1][1] == count
                    assert all(a[1] == b[0] for a, b in zip(ordered, ordered[1:]))
                    trials += 1
    return trials

def main():
    assert json.loads((work / "study.json").read_text())["complete"]
    screen.mkdir(exist_ok=True)
    result = {"rangeCases": check_ranges(), "workers": {}}
    # Reuse only the test executor. Its main three-way archive was finalized.
    study.save = lambda: None
    study.runtimes["reserved"] = work / "reserved-tail"
    for rep in range(3):
        for workers in ((4, 8) if rep % 2 == 0 else (8, 4)):
            arms = ["atomic", "reserved"]
            if (rep + int(workers == 8)) % 2:
                arms.reverse()
            cell = result["workers"].setdefault(str(workers), {"atomic": [], "reserved": []})
            for arm in arms:
                study.acquire("tail", "volume", workers, 10 + rep, arm)
                name = f"volume-{workers}-{10 + rep}-{arm}"
                for suffix in [".log", ".json"]:
                    key = name + suffix
                    (work / "tail" / key).replace(screen / key)
                waves = json.loads((screen / (name + ".json")).read_text())
                observed = [r for r in waves if r.get("reserved")]
                if arm == "reserved":
                    assert observed and all(r["winner"] != 0 or r["claims"] == 1 for r in observed)
                cell[arm].append(dict(
                    waves=len(waves), reservedWaves=len(observed),
                    gapMsPerFixedFrame=sum(r["gap"] for r in waves) / 18000,
                    serviceUs=statistics.mean(r["service"] for r in waves),
                    queueUs=statistics.mean(r["queue"] for r in waves),
                    ordinaryHottest=sum(not r["hotLeaf"] for r in waves) / len(waves)))
    for cell in result["workers"].values():
        diffs = [r["gapMsPerFixedFrame"] - a["gapMsPerFixedFrame"]
                 for r, a in zip(cell["reserved"], cell["atomic"])]
        cell["favorablePairs"] = sum(delta < 0 for delta in diffs)
        cell["medianDeltaMs"] = statistics.median(diffs)
    result["passed"] = all(c["favorablePairs"] >= 2 for c in result["workers"].values())
    (screen / "summary.json").write_text(json.dumps(result, indent=2))
    print(result)

if __name__ == "__main__":
    main()

