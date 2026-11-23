"""Read-only exact row grouping audit. Never modifies a captured program."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile


def audit(path):
    data = path.read_bytes()
    p = json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)
    nr = len(p['mu'])
    adjacency = [[] for _ in range(nr)]
    for r, b, *values in p['J']:
        adjacency[r].append((b, values))
    tangent_groups, full_groups, hard_parallel = defaultdict(list), defaultdict(list), defaultdict(list)
    ineligible = 0
    for r, entries in enumerate(adjacency):
        entries.sort(key=lambda item: item[0])
        k = 3 * r
        # Match CommonProgram.sameTangent exactly: no angle/offset tolerances.
        eligible = p['coulomb'][r] and p['lower'][k] >= 0 and all(
            p['compliance'][k + q] == 0 and float(p['lower'][k + q]) == -math.inf
            and float(p['upper'][k + q]) == math.inf for q in (1, 2))
        if eligible:
            tangent = (p['mu'][r], *p['offset'][k + 1:k + 3], tuple(
                (b, tuple(v[6:6 + p['dimensions'][b]] + v[12:12 + p['dimensions'][b]])) for b, v in entries))
            tangent_groups[tangent].append(r)
            if p['lower'][k] == 0 and float(p['upper'][k]) == math.inf and p['compliance'][k] == 0:
                normal = tuple((b, tuple(v[:p['dimensions'][b]])) for b, v in entries)
                hard_parallel[tangent, normal].append(r)
        else:
            ineligible += 1
        # Whole-row duplicates require the complete law and coordinate map.
        key = (p['coulomb'][r], p['mu'][r], *p['offset'][k:k + 3],
               *map(float, p['lower'][k:k + 3]), *map(float, p['upper'][k:k + 3]),
               *p['compliance'][k:k + 3], tuple((b, tuple(v)) for b, v in entries))
        full_groups[key].append(r)
    packed = nr + 2 * len(tangent_groups) + 2 * ineligible
    force = p.get('initialForce', [0.] * (3 * nr))
    by_support = Counter(tuple(b for b, _ in entries) for entries in adjacency)
    roundtrip = None
    if p.get('replay', {}).get('solved'):
        from validate_avbd_common_capture import validate
        before = validate(path)
        old_force = p['replay']['force']
        new_force = old_force.copy()
        for rows in tangent_groups.values():
            total = sum(old_force[3*r] for r in rows)
            assert all(old_force[3*r] >= 0 for r in rows), 'infeasible captured normal'
            tangent = [sum(old_force[3*r+q] for r in rows) for q in (1, 2)]
            for r in rows:
                for q in (1, 2):
                    new_force[3*r+q] = old_force[3*r] / total * tangent[q-1] if total else 0.
        p['replay'] = dict(p['replay'], force=new_force, source='shared tangent pack/unpack of a solved capture; no solve')
        with tempfile.TemporaryDirectory(prefix='avbd_tangent_audit_') as temp:
            replay_path = Path(temp) / 'roundtrip.json'
            replay_path.write_text(json.dumps(p), encoding='utf8')
            after = validate(replay_path)
        keys = ('passed', 'relativeStationarity', 'relativeUnitStepRoot', 'activeNormalForces')
        roundtrip = dict(before={k: before[k] for k in keys}, after={k: after[k] for k in keys},
                         maxForceChange=max(map(lambda pair: abs(pair[0]-pair[1]), zip(old_force, new_force)), default=0.))
        assert before['passed'] and after['passed'], roundtrip
    return dict(file=str(path), sha256=hashlib.sha256(data).hexdigest(), blocks=len(p['dimensions']),
                rowGroups=nr, scalarCoordinates=3 * nr,
                exactWholeRowClasses=len(full_groups), wholeRowCompressionAssumed=False,
                tangentClasses=len(tangent_groups), tangentIneligible=ineligible,
                tangentClassSizes=dict(sorted(Counter(map(len, tangent_groups.values())).items())),
                tangentPackedCoordinates=packed,
                exactHardParallelClasses=len(hard_parallel),
                hardParallelRows=sum(map(len, hard_parallel.values())),
                frozenDominanceCandidateSavings=sum(len(rows)-1 for rows in hard_parallel.values()),
                tangentCoordinateReduction=1 - packed / (3 * nr) if nr else 0,
                nonzeroCapturedForceRows=sum(any(v != 0 for v in force[3*r:3*r+3]) for r in range(nr)),
                rowsPerSupport=dict(sorted(Counter(by_support.values()).items())),
                materialOperatorBlocks=len(p['H']), metricBlocks=len(p.get('metricH', [])),
                solvedCaptureTangentRoundtrip=roundtrip,
                scope='exact equality inventory only; not a convergence, candidate-domain or speed qualification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path, nargs='+')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    results = [audit(path) for path in args.captures]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2), encoding='utf8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
