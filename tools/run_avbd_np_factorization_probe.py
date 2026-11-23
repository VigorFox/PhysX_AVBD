"""Verify complete path queries, including safe overhang and real intersections."""
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np
from run_avbd_ogc_certificate_probe import ROOT, run_probe


def rotation(quaternion):
    x, y, z, w = np.asarray(quaternion) / np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def convex_sat_gap(pair):
    """Independent, offline exhaustive SAT for the finite unrounded fixtures."""
    vertices, axes, edges = [], [], []
    for side in pair['sides']:
        assert side['radius'] == 0 and 4 <= len(side['points']) <= 32
        pose = side['end']
        world = np.array(side['points']) @ rotation(pose[3:]).T + pose[:3]
        vertices.append(world)
        edges.append([b-a for a, b in itertools.combinations(world, 2)])
        for a, b, c in itertools.combinations(world, 3):
            axis = np.cross(b-a, c-a)
            if np.linalg.norm(axis) < 1e-12:
                continue
            axis /= np.linalg.norm(axis)
            values = (world-a) @ axis
            if values.min() >= -1e-10 or values.max() <= 1e-10:
                axes.append(axis)
    # Includes every true hull edge pair (and harmless redundant directions).
    axes.extend(np.cross(a, b) for a in edges[0] for b in edges[1])
    gap = -float('inf')
    for axis in axes:
        if np.linalg.norm(axis) < 1e-12:
            continue
        axis = axis / np.linalg.norm(axis)
        a, b = (v @ axis for v in vertices)
        gap = max(gap, a.min()-b.max(), b.min()-a.max())
    return gap - pair['rest']


def fixtures(directory):
    result = []
    captures = sorted((ROOT / 'tools/fixtures/local-witness-20261111').glob('*.json'))
    captures += sorted((ROOT / 'tools/fixtures/local-coverage-20261112').glob('*.json'))
    captures += sorted((ROOT / 'tools/fixtures/four-error-20261112').glob('*.json'))
    for path in captures:
        pair = json.loads(path.read_text(encoding='utf8'))
        gap = convex_sat_gap(pair)
        expected = pair['expectedAccepted']
        # SAT is a terminal oracle only. Continuous acceptance is tested below
        # by the production polynomial and independent physical-path samples.
        if not (gap > .03 if expected else gap < -1e-5):
            raise RuntimeError(f'{path.name}: invalid physical fixture, gap={gap}')
        print(f'[NP_CAPTURE_SAT] case={path.stem} signedGap={gap:.12g} status=PASS', flush=True)
        numbers = [int(expected), int(pair['fixedAxis']), *pair['normal'], pair['rest']]
        for side in pair['sides']:
            numbers += [side['radius'], len(side['points']), *side['start'], *side['end']]
            for point in side['points']:
                numbers += point
        target = directory / (path.stem + '.txt')
        target.write_text(' '.join(str(v) for v in numbers), encoding='ascii')
        result.append(str(target))
    if len(result) != 6:
        raise RuntimeError('missing physical capture fixture')
    path = ROOT / 'tools/fixtures/contact53-20261112/near-touch-axis.json'
    pair = json.loads(path.read_text(encoding='utf8'))
    gap = convex_sat_gap(pair)
    if not 1.e-10 < gap < 1.e-8:
        raise RuntimeError(f'{path.name}: invalid near-touch capture, gap={gap}')
    # The initial float NP direction fails; the trailing double direction
    # seeds a new complete query, never an unconditional cached admission.
    numbers = [0, 0, *pair['normal'], pair['rest']]
    for side in pair['sides']:
        numbers += [side['radius'], len(side['points']), *side['start'], *side['end']]
        for point in side['points']:
            numbers += point
    numbers += pair['proofAxis']
    target = directory / (path.stem + '.txt')
    target.write_text(' '.join(str(v) for v in numbers), encoding='ascii')
    result.append(str(target))
    print(f'[NP_CAPTURE_SAT] case={path.stem} signedGap={gap:.12g} status=PASS', flush=True)
    return result


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='avbd_np_pairs_') as work:
        paths = fixtures(Path(work))
        for mode in ('precise', 'fast'):
            print('NP normal factorization /fp:' + mode, flush=True)
            code = run_probe(ROOT / 'tools/avbd_np_factorization_probe.cpp', args=paths,
                             compiler_options=('/fp:' + mode,),
                             extra_sources=[ROOT / 'physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp'])
            if code:
                raise SystemExit(code)
