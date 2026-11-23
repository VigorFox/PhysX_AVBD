"""Solve actual SDK coefficients with the shared mixed NCP; independent dense oracle.

No candidate from this tool is published to PhysX. Small captured scenes only.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
lab = ROOT / 'experiments/avbd_contact_lab'
if not (lab / 'split.py').is_file():
    raise SystemExit(f"Offline replay requires the local AVBD contact lab at {lab}")
sys.path.insert(0, str(lab))
import split
from run_mixed_coulomb import residual
from validate_avbd_common_capture import read_capture


def replay(path):
    data = read_capture(path)
    assert data['complete'], (path, data['reason'])
    dims = data['dimensions']
    assert dims and all(d == 6 for d in dims), 'Replay adapter currently requires Rigid6'
    n, m = 6 * len(dims), len(data['mu'])
    assert n <= 600, 'Use sparse validation for larger captures'
    H = np.zeros((n, n)); J = np.zeros((m, 3, n)); blocks = [{} for _ in dims]
    for row, col, *values in data['H']:
        B = np.array(values).reshape(6, 6)
        H[6*row:6*row+6, 6*col:6*col+6] += B
        blocks[row][col] = blocks[row].get(col, np.zeros((6, 6))) + B
    for row, col, *values in data['J']:
        J[row, :, 6*col:6*col+6] += np.array(values).reshape(3, 6)
    base = np.array(data['base']); o = np.array(data['offset']).reshape(m, 3)
    lo = np.array(data['lower'], dtype=float).reshape(m, 3); hi = np.array(data['upper'], dtype=float).reshape(m, 3)
    C = np.array(data['compliance']).reshape(m, 3)
    mu = np.array(data['mu']); kinds = np.array(data['coulomb'])
    symmetry = float(np.max(abs(H-H.T)))
    eigen = float(np.linalg.eigvalsh(H)[0])
    assert symmetry < 1e-9 and eigen > 0, (symmetry, eigen)
    p = split.Program(blocks, J, o, base)
    x, f, stats = p.solve_mixed_coulomb(mu, kinds, lo, hi, C)
    error, station = residual(H, J, o, base, mu, kinds, lo, hi, C, x, f)
    tx=np.array(data['tableStep']);tf=np.array(data['tableForce']).reshape(m,3)
    table_error,table_station=residual(H,J,o,base,mu,kinds,lo,hi,C,tx,tf)
    coupled = float(np.max([np.max(abs(B)) for r, row in enumerate(blocks) for c, B in row.items() if c != r], initial=0))
    # Negative control: solving the diagonal material approximation must be
    # checked against the ORIGINAL operator, not its own easier equation.
    diagonal = split.Program([{r:row[r]} for r,row in enumerate(blocks)], J, o, base)
    dx, df, ds = diagonal.solve_mixed_coulomb(mu, kinds, lo, hi, C)
    _, diagonal_station = residual(H, J, o, base, mu, kinds, lo, hi, C, dx, df)
    return dict(file=str(path), passed=bool(stats['converged'] and error<1e-10 and station<1e-7 and
                data['tableSolved'] and table_error<1e-10 and table_station<1e-7),
                dofs=n,groups=m,sourceRows=data['sourceRowCount'],symmetry=symmetry,minEigenvalue=eigen,
                crossBodyH=coupled,originalNcpResidual=error,originalStationarity=station,
                tableNcpResidual=table_error,tableStationarity=table_station,
                tableMaterialProducts=data['materialProducts'],tableIterations=data['tableIterations'],
                tableStepDifference=float(np.max(abs(tx-x))),tableForceDifference=float(np.max(abs(tf-f))),
                diagonalNegativeControlStationarity=diagonal_station,
                boundsUpper=[float(v) if np.isfinite(v) else 'inf' for v in hi[:,0]],
                compliance=C[:,0].tolist(),force=f.tolist(),step=x.tolist(),
                execution=stats,publication='none')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('captures',nargs='+',type=Path)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    results=[replay(path) for path in args.captures]
    result=dict(passed=all(r['passed'] for r in results),results=results,provenance=split.provenance())
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
