"""Frozen original-equation qualification; no physical scene publication."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import random
import tempfile

from run_avbd_local_response_probe import pack
from run_avbd_ogc_certificate_probe import ROOT, run_probe
from validate_avbd_common_capture import validate

INPUTS = {
    '135': 'local-convergence-20261030/remaining-135.json.gz',
    '240': 'local-patches-20261029/certificate-failure.json.gz',
    '280': 'local-convergence-20261030/projected-stagnation-280.json.gz',
    'extra280': 'local-convergence2-20261030/trials/extra-direction-condition-280.json.gz',
    'five': 'common-program-20261028/five.json.gz',
    'reaction': 'bounded-host-20261105/reaction-failure.json.gz',
}


def permute_rows(program):
    p = copy.deepcopy(program)
    order = list(range(len(p['mu'])))
    random.Random(7183).shuffle(order)
    inverse = {old: new for new, old in enumerate(order)}
    for key in ('mu', 'coulomb', 'sourceRows'):
        p[key] = [p[key][old] for old in order]
    for key in ('offset', 'lower', 'upper', 'compliance', 'initialForce'):
        if key in p:
            p[key] = [p[key][3 * old + q] for old in order for q in range(3)]
    p['J'] = [[inverse[row[0]], *row[1:]] for row in p['J']]
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--fp', choices=('precise', 'fast', 'both'), default='both')
    args = parser.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    results = []
    for fp in (('precise', 'fast') if args.fp == 'both' else (args.fp,)):
        with tempfile.TemporaryDirectory(prefix='avbd_bounded_suite_') as directory:
            work = Path(directory)
            jobs = []
            lines = []
            for name, source in INPUTS.items():
                original = json.loads(gzip.decompress((ROOT / 'tools/fixtures' / source).read_bytes()))
                for reordered in (False, True):
                    program = permute_rows(original) if reordered else original
                    for warm in ((False, True) if 'initialForce' in program else (False,)):
                        binary = work / f'{name}-{int(reordered)}-{int(warm)}.bin'
                        pack(binary, program, warm)
                        for threads in (1, 4):
                            key = f'{fp}-{name}-perm{int(reordered)}-warm{int(warm)}-t{threads}'
                            report = (args.reports / (key + '.json')).resolve()
                            lines.extend([str(binary), str(report), 'bounded', '1', str(threads)])
                            jobs.append((key, program, report))
            manifest = work / 'cases.txt'
            manifest.write_text('\n'.join(lines) + '\n', encoding='utf8')
            code = run_probe(ROOT / 'tools/avbd_local_response_probe.cpp', args=('--batch', str(manifest)),
                             compiler_options=('/fp:' + fp,),
                             extra_sources=(ROOT / 'physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp',))
            for key, program, report in jobs:
                if not report.is_file():
                    results.append(dict(case=key, passed=False, reason='no_result', processExit=code))
                    continue
                response = json.loads(report.read_text())
                p = copy.deepcopy(program)
                p['replay'] = dict(solved=response['converged'], step=response['step'], force=response['force'],
                                   source='bounded colored AL with local force exchanges')
                p['tableIterations'] = response['iterations']
                p['materialProducts'] = response['materialProducts']
                full = work / 'validation.json'
                full.write_text(json.dumps(p), encoding='utf8')
                result = validate(full)
                result['file'] = str(report)
                result.update(case=key, sweeps=response['iterations'], executionStatus=response['executionStatus'],
                              milliseconds=response['milliseconds'])
                results.append(result)
                report.with_suffix('.validation.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    summary = dict(cases=len(results), failed=sum(not r['passed'] for r in results), results=results)
    (args.reports / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}), flush=True)
    return int(summary['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
