#!/usr/bin/env python3
"""Compare SnippetJoint's fixed-chain fracture momentum with TGS, headlessly."""
import argparse
import csv
import hashlib
import io
import os
import subprocess
import tempfile
import json
import math
from pathlib import Path

from run_avbd_ogc_certificate_probe import ROOT, build_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, default=ROOT / 'physx/bin/win.x86_64.vc143.md/checked')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--include-reference', action='store_true')
    args = parser.parse_args()
    runtime = args.bin_dir.resolve()
    with tempfile.TemporaryDirectory(prefix='avbd_fracture_') as directory:
        work = Path(directory)
        code = build_probe(work, runtime, ROOT / 'tools/avbd_fixed_chain_impact_probe.cpp',
                           libraries=['PhysXExtensions_static_64.lib', 'PhysX_64.lib',
                                      'PhysXCommon_64.lib', 'PhysXCooking_64.lib', 'PhysXPvdSDK_static_64.lib'])
        if code:
            return code
        return run_matrix(args, runtime, work / 'probe.exe')


def run_matrix(args, runtime, exe):
    files = [exe] + sorted(runtime.glob('*.dll'))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.output_root.mkdir(parents=True, exist_ok=True)
    lanes = ['avbd']
    if args.include_reference:
        lanes.append('tgs')
    results = []
    cases = [(f'{speed}-{gravity}-{name}', speed, gravity, offset, slope, 1000, 100000, 60)
             for speed in (20, 200) for gravity in (0, -9.81)
             for name, offset, slope in (('center', 0, 0), ('offset', .25, 0), ('oblique', .25, .3))]
    cases += [(f'unbreakable-{speed}', speed, 0, .25, 0, 1e12, 1e12, 60) for speed in (20, 200)]
    cases += [(f'frequency-{hz}', 20, 0, .25, 0, 1000, 100000, hz) for hz in (30, 120)]
    cases += [('torque-only', 20, 0, .25, 0, 1e12, 50, 60)]
    for solver in lanes:
        for name, speed, gravity, offset, slope, force, torque, hz in cases:
            label = f'{solver}-{name}'
            command = [str(exe), solver, 'fixed', str(speed), '3', str(offset), str(slope),
                       str(gravity), '600', '120', str(force), str(torque), str(1/hz)]
            env = os.environ.copy()
            env['PATH'] = str(runtime) + os.pathsep + env['PATH']
            run = subprocess.run(command, cwd=exe.parent, env=env, capture_output=True,
                                 text=True, timeout=120)
            (args.output_root / f'{label}.csv').write_text(run.stdout, encoding='utf-8')
            (args.output_root / f'{label}.stderr').write_text(run.stderr, encoding='utf-8')
            errors, metrics = [], {}
            if run.returncode or run.stderr.strip():
                errors.append('process')
            try:
                rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(io.StringIO(run.stdout))]
                if len(rows) != 600 or any(r['frame'] != i+1 for i,r in enumerate(rows)):
                    errors.append('frames')
                if any(not math.isfinite(v) for r in rows for v in r.values()):
                    errors.append('nonfinite')
                hits = [i for i,r in enumerate(rows) if r['contacts']]
                breaks = [i for i,r in enumerate(rows) if r['broken']]
                if not hits: errors.append('no_impact')
                if any(int(r['broken']) != int(r['callback_mask']) or
                       bin(int(r['broken'])).count('1') != int(r['callbacks']) for r in rows):
                    errors.append('break_callback_identity')
                total = [r['chain_ke'] + r['ball_ke'] + r['chain_pe'] + r['ball_pe'] for r in rows]
                ratio = max(total[120:]) / total[120]
                metrics = dict(first_contact=hits[0]+1 if hits else None,
                               first_break=breaks[0]+1 if breaks else None,
                               final_broken_mask=int(rows[-1]['broken']),
                               peak_chain_energy=max(r['chain_ke'] for r in rows),
                               peak_mechanical_energy_ratio=ratio)
                if solver == 'avbd' and gravity:
                    # A section carrying n four-kilogram boxes has its centre
                    # of mass 2*n metres from that section's joint anchor.
                    for joint in range(5):
                        count = 5 - joint
                        load = 4 * count * abs(gravity)
                        moment = 2 * count * load
                        if any(abs(row[f'force{joint}'] / load - 1) > .01 or
                               abs(row[f'torque{joint}'] / moment - 1) > .01 for row in rows[:120]):
                            errors.append(f'static_wrench_{joint}')
                    metrics['root_static_force_N'] = rows[0]['force0']
                    metrics['root_static_torque_Nm'] = rows[0]['torque0']
                if ratio > 1.01: errors.append('energy_creation')
                if name.startswith('unbreakable'):
                    if breaks: errors.append('unexpected_fracture')
                    if solver == 'avbd' and metrics['peak_chain_energy'] > 1e-4:
                        errors.append('anchored_chain_motion')
                    if solver == 'avbd' and hits:
                        if max(r['ball_vz'] for r in rows[hits[0]:]) < .5*speed:
                            errors.append('missing_wall_restitution')
                        if max(r['force0'] for r in rows[hits[0]:]) < 1000:
                            errors.append('missing_support_reaction')
                else:
                    if not breaks: errors.append('no_fracture')
                    if breaks:
                        i = breaks[0]; r = rows[i]
                        metrics.update(fracture_energy=r['chain_ke'], fracture_max_speed=r['max_v'])
                        # A successful callback cannot hide velocity erasure on
                        # that very frame. Trajectories and break masks may differ.
                        if r['max_v'] < .1 * speed: errors.append('lost_fracture_momentum')
                        if hits and i < hits[0]: errors.append('fracture_before_impact')
                        if gravity == 0 and int(r['broken']) == 31:
                            ball_mass = 10 * 4/3 * math.pi * 3**3
                            def momentum(row):
                                return [4*sum(row[f'v{axis}{b}'] for b in range(5)) +
                                        ball_mass*row[f'ball_v{axis}'] for axis in 'xyz']
                            before, after = momentum(rows[i-1]), momentum(r)
                            relative = math.sqrt(sum((a-b)**2 for a,b in zip(after,before))) / (ball_mass*speed)
                            metrics['fracture_momentum_relative_error'] = relative
                            if solver == 'avbd' and relative > 2e-4:
                                errors.append('fracture_linear_momentum')
            except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
                errors.append(f'parse:{exc}')
            results.append(dict(name=label, command=command, returncode=run.returncode,
                                metrics=metrics, errors=errors, passed=not errors))
            print(f"[JOINT_FRACTURE] {label} {'FAIL' if errors else 'PASS'} "
                  f"speed={metrics.get('fracture_max_speed')} errors={','.join(errors) or 'none'}", flush=True)
    unchanged = manifest == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    passed = unchanged and all(r['passed'] for r in results)
    (args.output_root/'summary.json').write_text(json.dumps(dict(passed=passed,
        runtime=manifest, runtime_unchanged=unchanged, runs=results), indent=2), encoding='utf-8')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
