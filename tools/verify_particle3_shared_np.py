"""Shared NP integration checks; known physical failures remain RED."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

def main():
    sdk = Path(os.environ['AVBD_PROBE_BINARY_DIR']).resolve()
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sdk.iterdir())
                        if p.suffix.lower() in ('.dll', '.exe')}
    report = dict(scope='shared NP / mixed rows integration; no production qualification', before=identity(), checks=[])

    def run(name, command, negative=False, env=None):
        child_env = os.environ.copy()
        if env: child_env.update(env)
        with (output / (name + '.log')).open('wb') as stream:
            child = subprocess.run(list(map(str, command)), cwd=ROOT, env=child_env,
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        text = (output / (name + '.log')).read_text(encoding='utf8', errors='replace')
        passed = (child.returncode == 2 and 'step not published' in text) if negative else child.returncode == 0
        record = dict(name=name, exit=child.returncode, negativeDomainTest=negative, passed=passed)
        def fields(tag):
            return [dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)
                    for line in text.splitlines() if line.startswith(tag + ' ')]
        if name.startswith('warm-admission-'):
            epochs, checks = fields('[WARM_ORACLE]'), fields('[AVBD_WARM_CHECKS]')
            record['warmOracle'], record['warmChecks'] = epochs, checks
            parallel = name != 'warm-admission-0'
            record['passed'] &= (len(checks) == len(epochs) == 2 and
                [row['epoch'] for row in epochs] == ['0', '1'] and
                all(row['status'] == 'PASS' for row in epochs) and
                int(epochs[0]['crossing']) > 0 and int(epochs[1]['crossing']) == 0)
            for epoch, check in zip(epochs, checks):
                rows = int(epoch['faces']) if parallel else 0
                record['passed'] &= (int(check['checkedRows']) == rows and
                    int(check['masks']) == (rows + 31) // 32 and
                    int(check['failedRows']) == (int(epoch['crossing']) if parallel else 0))
        if name == 'surface-rest-revision':
            epochs, packed = fields('[WARM_ORACLE]'), fields('[AVBD_MATERIAL_PACK]')
            record['epochs'], record['materialPack'] = epochs, packed
            record['passed'] &= (len(epochs) == len(packed) == 4 and
                all(r['status'] == 'PASS' for r in epochs) and
                all(int(p['rows']) == int(e['faces']) for e,p in zip(epochs,packed)) and
                all(int(p['reusedTopologyRows']) == (int(p['rows']) if i%2 else 0)
                    for i,p in enumerate(packed)))
            if packed and 'retainedMaterials' in packed[0]:
                record['passed'] &= all(int(p['retainedSupportRows']) == int(p['reusedTopologyRows']) and
                    int(p['retainedMaterials']) <= int(p['retainedSupportRows']) for p in packed)
        if name == 'four-sheet-contact':
            gates,replicas=fields('[AVBD_GATE]'),fields('[DRAPE_REPLICA]')
            record['passed'] &= (len(gates)==1 and gates[0]['status']=='PASS' and
                gates[0]['completedFrames']=='150' and len(replicas)==3 and all(r['status']=='PASS' for r in replicas))
        if name.startswith('particle3-authored-') or name in ('rigid6-Surface', 'rigid6-Volume'):
            surface = name.lower().endswith('surface')
            default = name.startswith('particle3-')
            route_tag = '[AVBD_SURFACE_ROUTE]' if surface else '[AVBD_VOLUME_ROUTE]'
            routes, gates = fields(route_tag), fields('[AVBD_GATE]')
            expected = 'particle3' if default else ('joint_sheet' if surface else 'joint')
            record['passed'] &= (len(routes) == len(gates) == 1 and routes[0].get('model') == expected
                and gates[0].get('case') == ('static-box' if surface else 'visual-showcase')
                and gates[0].get('completedFrames') == '360' and gates[0].get('status') == 'PASS')
            if default:
                record['passed'] &= bool(routes) and routes[0].get('iterations') == '4'
            record['route'] = routes
            if name == 'rigid6-Volume':
                impacts = fields('[FIVE_REBOUND]')
                record['impacts'] = impacts
                record['passed'] &= len(gates) == 1 and len(impacts) == 1 and impacts[0].get('pair') == 'prism-sphere' and (
                    0 < int(impacts[0]['contactFrame']) <= int(gates[0]['completedFrames'])
                    and float(impacts[0]['incoming']) > 0)
        if name in ('drape-50', 'drape-settle'):
            fold = fields('[DRAPE_FOLD]')
            record['fold'] = fold
            # In this authored 60 Hz drop the hanging rim should descend at least
            # one grid spacing during the reported half-second of folding.
            if len(fold) == 1:
                spacing = float(fold[0]['gridSpacing'])
                # Limit abrupt braking over the entire first impact and fold,
                # including frames before the observed 50-66 pause.
                record['passed'] &= (int(fold[0]['samples']) == 31 and
                                     float(fold[0]['perimeterDrop']) >= spacing and
                                     float(fold[0]['earlyDrop']) >= 2 * spacing and
                                     float(fold[0]['peakBrakingTravel']) <= .5 * spacing)
            else:
                record['passed'] = False
        if name in ('resting-sphere', 'resting-cone'):
            samples = fields('[GROUND_REST]')
            record['groundRest'] = samples
            record['passed'] &= (len(samples) == 1 and samples[0]['case'] == name
                and samples[0]['frames'] == '2400' and samples[0]['status'] == 'PASS')
        if name.startswith('pedestal-tip'):
            samples=fields('[PEDESTAL_TIP]')
            record['tipping']=samples
            record['passed'] &= (len(samples)==1 and samples[0]['frames']=='240'
                and samples[0]['loaded']==('1' if name=='pedestal-tip-loaded' else '0')
                and samples[0]['status']=='PASS')
        if name in ('five-release', 'five-audit'):
            skin=fields('[FIVE_SKIN_STRETCH]')
            record['skinStretch']=skin
            record['passed'] &= (len(skin)==5 and {r['body'] for r in skin}==set('01234') and
                all(float(r['longestDoubledS'])<.5 for r in skin))
            tipping=fields('[FIVE_TIP]')
            record['tipping']=tipping
            record['passed'] &= (len(tipping)==1 and int(tipping[0]['impactFrame'])>0 and
                all(0<int(tipping[0][event])-int(tipping[0]['impactFrame'])<=120
                    for event in ('postImpact15Frame','floorBandFrame')))
            release=fields('[FIVE_RELEASE]')
            record['passed'] &= len(release)==1 and int(release[0]['firstGroundedSeparation'])>0
            residence=fields('[FIVE_CONTACT_RESIDENCE]')
            record['residence']=residence
            # Two seconds of uninterrupted proximity before landing is the
            # acceptance limit for this oblique drop at 60 Hz.
            record['passed'] &= (len(residence)==1 and 0<int(residence[0]['longestAirborneFrames'])<=120
                and 0<int(residence[0]['sustainedExit'])-int(residence[0]['firstEntry'])<=120)
            pedestal=fields('[FIVE_PEDESTAL]')
            record['pedestal'] = pedestal
            # The off-centre pedestal must allow unloading and motion of the
            # sandwich. While isolated its centre of mass follows gravity to
            # within one frame of gravitational acceleration.
            record['passed'] &= len(pedestal)==1 and int(pedestal[0]['contactFrame'])>0 and (
                float(pedestal[0]['flightDvError'])<9.81/60 and
                float(pedestal[0]['unloadedLowerVolume'])-float(pedestal[0]['minLowerVolume'])>.01 and
                pedestal[0]['status']=='PASS' and min(float(pedestal[0][key])
                    for key in ('lowerRise','boxUnloadingTravel','upperUnloadingTravel'))>.02)
            impacts=fields('[FIVE_REBOUND]')
            record['impacts'] = impacts
            record['passed'] &= {r['pair'] for r in impacts} == {'cone-ground', 'prism-sphere'}
            for r in impacts:
                # A late roll off the sphere is a different event from recoil.
                # The showcase runs at 60 Hz. Observe the first second after impact.
                events=['apexFrame'] if r['pair']=='cone-ground' else ['apexFrame','departureFrame']
                record['passed'] &= (all(0 < int(r[e]) - int(r['contactFrame']) <= 60 for e in events)
                    and float(r['rise']) > 0
                    and float(r['up']) > .1 * float(r['incoming'])
                    and float(r['recoveredVolume']) > float(r['minVolume']))
                if r['pair'] == 'cone-ground':
                    record['passed'] &= float(r['clearance']) > .02 and float(r['rise']) > .1
        if name in ('self-near','self-far'):
            entries = [dict(t.split('=',1) for t in line.split()[1:] if '=' in t)
                       for line in text.splitlines() if line.startswith('[NP_DEFORMABLE_PACKETS] ')]
            counts = [int(e['contacts']) for e in entries if e['valid']=='1' and e['pairs']=='1']
            record['selfContacts'] = counts
            record['passed'] &= bool(counts) and max(counts)==int(name=='self-far') and 'step not published' not in text
        if name.startswith('wide-star-'):
            tiles = [dict(t.split('=',1) for t in line.split()[1:] if '=' in t)
                     for line in text.splitlines() if line.startswith('[AVBD_LOCAL_TILES] ')]
            record['maxTiledValence'] = max((int(t['maxIncidentRows']) for t in tiles), default=0)
            record['passed'] &= len(tiles)==12 and '[WIDE_STAR]' in text and 'status=PASS' in text
            if name=='wide-star-bending':
                record['passed'] &= record['maxTiledValence']>256
        if name.endswith('-audit'):
            def rows(prefix):
                return [dict(token.split('=', 1) for token in line.split()[1:] if '=' in token)
                        for line in text.splitlines() if line.startswith(prefix + ' ')]
            steps, blocks, budgets, packets = (rows(p) for p in
                ('[AVBD_COMMON_FRAME]', '[AVBD_BLOCK_DOF]', '[AVBD_LOCAL_BUDGET]', '[NP_DEFORMABLE_PACKETS]'))
            audit = bool(steps and blocks and budgets and packets) and len(blocks) == len(budgets)
            audit &= all(r.get('iterations') == '4' and r.get('bdf') == '1' and
                         r.get('poseForcePublications') == '1' for r in steps)
            audit &= all(int(r['peakPacketsPerTask']) <= 16 and r['packetLimit'] == '16' and
                         r['scalarBytes'] == '4' and r['valid'] == '1' for r in packets)
            audit &= any(int(r['mixedContactRows']) > 0 for r in blocks)
            if name in ('drape-audit','five-audit','five-inline-audit'):
                dispatch = rows('[AVBD_LOCAL_DISPATCH]')
                audit &= len(steps)==len(budgets)==len(dispatch)==360
                audit &= all(int(r['particle3'])>0 and int(r['nodalMaterialRows'])>0 for r in blocks)
                audit &= all(int(r['budget'])==4 and int(r['iterations'])==4 and
                             r['status']=='ACCEPTED' for r in budgets)
                audit &= all(int(r['primalMaxBlocks'])<=32 and int(r['primalMaxRows'])<=256 and
                             int(r['dualMaxRows'])<=256 and int(r['queryMaxRows'])<=32 for r in dispatch)
                tiles = rows('[AVBD_LOCAL_TILES]')
                audit &= len(tiles) == 360 and all(
                    int(t['maxIncidentRows']) == 0 or
                    (int(t['leaves']) >= 2 * int(t['blocks']) and int(t['rowLimit']) == 256
                     and int(t['reductionInputs']) == 2) for t in tiles)
                record['maxTiledValence'] = max((int(t['maxIncidentRows']) for t in tiles), default=0)
                record['tiledFrames'] = sum(int(t['blocks']) > 0 for t in tiles)
            if name == 'cloth-impact-audit':
                audit &= len(steps)==len(blocks)==180 and all(
                    int(r['nodalMaterialRows'])==1152 and int(r['membranePrimal'])==4*1152 and
                    int(r['contactPrimal'])==4*int(r['mixedContactRows']) for r in blocks)
                dispatch = rows('[AVBD_LOCAL_DISPATCH]')
                audit &= len(dispatch)==180 and all(int(r['primalMaxBlocks'])<=32 and
                    int(r['primalMaxRows'])<=256 and int(r['dualMaxRows'])<=256 and
                    int(r['queryMaxRows'])<=32 for r in dispatch)
                audit &= all(int(b['dualRows'])==1152+4*int(r['mixedContactRows'])
                             for b,r in zip(budgets,blocks))
            if name == 'soft-soft-audit':
                audit &= any(int(r['pairedUpdates']) > 0 for r in blocks)
                audit &= all(int(r['nominationMaxRows']) <= 256 and
                             int(r['nodalSolves']) == 4*int(r['particle3']) for r in blocks)
            audit &= all(int(b['queryRows']) >= int(d['mixedContactRows'])
                         for d, b in zip(blocks, budgets))
            record['executionAuditPassed'] = audit
            record['passed'] &= audit
        report['checks'].append(record)
        print(record, flush=True)
        (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')

    for case in ('self-near','self-far'):
        run(case,[sys.executable,'tools/run_np_deformable_surface_input_probe.py',case],
            env=dict(PHYSX_NP_DEFORMABLE_STATS='1'))
    for suffix, flags in [('f32', []), ('f32-fast', ['--fast']), ('f64', ['--double'])]:
        run('domain-' + suffix, [sys.executable, 'tools/run_avbd_nodal_contact_domain_probe.py'] + flags)
        run('hinge-simd-' + suffix, [sys.executable, 'tools/run_avbd_hinge_simd_probe.py'] + flags)
        run('membrane-simd-' + suffix, [sys.executable, 'tools/run_avbd_membrane_simd_probe.py'] + flags)
    run('row-program', [sys.executable, 'tools/run_avbd_row_program_probe.py'])
    run('surface-rest-revision', [sys.executable,'tools/run_avbd_nodal_cloth_contact_probe.py',
        '--warm-admission',8,'rest-revision'], env=dict(PHYSX_AVBD_COMMON_STATS='1',PHYSX_AVBD_LOCAL_PROFILE='1'))
    run('fixed-chain-impact', [sys.executable, 'tools/run_avbd_fixed_chain_impact.py',
        '--bin-dir', sdk, '--include-reference', '--output-root', output/'fixed-chain'])
    for workers in (0, 4, 8):
        run(f'warm-admission-{workers}', [sys.executable, 'tools/run_avbd_nodal_cloth_contact_probe.py',
            '--warm-admission', workers], env=dict(PHYSX_AVBD_COMMON_STEP='budget',
                PHYSX_AVBD_LOCAL_ITERATIONS='4', PHYSX_AVBD_COMMON_STATS='1', PHYSX_AVBD_LOCAL_PROFILE='1'))
        for endpoint in ('vertex','tet'):
            run(f'attachment-{endpoint}-{workers}',[sys.executable,'tools/run_avbd_nodal_volume_step_probe.py',
                workers,'attachment-'+endpoint])
        run(f'mesh-{workers}', [sys.executable, 'tools/run_np_deformable_mesh_probe.py', workers, 24])
        run(f'plane-{workers}', [sys.executable, 'tools/run_avbd_nodal_volume_step_probe.py', workers, 'plane-contact'])
        run(f'cloth-{workers}', [sys.executable, 'tools/run_avbd_nodal_cloth_contact_probe.py', workers])
        run(f'cloth-impact-particle3-{workers}', [sys.executable, 'tools/run_avbd_nodal_release_probe.py', workers, 'impact'])
        run(f'cloth-impact-rigid6-{workers}', [sdk / 'SnippetDeformableSurfaceAVBD_64.exe',
            '--rigid6', '--headless', '--case=reaction', '--frames=180', f'--dispatcher-threads={workers}'])
        run(f'cloth-release-particle3-{workers}', [sys.executable, 'tools/run_avbd_nodal_release_probe.py', workers])
        run(f'cloth-release-rigid6-{workers}', [sdk / 'SnippetDeformableSurfaceAVBD_64.exe',
            '--rigid6', '--headless', '--case=strain-release', '--frames=180', f'--dispatcher-threads={workers}'])
        run(f'soft-soft-{workers}', [sys.executable, 'tools/run_avbd_nodal_soft_contact_probe.py', workers])
    run('soft-contact-translating',[sys.executable,'tools/run_avbd_nodal_soft_contact_probe.py',4,0,0,2])
    run('soft-contact-relative-slide',[sys.executable,'tools/run_avbd_nodal_soft_contact_probe.py',4,0,0,0,1,1])
    run('volume-plane-slide',[sys.executable,'tools/run_avbd_nodal_soft_contact_probe.py',4,'plane',.1])
    run('drape-settle',[sdk/'SnippetDeformableSurfaceAVBD_64.exe','--headless','--frames=3600'])
    run('four-sheet-contact',[sdk/'SnippetDeformableSurfaceAVBD_64.exe','--headless',
        '--case=static-box-four-sheets','--frames=150','--dispatcher-threads=8'])
    for resolution in (25,50):
        run(f'drape-{resolution}',[sdk/'SnippetDeformableSurfaceAVBD_64.exe','--headless',
            f'--case=static-box-{resolution}','--frames=360','--dispatcher-threads=8'])
    run('five-release',[sdk/'SnippetDeformableVolumeAVBD_64.exe','--headless','--frames=720'])
    for case in ('pedestal-tip','pedestal-tip-loaded'):
        run(case,[sdk/'SnippetDeformableVolumeAVBD_64.exe','--headless','--case='+case,
                  '--frames=240','--dispatcher-threads=4'])
    for case in ('resting-sphere', 'resting-cone'):
        run(case, [sdk/'SnippetDeformableVolumeAVBD_64.exe', '--headless',
            '--case=' + case, '--frames=2400', '--dispatcher-threads=4'])
    run('volume-flight', [sys.executable, 'tools/run_avbd_nodal_volume_step_probe.py', 4, 'flight'])
    run('volume-release', [sys.executable, 'tools/run_avbd_nodal_volume_step_probe.py', 4, 'strain'])
    run('volume-compression-release', [sys.executable, 'tools/run_avbd_nodal_volume_step_probe.py', 4, 'compression-release'])
    run('soft-contact-rebound', [sys.executable, 'tools/run_avbd_nodal_soft_contact_probe.py', 4, 0, 0, 0, 4])
    for name, arguments in [('reset',['reset']),('eccentric',['impact','-.23','.17']),
                             ('frictionless',['impact-frictionless'])]:
        run('cloth-'+name, [sys.executable,'tools/run_avbd_nodal_release_probe.py',4]+arguments)
    run('cloth-impact-audit',[sys.executable,'tools/run_avbd_nodal_release_probe.py',8,'impact'],
        env=dict(PHYSX_AVBD_COMMON_STATS='1',PHYSX_NP_DEFORMABLE_STATS='1'))
    # Shift the initial height by a fraction of the prescribed 1 m/s travel
    # per frame. A favorable color order at a single impact time is insufficient.
    for label, offset in [('early-half', '-0.008333333'), ('late-half', '0.008333333'),
                          ('late-full', '0.016666667')]:
        for model in ('rigid6', 'particle3'):
            run(f'reaction-{model}-{label}', [sys.executable, 'tools/run_avbd_mixed_reaction_probe.py',
                                             model, 4, offset])
        run('soft-soft-' + label, [sys.executable, 'tools/run_avbd_nodal_soft_contact_probe.py', 4, offset])
    for label, offset in [('eccentric', .25), ('mirror', -.25)]:
        for model in ('rigid6', 'particle3', 'rigid6-soft'):
            run(label + '-' + model, [sys.executable, 'tools/run_avbd_mixed_reaction_probe.py', model, 4, 0, offset])
        run(label + '-particle3-soft', [sys.executable, 'tools/run_avbd_nodal_soft_contact_probe.py', 4, 0, offset])
    for kind in ('plane', 'sphere'):
        run('initial-overlap-' + kind, [sys.executable, 'tools/run_avbd_nodal_volume_step_probe.py', 4, kind + '-rejection'], negative=True)
    run('reaction-audit', [sys.executable, 'tools/run_avbd_mixed_reaction_probe.py', 'particle3', 4],
        env=dict(PHYSX_AVBD_COMMON_STATS='1', PHYSX_NP_DEFORMABLE_STATS='1', PHYSX_AVBD_MOMENTUM_TRACE='1'))
    run('cloth-audit', [sys.executable, 'tools/run_avbd_nodal_cloth_contact_probe.py', 8],
        env=dict(PHYSX_AVBD_COMMON_STATS='1', PHYSX_NP_DEFORMABLE_STATS='1'))
    run('soft-soft-audit', [sys.executable, 'tools/run_avbd_nodal_soft_contact_probe.py', 8],
        env=dict(PHYSX_AVBD_COMMON_STATS='1', PHYSX_NP_DEFORMABLE_STATS='1'))
    run('shell-work', [sys.executable, 'tools/run_avbd_nodal_shell_probe.py'])
    run('volume-work', [sys.executable, 'tools/run_avbd_nodal_volume_probe.py'])
    run('five-observer', [sys.executable, 'tools/run_particle3_showcase_observer_probe.py'])
    for solver in ('avbd', 'pgs', 'tgs'):
        run('hello-' + solver, [sys.executable, 'tools/run_snippet_hello_world_headless.py', '--bin-dir', sdk, '--solver', solver, '--cases', 'gated'])
        for family, runner in [('joint', 'run_snippet_joint_headless.py'), ('articulation', 'run_snippet_articulation_rc_headless.py')]:
            run(family + '-' + solver, [sys.executable, 'tools/' + runner, '--bin-dir', sdk, '--solver', solver, '--cases', 'gated'])
    for case in ('membrane', 'bending', 'free-flight'):
        run('material-' + case, [sdk / 'SnippetDeformableSurfaceAVBD_64.exe', '--particle3', '--headless', '--case=' + case, '--frames=180', '--dispatcher-threads=4'])
    for model, case in [('Surface', 'static-box'), ('Volume', 'visual-showcase')]:
        run('rigid6-' + model, [sdk / f'SnippetDeformable{model}AVBD_64.exe', '--rigid6', '--headless', '--case=' + case, '--frames=360', '--dispatcher-threads=4'])
    run('particle3-authored-surface', [sdk / 'SnippetDeformableSurfaceAVBD_64.exe',
        '--headless'])
    run('particle3-authored-volume', [sdk / 'SnippetDeformableVolumeAVBD_64.exe',
        '--headless'])
    for name,suffix,case in [('drape-audit','Surface','static-box'),('five-audit','Volume','visual-showcase')]:
        run(name,[sdk / f'SnippetDeformable{suffix}AVBD_64.exe', '--particle3','--headless',
            '--case='+case,'--frames=360','--dispatcher-threads=8'],
            env=dict(PHYSX_AVBD_COMMON_STATS='1', PHYSX_NP_DEFORMABLE_STATS='1'))
    for material in ('bending','membrane'):
        run('wide-star-'+material, [sys.executable, 'tools/run_avbd_nodal_cloth_contact_probe.py',
            '--wide-star',material,8], env=dict(PHYSX_AVBD_COMMON_STATS='1'))
    run('five-inline-audit', [sdk / 'SnippetDeformableVolumeAVBD_64.exe', '--particle3', '--headless',
        '--case=visual-showcase','--frames=360','--dispatcher-threads=0'],
        env=dict(PHYSX_AVBD_COMMON_STATS='1', PHYSX_NP_DEFORMABLE_STATS='1'))
    run('smoke', [sdk / 'avbd_smoke.exe'])
    report['after'] = identity()
    report['binariesUnchanged'] = report['before'] == report['after']
    report['allPassed'] = all(c['passed'] for c in report['checks']) and report['binariesUnchanged']
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    return 0 if report['allPassed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
