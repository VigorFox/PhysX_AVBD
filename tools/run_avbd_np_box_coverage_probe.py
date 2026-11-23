"""Replay the first macroscopic crossing from the reordered native five-body case."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe
if __name__ == '__main__':
    for mode in ('precise', 'fast'):
        print('Box proximity probe /fp:' + mode, flush=True)
        code = run_probe(ROOT / 'tools/avbd_np_box_coverage_probe.cpp',
            include_dirs=(ROOT / 'physx/source/geomutils/src/pcm',),
            compiler_options=('/fp:' + mode,),
            libraries=('PhysXExtensions_static_64.lib', 'PhysXPvdSDK_static_64.lib',
                       'PhysX_64.lib', 'PhysXCommon_64.lib', 'PhysXCooking_64.lib'))
        if code:
            raise SystemExit(code)
