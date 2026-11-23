"""CPU NP borrowed mapping and bounded sparse candidate coverage oracle."""
import sys
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    args = sys.argv[1:]
    scalar = 'double' if '--double' in args else 'float'
    fp = '/fp:fast' if '--fast' in args else '/fp:precise'
    args = [arg for arg in args if arg not in ('--double', '--fast')]
    raise SystemExit(run_probe(ROOT / 'tools/np_deformable_mesh_probe.cpp', args=args,
        include_dirs=[ROOT / 'physx/source/lowlevel/common/include/pipeline'],
        defines=['PX_NP_DEFORMABLE_REAL=' + scalar],
        compiler_options=[fp],
        libraries=['PhysXExtensions_static_64.lib', 'PhysX_64.lib', 'PhysXPvdSDK_static_64.lib', 'PhysXCommon_64.lib']))
