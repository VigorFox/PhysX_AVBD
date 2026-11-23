"""Complete mixed geometry certificate, f32 and optional f64/fast-math builds."""
import sys
from run_avbd_ogc_certificate_probe import ROOT, run_probe
if __name__ == '__main__':
    scalar = 'double' if '--double' in sys.argv else 'float'
    raise SystemExit(run_probe(ROOT / 'tools/avbd_nodal_contact_domain_probe.cpp',
        include_dirs=[ROOT / 'physx/source/lowlevel/common/include/pipeline',
                      ROOT / 'physx/source/lowlevel/software/include',
                      ROOT / 'physx/source/geomutils/include'],
        libraries=['PhysXCommon_64.lib'],
        defines=['PX_AVBD_NODAL_REAL=' + scalar, 'PX_NP_DEFORMABLE_REAL=' + scalar],
        compiler_options=['/fp:fast' if '--fast' in sys.argv else '/fp:precise']))
