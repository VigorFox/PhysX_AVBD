"""Nodal volume derivatives, exact 3x3 Hessian and CPU Table dispatch."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    for scalar in ('float', 'double'):
        for mode in ('/fp:precise', '/fp:fast'):
            result = run_probe(
                ROOT / 'tools/avbd_nodal_volume_probe.cpp',
                extra_sources=[ROOT / 'physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp'],
                defines=['PX_AVBD_NODAL_REAL=' + scalar], compiler_options=(mode,))
            if result:
                raise SystemExit(result)
