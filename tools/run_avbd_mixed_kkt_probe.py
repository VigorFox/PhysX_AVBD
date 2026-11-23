"""Compile the actual mixed KKT Table entry with both arithmetic modes."""
from run_avbd_ogc_certificate_probe import ROOT,run_probe
if __name__=='__main__':
    for option in ('/fp:precise','/fp:fast'):
        print('Mixed KKT '+option,flush=True)
        code=run_probe(ROOT/'tools/avbd_mixed_kkt_probe.cpp',
            extra_sources=[ROOT/'physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp'],compiler_options=(option,))
        if code:raise SystemExit(code)
