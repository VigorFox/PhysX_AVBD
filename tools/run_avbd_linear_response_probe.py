"""Check the real Table bindings and nested full-material response."""
from run_avbd_ogc_certificate_probe import ROOT,run_probe
if __name__=='__main__':
    for option in ('/fp:precise','/fp:fast'):
        print('Linear response '+option,flush=True)
        code=run_probe(ROOT/'tools/avbd_linear_response_probe.cpp',
                       extra_sources=[ROOT/'physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp'],
                       compiler_options=(option,))
        if code:raise SystemExit(code)
