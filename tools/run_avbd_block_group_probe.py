"""Validate the production group Table against independently assembled full H."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    for option in ('/fp:precise', '/fp:fast'):
        print('Block group ' + option, flush=True)
        code = run_probe(ROOT / 'tools/avbd_block_group_probe.cpp',
                         extra_sources=[ROOT / 'physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp'],
                         compiler_options=(option,))
        if code:
            raise SystemExit(code)
