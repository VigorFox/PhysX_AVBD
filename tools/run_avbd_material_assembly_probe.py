"""Check production material assembly against independent full support matrices."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    for option in ('/fp:precise', '/fp:fast'):
        print('Material assembly ' + option, flush=True)
        code = run_probe(ROOT / 'tools/avbd_material_assembly_probe.cpp',
                         extra_sources=[ROOT / 'physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp'],
                         compiler_options=(option,))
        if code:
            raise SystemExit(code)
