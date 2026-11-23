from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    for option in ('/fp:precise', '/fp:fast'):
        print('Common program ' + option, flush=True)
        code = run_probe(ROOT / 'tools/avbd_common_program_probe.cpp', compiler_options=(option,),
                         extra_sources=[ROOT / 'physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp'])
        if code:
            raise SystemExit(code)
