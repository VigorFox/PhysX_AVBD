from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    for mode in ('precise', 'fast'):
        print('Row working set /fp:' + mode, flush=True)
        code = run_probe(ROOT / 'tools/avbd_row_working_set_probe.cpp',
                         compiler_options=('/fp:' + mode,),
                         extra_sources=(ROOT / 'physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp',))
        if code:
            raise SystemExit(code)
