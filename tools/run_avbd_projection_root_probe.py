from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    for mode in ('precise', 'fast'):
        print('Projection root /fp:' + mode, flush=True)
        code = run_probe(ROOT / 'tools/avbd_projection_root_probe.cpp', compiler_options=('/fp:' + mode,))
        if code:
            raise SystemExit(code)
