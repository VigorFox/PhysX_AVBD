from run_avbd_ogc_certificate_probe import ROOT,run_probe
if __name__=='__main__':
    for mode in ('/fp:precise','/fp:fast'):
        print(mode,flush=True)
        code=run_probe(ROOT/'tools/avbd_cone_qp_probe.cpp',compiler_options=(mode,))
        if code:raise SystemExit(code)
