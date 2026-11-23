"""Check exact frame second variations and assembled local metric protection."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    for mode in ('precise', 'fast'):
        print('Frame curvature /fp:' + mode, flush=True)
        code = run_probe(ROOT / 'tools/avbd_frame_curvature_probe.cpp', compiler_options=('/fp:' + mode,))
        if code:
            raise SystemExit(code)
