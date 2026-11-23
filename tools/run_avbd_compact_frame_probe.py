"""Validate the compact production potential, endpoint curvature and metric."""
import sys
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    for mode in ('precise', 'fast'):
        print('Compact frame /fp:' + mode, flush=True)
        code = run_probe(ROOT / 'tools/avbd_compact_frame_probe.cpp',
                         args=sys.argv[1:], compiler_options=('/fp:' + mode,))
        if code:
            raise SystemExit(code)
