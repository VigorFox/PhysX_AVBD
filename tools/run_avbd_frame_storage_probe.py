"""Validate cooked frame sharing, parameter edits, epochs and parallel reads."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    for mode in ('precise', 'fast'):
        print('Frame storage /fp:' + mode, flush=True)
        code = run_probe(ROOT / 'tools/avbd_frame_storage_probe.cpp', compiler_options=('/fp:' + mode,))
        if code:
            raise SystemExit(code)
