"""Validate the production four-lane membrane and its candidate polynomial."""
import sys
from pathlib import Path
from run_avbd_ogc_certificate_probe import ROOT, run_probe
if __name__ == '__main__':
    inputs = [str(Path(a.split('=', 1)[1]).resolve()) for a in sys.argv[1:] if a.startswith('--replay=')]
    if inputs and '--double' in sys.argv:
        raise SystemExit('The captured production replay uses f32 inputs.')
    raise SystemExit(run_probe(ROOT / 'tools/avbd_membrane_simd_probe.cpp', args=inputs,
        defines=['PX_AVBD_NODAL_REAL=' + ('double' if '--double' in sys.argv else 'float')],
        compiler_options=['/fp:fast' if '--fast' in sys.argv else '/fp:precise']))
