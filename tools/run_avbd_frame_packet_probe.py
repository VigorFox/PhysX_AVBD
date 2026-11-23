"""Validate bounded frame packets, packed endpoints and parallel destinations."""
import sys
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    for option in ('/fp:precise', '/fp:fast'):
        print('Frame packet ' + option, flush=True)
        code = run_probe(ROOT / 'tools/avbd_frame_packet_probe.cpp', args=sys.argv[1:],
                         extra_sources=[ROOT / 'physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp'],
                         compiler_options=(option,))
        if code:
            raise SystemExit(code)
