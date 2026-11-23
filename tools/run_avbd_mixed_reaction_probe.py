"""Same-machine external reaction comparison; failures remain failures."""
import sys
from run_avbd_ogc_certificate_probe import ROOT, run_probe
if __name__ == '__main__':
    raise SystemExit(run_probe(ROOT / 'tools/avbd_mixed_reaction_probe.cpp', args=sys.argv[1:],
        libraries=['PhysXExtensions_static_64.lib','PhysXPvdSDK_static_64.lib',
                   'PhysXCooking_64.lib','PhysX_64.lib','PhysXCommon_64.lib']))
