"""Public Host -> unified four sweeps -> BDF, including pre-first-step strain."""
import sys
from run_avbd_ogc_certificate_probe import ROOT, run_probe
if __name__ == '__main__':
    raise SystemExit(run_probe(ROOT / 'tools/avbd_nodal_volume_step_probe.cpp', args=sys.argv[1:],
                              libraries=['PhysXExtensions_static_64.lib','PhysXPvdSDK_static_64.lib',
                                         'PhysXCooking_64.lib','PhysX_64.lib','PhysXCommon_64.lib']))
