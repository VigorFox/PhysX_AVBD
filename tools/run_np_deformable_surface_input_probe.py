"""Verify Surface public API -> shared NP geometry; response must still reject."""
import sys
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    raise SystemExit(run_probe(ROOT / 'tools/np_deformable_surface_input_probe.cpp', args=sys.argv[1:],
        libraries=['PhysXExtensions_static_64.lib', 'PhysXPvdSDK_static_64.lib',
                   'PhysXCooking_64.lib', 'PhysX_64.lib', 'PhysXCommon_64.lib']))
