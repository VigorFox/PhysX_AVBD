"""Verify public translation locks during NP prescribed-boundary restoration."""
import os
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    os.environ["PHYSX_NP_GEOMETRY_CERTIFICATES"] = "2"
    raise SystemExit(run_probe(ROOT / "tools/avbd_np_locked_motion_probe.cpp",
        libraries=("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
                   "PhysX_64.lib", "PhysXCommon_64.lib", "PhysXCooking_64.lib")))
