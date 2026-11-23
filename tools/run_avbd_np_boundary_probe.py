"""Prescribed boundary motion must reach its target without invalidating NP admission."""
import os
from run_avbd_ogc_certificate_probe import ROOT, run_probe
if __name__ == "__main__":
    os.environ["PHYSX_NP_GEOMETRY_CERTIFICATES"] = "2"
    raise SystemExit(run_probe(ROOT / "tools/avbd_np_boundary_probe.cpp",
        libraries=("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
                   "PhysX_64.lib", "PhysXCommon_64.lib", "PhysXCooking_64.lib")))
