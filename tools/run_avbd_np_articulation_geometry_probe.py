"""Check complete NP support geometry with nonidentity articulation mass frames."""
import os
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    os.environ["PHYSX_NP_GEOMETRY_CERTIFICATES"] = "2"
    os.environ["PHYSX_AVBD_SCHEDULE_STATS"] = "1"
    raise SystemExit(run_probe(ROOT / "tools/avbd_np_articulation_geometry_probe.cpp",
        libraries=("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
                   "PhysX_64.lib", "PhysXCommon_64.lib", "PhysXCooking_64.lib")))
