#!/usr/bin/env python3
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    raise SystemExit(run_probe(ROOT / "tools/avbd_schedule_lifecycle_probe.cpp",
                             libraries=("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
                                        "PhysX_64.lib", "PhysXCommon_64.lib", "PhysXCooking_64.lib")))
