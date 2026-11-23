"""Replay a no-contact pair from the native five-body trial through public NP."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe
if __name__ == "__main__":
    raise SystemExit(run_probe(ROOT / "tools/avbd_np_missed_pair_probe.cpp",
        libraries=("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
                   "PhysX_64.lib", "PhysXCommon_64.lib", "PhysXCooking_64.lib")))
