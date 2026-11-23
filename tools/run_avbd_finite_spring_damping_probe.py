"""Verify common finite motion and the spring damping epoch through the SDK."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("checked", "debug", "release"), default="checked")
    parser.add_argument("--large-motion", action="store_true")
    options = parser.parse_args()
    raise SystemExit(run_probe(ROOT / "tools/avbd_finite_spring_damping_probe.cpp",
        args=("--large-motion",) if options.large_motion else (), config=options.config,
        libraries=("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
                   "PhysX_64.lib", "PhysXCommon_64.lib", "PhysXCooking_64.lib")))
