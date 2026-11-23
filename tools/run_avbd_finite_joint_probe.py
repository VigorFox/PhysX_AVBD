"""Check native finite-joint strain derivatives, common motion and half turns."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inversion-domain", action="store_true",
                        help="show the native finite-energy inversion counterexample (currently FAIL)")
    options = parser.parse_args()
    raise SystemExit(run_probe(ROOT / "tools/avbd_finite_joint_probe.cpp",
        args=("--inversion-domain",) if options.inversion_domain else (),
        libraries=("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
                   "PhysX_64.lib", "PhysXCommon_64.lib", "PhysXCooking_64.lib")))
