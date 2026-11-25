"""Compare point-contact and closed-loop kernels with dense references, then exercise concurrent loops."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    result = run_probe(ROOT / "tools/avbd_point_response_probe.cpp")
    if result:
        raise SystemExit(result)
    result = run_probe(ROOT / "tools/avbd_rigid_loop_probe.cpp", libraries=("PhysXExtensions_static_64.lib",))
    if result:
        raise SystemExit(result)
    raise SystemExit(run_probe(ROOT / "tools/avbd_rigid_loop_dispatch_probe.cpp",
                               libraries=("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
                                          "PhysX_64.lib", "PhysXCommon_64.lib", "PhysXCooking_64.lib")))
