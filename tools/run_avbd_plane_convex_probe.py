"""Verify both faces of an edge-facing convex through public PhysX NP."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    raise SystemExit(run_probe(ROOT / "tools/avbd_plane_convex_probe.cpp",
        libraries=("PhysX_64.lib", "PhysXCommon_64.lib", "PhysXCooking_64.lib")))
