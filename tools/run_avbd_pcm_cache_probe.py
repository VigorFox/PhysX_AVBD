"""Replay a captured full-scene PCM cache miss through public contact generation."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    raise SystemExit(run_probe(ROOT / "tools/avbd_pcm_cache_probe.cpp",
        include_dirs=tuple(ROOT / p for p in (
            "physx/source/common/src", "physx/source/common/include",
            "physx/source/geomutils/include", "physx/source/geomutils/src",
            "physx/source/geomutils/src/pcm", "physx/source/geomutils/src/convex",
            "physx/source/geomutils/src/gjk")),
        libraries=("PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib",
                   "PhysX_64.lib", "PhysXCommon_64.lib", "PhysXCooking_64.lib")))
