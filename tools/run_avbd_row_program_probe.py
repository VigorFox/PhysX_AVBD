"""Validate the production multi-support incidence, components and coloring."""
import os

from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    probe = ROOT / "tools/avbd_row_program_probe.cpp"
    os.environ["PHYSX_AVBD_MATERIAL_STATS"] = "1"
    for isa in ("sse2", "auto"):
        os.environ["PHYSX_AVBD_CPU_ISA"] = isa
        result = run_probe(probe, libraries=("LowLevelDynamics_static_64.lib",))
        if result:
            raise SystemExit(result)
