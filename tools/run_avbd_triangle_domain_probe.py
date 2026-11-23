#!/usr/bin/env python3
"""Verify the production shared-domain sampling and scalar table binding."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    raise SystemExit(run_probe(ROOT / "tools/avbd_triangle_domain_probe.cpp", extra_sources=[
        ROOT / "physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp"]))
