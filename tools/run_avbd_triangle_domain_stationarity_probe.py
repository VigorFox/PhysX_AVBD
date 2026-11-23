#!/usr/bin/env python3
"""Reproduce coordinate stationarity without joint stationarity; not physics PASS."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe
import sys

if __name__ == "__main__":
    raise SystemExit(run_probe(
        ROOT / "tools/avbd_triangle_domain_stationarity_probe.cpp",
        args=sys.argv[1:],
        extra_sources=[ROOT / "physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp"]))
