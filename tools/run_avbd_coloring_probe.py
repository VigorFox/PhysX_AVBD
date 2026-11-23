#!/usr/bin/env python3
"""Check dependency coverage; color IDs and runtime trajectories may differ."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    raise SystemExit(run_probe(ROOT / "tools/avbd_coloring_probe.cpp"))
