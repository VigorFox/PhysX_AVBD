#!/usr/bin/env python3
"""Build and finite-difference the same potential used by production rows."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe


if __name__ == "__main__":
    raise SystemExit(run_probe(ROOT / "tools/avbd_contact_potential_probe.cpp"))
