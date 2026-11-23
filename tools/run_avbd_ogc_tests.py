#!/usr/bin/env python3
"""Run production certificate identity, intersection, and composed-domain tests."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    raise SystemExit(run_probe(ROOT / "tools/avbd_ogc_tests.cpp"))
