"""Validate compact row storage through the production container."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    raise SystemExit(run_probe(ROOT / 'tools/avbd_row_storage_probe.cpp'))
