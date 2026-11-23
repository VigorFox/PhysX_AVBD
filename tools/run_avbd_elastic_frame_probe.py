"""Validate the shared native finite material and its relative PSD factor."""
from run_avbd_ogc_certificate_probe import ROOT,run_probe
if __name__=='__main__':raise SystemExit(run_probe(ROOT/'tools/avbd_elastic_frame_probe.cpp'))
