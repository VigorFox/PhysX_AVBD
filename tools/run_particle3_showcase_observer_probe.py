"""Check the independent five-body penetration observer on known overlaps."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    raise SystemExit(run_probe(ROOT / 'tools/particle3_showcase_observer_probe.cpp',
                               libraries=['PhysXCommon_64.lib']))
