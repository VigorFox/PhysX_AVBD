"""Compile the shared normal-cone operator with production fast arithmetic."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe
if __name__ == '__main__':
    raise SystemExit(run_probe(ROOT / 'tools/avbd_normal_cone_probe.cpp',compiler_options=('/fp:fast',)))
