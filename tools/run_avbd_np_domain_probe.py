"""Test complete NP support/path admission, including actual float endpoints."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe
if __name__ == "__main__":
    raise SystemExit(run_probe(ROOT / "tools/avbd_np_domain_probe.cpp",
        compiler_options=("/fp:fast",)))
