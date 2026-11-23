"""Verify the shared OGC normal envelope, including its closed-boundary input."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe
if __name__ == "__main__":
    raise SystemExit(run_probe(ROOT / "tools/avbd_np_normal_envelope_probe.cpp",
        compiler_options=("/fp:fast",)))
