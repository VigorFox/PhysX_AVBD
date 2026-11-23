"""Check production transform under the SDK's fast floating-point model."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    raise SystemExit(run_probe(ROOT / "tools/avbd_contact_transform_probe.cpp",
                              compiler_options=("/fp:fast",)))
