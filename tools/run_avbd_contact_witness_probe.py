"""Check finite-rotation contact coordinates and their normal derivatives."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    for mode in ("precise", "fast"):
        print("Contact surface witnesses /fp:" + mode, flush=True)
        code = run_probe(ROOT / "tools/avbd_contact_witness_probe.cpp",
                         include_dirs=(ROOT / "physx/source/lowlevel/software/include",),
                         compiler_options=("/fp:" + mode,))
        if code:
            raise SystemExit(code)
