"""Check the production local translation subspace through KernelTable."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    for mode in ("/fp:precise", "/fp:fast"):
        print("Local reaction " + mode, flush=True)
        code = run_probe(ROOT / "tools/avbd_local_reaction_probe.cpp",
                         extra_sources=[ROOT / "physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp"],
                         compiler_options=(mode,))
        if code:
            raise SystemExit(code)
