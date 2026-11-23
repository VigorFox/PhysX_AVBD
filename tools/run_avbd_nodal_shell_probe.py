#!/usr/bin/env python3
"""Independent derivatives, invariance, and actual 3x3 Table execution."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    for mode in ("/fp:precise", "/fp:fast"):
        result = run_probe(ROOT / "tools/avbd_nodal_shell_probe.cpp",
                           extra_sources=[ROOT / "physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp"],
                           compiler_options=(mode,))
        if result:
            raise SystemExit(result)
