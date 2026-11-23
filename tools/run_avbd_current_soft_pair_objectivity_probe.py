#!/usr/bin/env python3
"""Check a constructed current production row; does not qualify full scenes."""
import xml.etree.ElementTree as ET
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == "__main__":
    project = ROOT / "physx/compiler/vc17win64-cpu-only/sdk_source_bin/LowLevelDynamics.vcxproj"
    include = ET.parse(project).find(".//{*}AdditionalIncludeDirectories").text
    directories = [path for path in include.split(";") if path and "$(" not in path and "%(" not in path]
    raise SystemExit(run_probe(ROOT / "tools/avbd_current_soft_pair_objectivity_probe.cpp",
                              include_dirs=directories, defines=["DISABLE_CUDA_PHYSX"],
                              extra_sources=[ROOT / "physx/source/lowleveldynamics/src/avbd/DyKernelTable.cpp"]))
