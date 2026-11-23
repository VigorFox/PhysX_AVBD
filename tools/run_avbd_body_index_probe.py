"""Verify temporary Host-body indexing, with optional cost calibration."""
import sys,xml.etree.ElementTree as ET
from run_avbd_ogc_certificate_probe import ROOT,run_probe
if __name__=='__main__':
    node=ET.parse(ROOT/'physx/compiler/vc17win64-cpu-only/sdk_source_bin/LowLevelDynamics.vcxproj').find('.//{*}AdditionalIncludeDirectories')
    includes=[p for p in node.text.split(';') if p and '%(' not in p]
    raise SystemExit(run_probe(ROOT/'tools/avbd_body_index_probe.cpp',args=sys.argv[1:],include_dirs=includes,compiler_options=('/fp:fast',)))
