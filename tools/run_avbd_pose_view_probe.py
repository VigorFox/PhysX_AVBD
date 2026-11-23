"""Verify production candidate views against private complete-state evaluations."""
import argparse
import xml.etree.ElementTree as ET
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', choices=('checked', 'debug', 'release'), default='checked')
    options = parser.parse_args()
    project = ROOT/'physx/compiler/vc17win64-cpu-only/sdk_source_bin/LowLevelDynamics.vcxproj'
    node = ET.parse(project).find('.//{*}AdditionalIncludeDirectories')
    includes = [p for p in node.text.split(';') if p and '%(' not in p]
    raise SystemExit(run_probe(ROOT/'tools/avbd_pose_view_probe.cpp', include_dirs=includes,
        compiler_options=('/fp:fast',), config=options.config))
