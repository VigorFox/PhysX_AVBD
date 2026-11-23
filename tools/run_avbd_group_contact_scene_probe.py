"""Exercise a real two-end NP contact inside the production material group."""
import os
import json
import subprocess
import sys
from run_avbd_ogc_certificate_probe import ROOT, run_probe
from run_snippet_joint_soft_body_headless import group_kernel_observation

if __name__ == '__main__':
    if '--child' in sys.argv:
        os.environ['PHYSX_AVBD_SCHEDULE_STATS'] = '1'
        os.environ['PHYSX_NP_GEOMETRY_CERTIFICATES'] = '0'
        raise SystemExit(run_probe(ROOT / 'tools/avbd_group_contact_scene_probe.cpp',
            libraries=('PhysXExtensions_static_64.lib', 'PhysXPvdSDK_static_64.lib',
                       'PhysX_64.lib', 'PhysXCommon_64.lib', 'PhysXCooking_64.lib')))
    process = subprocess.run([sys.executable, __file__, '--child'], capture_output=True)
    output = (process.stdout + process.stderr).decode('utf-8', errors='replace')
    observation = group_kernel_observation(output)
    contact = observation['contactTotals'] or {}
    exercised = observation['valid'] and contact.get('sharedContactAccepted', 0) > 0 and contact.get('frictionAccepted', 0) > 0
    for line in output.splitlines():
        if '[GROUP_CONTACT_SCENE]' in line:
            print(line)
    print('[GROUP_CONTACT_COVERAGE] ' + json.dumps(observation))
    if process.returncode or not exercised:
        print(output)
        raise SystemExit(process.returncode or 1)
