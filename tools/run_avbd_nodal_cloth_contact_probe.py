"""Actual NP cloth drape under an explicitly restricted material contract."""
import sys
from run_avbd_ogc_certificate_probe import ROOT, run_probe
if __name__ == '__main__':
    args = sys.argv[1:]
    probes = {'--wide-star': 'avbd_nodal_wide_star_probe.cpp',
              '--warm-admission': 'avbd_warm_admission_probe.cpp'}
    source = 'avbd_nodal_cloth_contact_probe.cpp'
    if args and args[0] in probes:
        source = probes[args.pop(0)]
    raise SystemExit(run_probe(ROOT / 'tools' / source, args=args,
        libraries=['PhysXExtensions_static_64.lib','PhysXPvdSDK_static_64.lib',
                   'PhysXCooking_64.lib','PhysX_64.lib','PhysXCommon_64.lib']))
