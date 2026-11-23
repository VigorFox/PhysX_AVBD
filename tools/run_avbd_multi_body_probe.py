from run_avbd_ogc_certificate_probe import ROOT, run_probe
import argparse
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Four-support material math and public Host lifecycle')
    parser.add_argument('--config',choices=('checked','debug','release'),default='checked')
    subset=parser.add_mutually_exclusive_group()
    subset.add_argument('--mixed-only',action='store_true')
    subset.add_argument('--sleep-count-only',action='store_true')
    args=parser.parse_args()
    raise SystemExit(run_probe(ROOT/'tools/avbd_multi_body_probe.cpp',
        args=['--mixed-only'] if args.mixed_only else (['--sleep-count-only'] if args.sleep_count_only else []),
        libraries=('PhysXExtensions_static_64.lib','PhysXPvdSDK_static_64.lib','PhysX_64.lib','PhysXCommon_64.lib','PhysXCooking_64.lib'),config=args.config))
