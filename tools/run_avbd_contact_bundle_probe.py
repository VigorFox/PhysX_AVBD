"""Verify contact aggregation identities and failure boundaries, not a solver."""
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    for mode in ('precise', 'fast'):
        print('Contact bundle /fp:' + mode, flush=True)
        code = run_probe(ROOT / 'tools/avbd_contact_bundle_probe.cpp', compiler_options=('/fp:' + mode,))
        if code:
            print('Probe exit:', code, flush=True)
            raise SystemExit(code)
