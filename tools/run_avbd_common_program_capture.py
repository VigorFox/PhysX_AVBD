"""Capture actual production linearizations; no physical candidate is published."""
import argparse
import json
import math
import os
from pathlib import Path
from run_avbd_ogc_certificate_probe import ROOT, run_probe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, choices=range(6), default=0)
    parser.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    args.capture.parent.mkdir(parents=True, exist_ok=True)
    os.environ['PHYSX_AVBD_COMMON_CAPTURE'] = str(args.capture.resolve())
    os.environ['PHYSX_AVBD_COMMON_CAPTURE_CONTACTS'] = '1'
    os.environ['PHYSX_AVBD_COMMON_CAPTURE_SOLVE'] = '1'
    os.environ['PHYSX_NP_GEOMETRY_CERTIFICATES'] = '2'
    code=run_probe(ROOT / 'tools/avbd_common_program_scene_probe.cpp', args=[str(args.mode)],
        libraries=('PhysXExtensions_static_64.lib', 'PhysXPvdSDK_static_64.lib',
                   'PhysX_64.lib', 'PhysXCommon_64.lib', 'PhysXCooking_64.lib'))
    if code:
        raise SystemExit(code)
    data=json.loads(args.capture.read_text(encoding='utf-8'))
    assert data['complete'] and data['tableSolved'],data['reason']
    normal_cap=float(data['upper'][0])
    assert abs(normal_cap-.06)<1e-7 if args.mode==1 else math.isinf(normal_cap)
    assert abs(float(data['upper'][3])-.8)<1e-7 and abs(float(data['lower'][3])+.8)<1e-7
