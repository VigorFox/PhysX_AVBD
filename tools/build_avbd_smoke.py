#!/usr/bin/env python3
"""Compile tools/avbd_smoke.cpp against the chosen CPU-only PhysX config."""
from __future__ import print_function

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VCVARS = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="checked",
                        choices=("checked", "debug", "release", "profile"))
    parser.add_argument("--output", default=None,
                        help="Output exe path. Default: tools/avbd_smoke_new.exe")
    parser.add_argument("--bin-dir", default=None,
                        help="Explicit SDK library/DLL directory for an isolated build")
    args = parser.parse_args()
    bin_dir = os.path.abspath(args.bin_dir or os.path.join(
        ROOT, "physx", "bin", "win.x86_64.vc143.md", args.config))
    output = os.path.abspath(args.output or os.path.join(ROOT, "tools", "avbd_smoke_new.exe"))
    src = os.path.join(ROOT, "tools", "avbd_smoke.cpp")
    if not os.path.isfile(VCVARS):
        print("[AVBD_SMOKE_BUILD_ERROR] missing vcvars64: %s" % VCVARS)
        return 2
    if not os.path.isfile(os.path.join(bin_dir, "PhysX_64.lib")):
        print("[AVBD_SMOKE_BUILD_ERROR] missing PhysX_64.lib in %s" % bin_dir)
        return 2
    defs = " ".join([
        "/DWIN32", "/DWIN64", "/DNDEBUG", "/DPX_CHECKED=1" if args.config == "checked" else "/DPX_CHECKED=0",
        "/DPX_NVTX=0", "/DPX_SUPPORT_PVD=1", "/DPX_PUBLIC_RELEASE=1", "/DDISABLE_CUDA_PHYSX",
        "/D_CRT_SECURE_NO_DEPRECATE", "/D_CRT_NONSTDC_NO_DEPRECATE",
        "/D_HAS_ITERATOR_DEBUGGING=0", "/D_ITERATOR_DEBUG_LEVEL=0",
    ])
    includes = " ".join([
        '/I"%s"' % os.path.join(ROOT, "physx", "include"),
        '/I"%s"' % os.path.join(ROOT, "physx", "include", "cudamanager"),
        '/I"%s"' % os.path.join(ROOT, "tools"),
    ])
    libs = " ".join([
        os.path.join(bin_dir, "PhysXExtensions_static_64.lib"),
        os.path.join(bin_dir, "PhysXPvdSDK_static_64.lib"),
        os.path.join(bin_dir, "PhysXCooking_64.lib"),
        os.path.join(bin_dir, "PhysX_64.lib"),
        os.path.join(bin_dir, "PhysXCommon_64.lib"),
        os.path.join(bin_dir, "PhysXFoundation_64.lib"),
    ])
    compile_line = (
        'cl /nologo /O2 /MD /EHsc /std:c++14 /W4 /wd4514 /wd4820 /wd4127 /wd4710 /wd4711 /wd4577 /wd4996 '
        '%s %s /Fe:"%s" "%s" /link /LIBPATH:"%s" %s' %
        (defs, includes, output, src, bin_dir, libs)
    )
    bat_path = os.path.join(ROOT, "tools", "_build_avbd_smoke.cmd")
    with open(bat_path, "w", encoding="ascii") as handle:
        handle.write("call \"%s\"\r\n%s\r\n" % (VCVARS, compile_line))
    print("[AVBD_SMOKE_BUILD] config=%s output=%s" % (args.config, output))
    print("[AVBD_SMOKE_BUILD_CMD] %s" % compile_line)
    try:
        proc = subprocess.run(["cmd", "/c", bat_path], cwd=os.path.join(ROOT, "tools"))
    finally:
        try:
            os.remove(bat_path)
        except OSError:
            pass
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
