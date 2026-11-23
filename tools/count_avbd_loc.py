#!/usr/bin/env python3
"""Count AVBD CPU production lines (outline stage 0 / P6 budget).

Counts:
  physx/source/lowleveldynamics/src/avbd/**
  physx/source/lowleveldynamics/include/DyAvbd*
  physx/source/physx/src/NpAvbd*

CUDA AVBD (gpusolver/) is listed separately and does not count toward 30k.
"""
from __future__ import print_function

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYSX = os.path.join(ROOT, "physx", "source")

CPU_GLOBS = [
    ("lowleveldynamics/src/avbd", os.path.join(PHYSX, "lowleveldynamics", "src", "avbd")),
    ("lowleveldynamics/include DyAvbd*", os.path.join(PHYSX, "lowleveldynamics", "include")),
    ("physx/src NpAvbd*", os.path.join(PHYSX, "physx", "src")),
]

CUDA_DIR = os.path.join(PHYSX, "gpusolver")

EXTS = {".h", ".hpp", ".c", ".cpp", ".cu", ".cuh"}


def count_file(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def collect_cpu():
    files = []
    avbd_dir = os.path.join(PHYSX, "lowleveldynamics", "src", "avbd")
    if os.path.isdir(avbd_dir):
        for root, _, names in os.walk(avbd_dir):
            for name in names:
                ext = os.path.splitext(name)[1].lower()
                if ext in EXTS:
                    files.append(os.path.join(root, name))
    inc = os.path.join(PHYSX, "lowleveldynamics", "include")
    if os.path.isdir(inc):
        for name in os.listdir(inc):
            if name.startswith("DyAvbd") and os.path.splitext(name)[1].lower() in EXTS:
                files.append(os.path.join(inc, name))
    np_dir = os.path.join(PHYSX, "physx", "src")
    if os.path.isdir(np_dir):
        for name in os.listdir(np_dir):
            if name.startswith("NpAvbd") and os.path.splitext(name)[1].lower() in EXTS:
                files.append(os.path.join(np_dir, name))
    files.sort()
    return files


def collect_cuda():
    files = []
    if not os.path.isdir(CUDA_DIR):
        return files
    for root, _, names in os.walk(CUDA_DIR):
        for name in names:
            lower = name.lower()
            if "avbd" not in lower:
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in EXTS:
                files.append(os.path.join(root, name))
    files.sort()
    return files


def print_group(title, files):
    print(title)
    total = 0
    for path in files:
        n = count_file(path)
        total += n
        rel = os.path.relpath(path, ROOT)
        print("  %6d  %s" % (n, rel.replace("\\", "/")))
    print("  files=%d  lines=%d" % (len(files), total))
    return len(files), total


def main():
    cpu_files = collect_cpu()
    cuda_files = collect_cuda()
    nfiles, nlines = print_group("AVBD CPU production", cpu_files)
    print("")
    print_group("CUDA AVBD (gpusolver, not in 30k)", cuda_files)
    print("")
    print("budget: <30000 lines, <40 files")
    print("cpu:    %d lines, %d files" % (nlines, nfiles))
    ok = nlines < 30000 and nfiles < 40
    print("status: %s" % ("OK" if ok else "OVER"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
