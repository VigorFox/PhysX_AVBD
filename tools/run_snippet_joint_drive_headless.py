#!/usr/bin/env python3
"""Run the distilled SnippetJointDrive velocity/position drive matrix."""

from __future__ import print_function

import argparse
import os
import subprocess
import sys

import avbd_gate


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BIN_DIR = os.path.join(
    REPO_ROOT, "physx", "bin", "win.x86_64.vc143.md", "checked"
)
EXECUTABLE = "SnippetJointDrive_64.exe"
CASES = (
    "velocity-x",
    "velocity-x-frame",
    "velocity-twist",
    "velocity-swing1",
    "velocity-swing2",
    "velocity-slerp",
    "velocity-slerp-frame",
    "force-limit-x",
    "position-x",
    "position-x-frame",
    "position-twist",
    "position-swing1",
    "position-swing2",
    "position-slerp",
)


def parse_fields(line):
    fields = {}
    errors = []
    for token in line.split()[1:]:
        if "=" not in token:
            errors.append("malformed token: %s" % token)
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append("duplicate key: %s" % key)
        fields[key] = value
    return fields, errors


def run_case(bin_dir, solver, execution, case_name, frames, timeout):
    exe = os.path.join(bin_dir, EXECUTABLE)
    argv = [
        exe,
        "--headless",
        "--solver=%s" % solver,
        "--case=%s" % case_name,
        "--frames=%s" % frames,
        "--execution=%s" % execution,
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    proc = subprocess.Popen(
        argv,
        cwd=bin_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        universal_newlines=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    combined = stdout + (("\n" + stderr) if stderr else "")
    errors = ["timed out"] if timed_out else []
    lines = [line.strip() for line in combined.splitlines()
             if line.startswith("[AVBD_GATE] ")]
    fields = {}
    if len(lines) != 1:
        errors.append("gate count is %s, expected 1" % len(lines))
    else:
        fields, parse_errors = parse_fields(lines[0])
        errors.extend(parse_errors)
        required = {
            "schema": "1",
            "snippet": "SnippetJointDrive",
            "solver": solver,
            "case": case_name,
            "status": "PASS",
            "reason": "none",
            "capability": "SUPPORTED",
            "validation": "GATED",
            "completedFrames": str(frames),
            "nonFinite": "0",
            "fatalErrors": "0",
        }
        for key, expected in required.items():
            if fields.get(key) != expected:
                errors.append("%s=%r, expected %r" %
                              (key, fields.get(key), expected))
    if proc.returncode != 0:
        errors.append("exit code %s, expected 0" % proc.returncode)
    print("[JOINT_DRIVE_HEADLESS] case=%s solver=%s execution=%s exit=%s runner=%s" %
          (case_name, solver, execution, proc.returncode,
           "PASS" if not errors else "FAIL"))
    if combined:
        sys.stdout.write(combined.rstrip() + "\n")
    for error in errors:
        print("[JOINT_DRIVE_HEADLESS_ERROR] case=%s error=%s" %
              (case_name, error))
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", default=DEFAULT_BIN_DIR)
    parser.add_argument("--solver", default="avbd")
    parser.add_argument("--execution", choices=("parallel", "sequential"),
                        default="parallel")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--cases", default="all")
    args = parser.parse_args()
    bin_dir = os.path.abspath(args.bin_dir)
    exe = os.path.join(bin_dir, EXECUTABLE)
    if not os.path.isfile(exe):
        print("[JOINT_DRIVE_HEADLESS_ERROR] missing executable: %s" % exe)
        return 2
    error, selected = avbd_gate.resolve_selection(
        args.cases, set(CASES), aliases={"all": None})
    if error:
        print("[JOINT_DRIVE_HEADLESS_ERROR] %s" % error)
        return 2
    if selected is None:
        selected = CASES
    if not selected:
        print("[JOINT_DRIVE_HEADLESS_ERROR] empty case selection")
        return 2
    passed = True
    for case_name in selected:
        passed = run_case(bin_dir, args.solver, args.execution,
                          case_name, args.frames,
                          args.timeout) and passed
    print("[JOINT_DRIVE_HEADLESS_MATRIX] solver=%s execution=%s result=%s" %
          (args.solver, args.execution, "PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
