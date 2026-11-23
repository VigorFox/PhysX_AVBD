#!/usr/bin/env python3
"""Run distilled SnippetArticulationRC headless gates (P6)."""

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
EXECUTABLE = "SnippetArticulationRC_64.exe"

GATED_CASES = (
    ("scissor-unloaded", 600, (
        "fetchFailures", "fetchErrorState", "nonFinite", "physicsErrors")),
    ("scissor-cycle", 600, (
        "fetchFailures", "fetchErrorState", "nonFinite", "physicsErrors")),
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


def run_exe(bin_dir, extra_args, timeout):
    exe = os.path.join(bin_dir, EXECUTABLE)
    argv = [exe, "--headless"] + extra_args
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
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return 124, stdout, stderr, True
    return proc.returncode, stdout, stderr, False


def gate_lines(text):
    return [
        line.strip()
        for line in text.splitlines()
        if line.startswith("[AVBD_GATE] ")
    ]


def run_gated(bin_dir, solver, case_name, frames, zero_keys, timeout):
    extra = [
        "--solver=%s" % solver,
        "--case=%s" % case_name,
        "--frames=%s" % frames,
        "--execution=parallel",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    code, stdout, stderr, timed_out = run_exe(bin_dir, extra, timeout)
    combined = stdout + (("\n" + stderr) if stderr else "")
    errors = []
    if timed_out:
        errors.append("timed out")
    lines = gate_lines(combined)
    fields = {}
    if len(lines) != 1:
        errors.append("gate count is %s, expected 1" % len(lines))
    else:
        fields, parse_errors = parse_fields(lines[0])
        errors.extend(parse_errors)
        required = {
            "schema": "1",
            "snippet": "SnippetArticulationRC",
            "case": case_name,
            "solver": solver,
            "status": "PASS",
            "reason": "none",
            "capability": "SUPPORTED",
            "validation": "GATED",
        }
        for key, expected in required.items():
            if fields.get(key) != expected:
                errors.append("%s=%r, expected %r" % (key, fields.get(key), expected))
        for key in zero_keys:
            if fields.get(key) != "0":
                errors.append("%s=%r, expected '0'" % (key, fields.get(key)))
        if fields.get("completedFrames") != str(frames):
            errors.append(
                "completedFrames=%r, expected %s" % (fields.get("completedFrames"), frames)
            )
        if fields.get("linkCount") != "18":
            errors.append("linkCount=%r, expected '18'" % fields.get("linkCount"))
        if fields.get("loopJointCount") != "8":
            errors.append(
                "loopJointCount=%r, expected '8'" % fields.get("loopJointCount")
            )
        expected_cargo = "8" if case_name == "scissor-cycle" else "0"
        if fields.get("cargoCount") != expected_cargo:
            errors.append(
                "cargoCount=%r, expected %r" % (fields.get("cargoCount"), expected_cargo)
            )
    if code != 0:
        errors.append("exit code %s, expected 0" % code)
    print(
        "[ARTIC_HEADLESS] case=%s solver=%s exit=%s runner=%s"
        % (case_name, solver, code, "PASS" if not errors else "FAIL")
    )
    if combined:
        sys.stdout.write(combined.rstrip() + "\n")
    for error in errors:
        print("[ARTIC_HEADLESS_ERROR] case=%s error=%s" % (case_name, error))
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", default=DEFAULT_BIN_DIR)
    parser.add_argument("--solver", default="avbd")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--cases",
        default="gated",
        help="gated, or a comma-separated case list",
    )
    args = parser.parse_args()
    bin_dir = os.path.abspath(args.bin_dir)
    exe = os.path.join(bin_dir, EXECUTABLE)
    if not os.path.isfile(exe):
        print("[ARTIC_HEADLESS_ERROR] missing executable: %s" % exe)
        return 2

    want = args.cases
    selected = None
    known = set(name for name, _, _ in GATED_CASES)
    if want not in ("gated", "all"):
        error, selected = avbd_gate.resolve_selection(want, known)
        if error:
            print("[ARTIC_HEADLESS_ERROR] %s" % error)
            print("[ARTIC_HEADLESS_MATRIX] solver=%s result=FAIL" % args.solver)
            return avbd_gate.CONFIG_ERROR

    passed = True
    executed = 0
    for case_name, frames, zero_keys in GATED_CASES:
        if selected is not None and case_name not in selected:
            continue
        executed += 1
        passed = (
            run_gated(bin_dir, args.solver, case_name, frames, zero_keys, args.timeout)
            and passed
        )
    if executed == 0:
        print("[ARTIC_HEADLESS_ERROR] empty execution set")
        print("[ARTIC_HEADLESS_MATRIX] solver=%s result=FAIL" % args.solver)
        return avbd_gate.CONFIG_ERROR
    print("[ARTIC_HEADLESS_MATRIX] solver=%s result=%s" % (args.solver, "PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
