#!/usr/bin/env python3
"""Run distilled SnippetJoint headless gates (P6)."""

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
EXECUTABLE = "SnippetJoint_64.exe"

GATED_CASES = (
    ("hang-spherical", 180, ("fetchFailures", "fetchErrorState", "launchFailures",
                             "brokenCount", "breakCallbacks")),
    ("hang-d6", 180, ("fetchFailures", "fetchErrorState", "launchFailures",
                      "brokenCount", "breakCallbacks")),
    ("fixed-no-break", 600, ("fetchFailures", "fetchErrorState", "launchFailures",
                             "brokenCount", "breakCallbacks")),
    ("fixed-break", 600, ("fetchFailures", "fetchErrorState", "launchFailures")),
)

UNSUPPORTED_CASES = (
    ("hang", "multi_body_fixed_lock_chain"),
    ("hang-fixed", "multi_body_fixed_lock_chain"),
    ("impact-all", "not_ported_this_slice"),
    ("wide-joint-stress", "not_ported_this_slice"),
    ("impact", "not_ported_this_slice"),
    ("force-static", "not_ported_this_slice"),
    ("force-offset", "not_ported_this_slice"),
    ("force-pair", "not_ported_this_slice"),
    ("force-pair-disabled", "not_ported_this_slice"),
    ("spherical-cone-inside", "not_ported_this_slice"),
    ("spherical-cone-outside", "not_ported_this_slice"),
    ("native-reaction", "not_ported_this_slice"),
    ("native-no-break", "not_ported_this_slice"),
    ("native-break", "not_ported_this_slice"),
    ("revolute-motor", "not_ported_this_slice"),
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
            "snippet": "SnippetJoint",
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
        if case_name == "fixed-break":
            if fields.get("brokenCount") != "1":
                errors.append("brokenCount=%r, expected '1'" % fields.get("brokenCount"))
            if fields.get("breakCallbacks") != "1":
                errors.append(
                    "breakCallbacks=%r, expected '1'" % fields.get("breakCallbacks")
                )
            try:
                first_broken = int(fields.get("firstBrokenFrame", "-1"))
            except ValueError:
                first_broken = -1
            if first_broken < 1 or first_broken > frames:
                errors.append(
                    "firstBrokenFrame=%s, expected [1,%s]" % (first_broken, frames)
                )
    if code != 0:
        errors.append("exit code %s, expected 0" % code)
    print(
        "[JOINT_HEADLESS] case=%s solver=%s exit=%s runner=%s"
        % (case_name, solver, code, "PASS" if not errors else "FAIL")
    )
    if combined:
        sys.stdout.write(combined.rstrip() + "\n")
    for error in errors:
        print("[JOINT_HEADLESS_ERROR] case=%s error=%s" % (case_name, error))
    return not errors


def run_unsupported(bin_dir, solver, case_name, reason, timeout):
    extra = ["--solver=%s" % solver, "--case=%s" % case_name]
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
            "snippet": "SnippetJoint",
            "case": case_name,
            "capability": "UNSUPPORTED",
            "status": "ERROR",
            "reason": reason,
        }
        for key, expected in required.items():
            if fields.get(key) != expected:
                errors.append("%s=%r, expected %r" % (key, fields.get(key), expected))
    if code != 3:
        errors.append("exit code %s, expected 3" % code)
    print(
        "[JOINT_HEADLESS] case=%s unsupported exit=%s runner=%s"
        % (case_name, code, "PASS" if not errors else "FAIL")
    )
    if combined:
        sys.stdout.write(combined.rstrip() + "\n")
    for error in errors:
        print("[JOINT_HEADLESS_ERROR] case=%s error=%s" % (case_name, error))
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", default=DEFAULT_BIN_DIR)
    parser.add_argument("--solver", default="avbd")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--cases",
        default="all",
        help="all, gated, unsupported, or a comma-separated case list",
    )
    args = parser.parse_args()
    bin_dir = os.path.abspath(args.bin_dir)
    exe = os.path.join(bin_dir, EXECUTABLE)
    if not os.path.isfile(exe):
        print("[JOINT_HEADLESS_ERROR] missing executable: %s" % exe)
        return 2

    want = args.cases
    selected = None
    run_gated_cases = want in ("all", "gated")
    run_unsupported_cases = want in ("all", "unsupported")
    known = set(name for name, _, _ in GATED_CASES)
    known.update(name for name, _ in UNSUPPORTED_CASES)
    if want not in ("all", "gated", "unsupported"):
        error, selected = avbd_gate.resolve_selection(want, known)
        if error:
            print("[JOINT_HEADLESS_ERROR] %s" % error)
            print("[JOINT_HEADLESS_MATRIX] solver=%s result=FAIL" % args.solver)
            return avbd_gate.CONFIG_ERROR
        run_gated_cases = True
        run_unsupported_cases = True

    passed = True
    executed = 0
    if run_gated_cases:
        for case_name, frames, zero_keys in GATED_CASES:
            if selected is not None and case_name not in selected:
                continue
            executed += 1
            passed = (
                run_gated(bin_dir, args.solver, case_name, frames, zero_keys, args.timeout)
                and passed
            )
    if run_unsupported_cases:
        for case_name, reason in UNSUPPORTED_CASES:
            if selected is not None and case_name not in selected:
                continue
            executed += 1
            passed = (
                run_unsupported(bin_dir, args.solver, case_name, reason, args.timeout)
                and passed
            )
    if executed == 0:
        print("[JOINT_HEADLESS_ERROR] empty execution set")
        print("[JOINT_HEADLESS_MATRIX] solver=%s result=FAIL" % args.solver)
        return avbd_gate.CONFIG_ERROR
    print("[JOINT_HEADLESS_MATRIX] solver=%s result=%s" % (args.solver, "PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
