#!/usr/bin/env python3
"""Dispatch AVBD contact-coverage cases from the manifest.

The runner only schedules, parses GATE records, and aggregates. It does not
reimplement collision or physics oracles. Planned/unsupported/invalid-sdk
cells never enter the support-rate numerator.
"""
from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
import time

import avbd_gate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "tools", "avbd_contact_coverage_cases.json")
SNIPPET_EXE = {
    "volume": "SnippetDeformableVolumeAVBD_64.exe",
    "surface": "SnippetDeformableSurfaceAVBD_64.exe",
    "hello": "SnippetHelloWorld_64.exe",
    "joint": "SnippetJoint_64.exe",
    "artic": "SnippetArticulationRC_64.exe",
    "chainmail": "SnippetChainmail_64.exe",
    "joint-drive": "SnippetJointDrive_64.exe",
}
SNIPPET_LABEL = {key: name.removesuffix("_64.exe") for key, name in SNIPPET_EXE.items()}


def expected_status_for(case, config):
    by_config = case.get("expected_status_by_config") or {}
    if config in by_config:
        return by_config[config]
    return case.get("expected_status", "planned")


def select_cases(manifest, package, case_filter):
    pool = avbd_gate.cases_for_package(manifest, package)
    known_names = set()
    for item in manifest.get("cases") or []:
        if item.get("id"):
            known_names.add(item["id"])
        if item.get("case"):
            known_names.add(item["case"])
    if case_filter in (None, "", "all"):
        if not pool:
            return "empty case selection", []
        return None, pool
    error, names = avbd_gate.resolve_selection(case_filter, known_names)
    if error:
        return error, []
    wanted = set(names)
    selected = []
    found = set()
    for item in pool:
        if item.get("id") in wanted or item.get("case") in wanted:
            selected.append(item)
            found.add(item.get("id"))
            found.add(item.get("case"))
    missing = [name for name in names if name not in found]
    if missing:
        return "unknown cases: %s" % ",".join(missing), []
    if not selected:
        return "empty case selection", []
    return None, selected


def process_output(stdout, stderr):
    # TimeoutExpired may contain bytes even when subprocess.run uses text=True.
    def decode(value):
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
    out, err = decode(stdout), decode(stderr)
    return out + (("\n" + err) if err else "")


def run_snippet(bin_dir, spec, solver, timeout):
    exe_name = spec.get("exe") or SNIPPET_EXE.get(spec.get("snippet"))
    exe = os.path.join(bin_dir, exe_name)
    argv = [
        exe,
        "--headless",
        "--solver=" + solver,
        "--case=" + spec["case"],
        "--frames=" + str(spec.get("frames", 1)),
        "--dispatcher-threads=" + str(spec.get("dispatcher_threads", 2)),
    ]
    if spec.get("snippet") in ("volume", "surface"):
        argv.append("--particle3")  # Existing coverage cases exercise public deformable APIs.
    if spec.get("snippet") in ("hello", "joint", "artic"):
        argv.extend(["--execution=parallel", "--seed=1"])
    if not os.path.isfile(exe):
        return {
            "ran": False,
            "exit": avbd_gate.CONFIG_ERROR,
            "timed_out": False,
            "text": "",
            "error": "missing executable: %s" % exe,
            "argv": argv,
        }
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    if spec.get("case") == "visual-showcase" or spec.get("snippet") == "chainmail":
        env.setdefault("PHYSX_AVBD_HEADLESS_PROGRESS", "1")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv, cwd=bin_dir, text=True, capture_output=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ran": True,
            "exit": 124,
            "timed_out": True,
            "timeout_seconds": timeout,
            "elapsed_seconds": time.monotonic() - started,
            "text": process_output(exc.stdout, exc.stderr),
            "error": "timeout",
            "argv": argv,
        }
    text = process_output(proc.stdout, proc.stderr)
    return {
        "ran": True,
        "exit": proc.returncode,
        "timed_out": False,
        "timeout_seconds": timeout,
        "elapsed_seconds": time.monotonic() - started,
        "text": text,
        "error": None,
        "argv": argv,
    }


def run_smoke(root, bin_dir, timeout):
    exe = os.path.join(root, "tools", spec_smoke_name())
    if not os.path.isfile(exe):
        exe = os.path.join(root, "tools", "avbd_smoke.exe")
    argv = [exe]
    if not os.path.isfile(exe):
        return {
            "ran": False,
            "exit": avbd_gate.CONFIG_ERROR,
            "timed_out": False,
            "text": "",
            "error": "missing smoke executable: %s" % exe,
            "argv": argv,
        }
    env = os.environ.copy()
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv, cwd=bin_dir, text=True, capture_output=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ran": True,
            "exit": 124,
            "timed_out": True,
            "timeout_seconds": timeout,
            "elapsed_seconds": time.monotonic() - started,
            "text": process_output(exc.stdout, exc.stderr),
            "error": "timeout",
            "argv": argv,
        }
    text = process_output(proc.stdout, proc.stderr)
    return {
        "ran": True,
        "exit": proc.returncode,
        "timed_out": False,
        "timeout_seconds": timeout,
        "elapsed_seconds": time.monotonic() - started,
        "text": text,
        "error": None,
        "argv": argv,
    }


def spec_smoke_name():
    return "avbd_smoke_new.exe"


def evaluate_run(spec, run, config, solver):
    expected_status = expected_status_for(spec, config)
    row = {
        "id": spec.get("id"),
        "snippet": spec.get("snippet"),
        "case": spec.get("case"),
        "kind": spec.get("kind", "snippet"),
        "pair": spec.get("pair"),
        "geometry": spec.get("geometry"),
        "motion": spec.get("motion"),
        "expected_status": expected_status,
        "selected": True,
        "executed": False,
        "completed": False,
        "observed_status": None,
        "capability": None,
        "exit": None,
        "errors": [],
        "fields": {},
        "physical_valid": False,
        "argv": run.get("argv") if run else None,
        "process_output": run.get("text", "") if run else "",
        "timed_out": run.get("timed_out", False) if run else False,
        "timeout_seconds": run.get("timeout_seconds") if run else None,
        "elapsed_seconds": run.get("elapsed_seconds") if run else None,
    }
    if expected_status == "planned":
        row["observed_status"] = "planned"
        return row
    if expected_status == "invalid-sdk":
        row["observed_status"] = "invalid-sdk"
        return row
    if not run or not run.get("ran"):
        row["errors"].append(run.get("error") if run else "not run")
        row["observed_status"] = "error"
        return row
    row["executed"] = True
    row["exit"] = run.get("exit")
    if run.get("timed_out"):
        row["errors"].append("timeout")
        row["observed_status"] = "error"
        return row
    if spec.get("kind") == "smoke":
        text = run.get("text") or ""
        fail_lines = [line for line in text.splitlines() if line.startswith("FAIL ")]
        pass_lines = [line for line in text.splitlines() if line.startswith("PASS ")]
        if expected_status == "pass":
            ok = run.get("exit") == 0 and not fail_lines and len(pass_lines) > 0
        else:
            ok = bool(fail_lines) and run.get("exit") == 1
        row["completed"] = True
        row["observed_status"] = "pass" if (run.get("exit") == 0 and not fail_lines) else "fail"
        row["fields"] = {
            "pass_lines": len(pass_lines),
            "fail_lines": len(fail_lines),
            "first_fail": fail_lines[0] if fail_lines else None,
        }
        row["physical_valid"] = bool(pass_lines or fail_lines) and (
            (run.get("exit") == 0 and not fail_lines) or
            (run.get("exit") == 1 and bool(fail_lines)))
        if not ok:
            row["errors"].append("smoke result did not match expected_status=%s" % expected_status)
        return row

    expected = {
        "case": spec.get("case"),
        "expected_status": expected_status if expected_status != "unsupported" else "unsupported",
        "required_fields": spec.get("required_fields") or ["schema", "snippet", "case", "status"],
        "numeric_fields": spec.get("numeric_fields") or [],
        "capability": spec.get("capability"),
    }
    expected["solver"] = solver
    if spec.get("snippet") in SNIPPET_LABEL:
        expected["snippet"] = SNIPPET_LABEL[spec["snippet"]]
    if expected_status in ("pass", "fail") and "frames" in spec:
        expected["frames"] = spec["frames"]
        expected["completedFrames"] = spec["frames"]
    ok, record = avbd_gate.evaluate_gate_text(run.get("text") or "", expected)
    row["fields"] = record.get("fields") or {}
    row["capability"] = record.get("capability")
    row["errors"].extend(record.get("errors") or [])
    status = (record.get("status") or "").lower()
    if record.get("capability") == "UNSUPPORTED":
        row["observed_status"] = "unsupported"
    elif status == "pass":
        row["observed_status"] = "pass"
    elif status == "fail":
        row["observed_status"] = "fail"
    else:
        row["observed_status"] = "error"
    if row["observed_status"] in ("pass", "fail", "unsupported"):
        row["completed"] = True
    if not avbd_gate.exit_matches_status(run.get("exit"), expected_status,
                                       normalized_nonzero=(os.name == "nt")):
        row["errors"].append("exit code %s inconsistent with %s" %
                             (run.get("exit"), expected_status))
        ok = False
    if expected_status == "unsupported" and record.get("capability") != "UNSUPPORTED":
        ok = False
    if expected_status == "fail" and row["observed_status"] != "fail":
        ok = False
    if expected_status == "pass" and row["observed_status"] != "pass":
        ok = False
    if not ok and not row["errors"]:
        row["errors"].append("result did not match expected_status=%s" % expected_status)
    if row["observed_status"] in ("pass", "fail"):
        # A baseline mismatch can still be a valid physical measurement. A
        # crash, corrupt GATE or runtime error cannot enter either numerator.
        observed = dict(expected, expected_status=row["observed_status"])
        valid, _ = avbd_gate.evaluate_gate_text(run.get("text") or "", observed)
        row["physical_valid"] = valid and avbd_gate.exit_matches_status(
            run.get("exit"), row["observed_status"])
    return row


def hash_binaries(bin_dir, extra_paths):
    names = [
        "PhysX_64.dll",
        "PhysXCommon_64.dll",
        "PhysXCooking_64.dll",
        "PhysXFoundation_64.dll",
        "SnippetDeformableVolumeAVBD_64.exe",
        "SnippetDeformableSurfaceAVBD_64.exe",
        "SnippetHelloWorld_64.exe",
        "SnippetJoint_64.exe",
        "SnippetArticulationRC_64.exe",
        "SnippetChainmail_64.exe",
    ]
    out = {}
    for name in names:
        path = os.path.join(bin_dir, name)
        if os.path.isfile(path):
            out[name] = avbd_gate.file_sha256(path)
    for path in extra_paths:
        if path and os.path.isfile(path):
            out[os.path.basename(path)] = avbd_gate.file_sha256(path)
    return out


def matrix_outcome(counts, tool_ok, require_physical_pass=False, measurement_ok=None):
    if measurement_ok is None:
        measurement_ok = tool_ok
    physical = ("FAIL" if counts["physical_fail"] else
                "PASS" if counts["physical_pass"] and measurement_ok else "NOT_VERIFIED")
    accepted = (measurement_ok and physical == "PASS") if require_physical_pass else tool_ok
    return ("PASS" if accepted else "FAIL"), physical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--package", default="C00")
    parser.add_argument("--config", default="checked",
                        choices=("checked", "debug", "release", "profile"))
    parser.add_argument("--solver", default="avbd")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--bin-dir", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--require-physical-pass", action="store_true",
                        help="Fail acceptance on physical FAIL or no verified physical execution, even if expected.")
    args = parser.parse_args()

    if not os.path.isfile(args.manifest):
        print("[AVBD_COVERAGE_ERROR] missing manifest: %s" % args.manifest)
        return avbd_gate.CONFIG_ERROR
    manifest = avbd_gate.load_manifest(args.manifest)
    error, selected = select_cases(manifest, args.package, args.cases)
    requested_names = avbd_gate.parse_case_list(args.cases) if args.cases not in ("all", "") else [
        item.get("id") for item in avbd_gate.cases_for_package(manifest, args.package)
    ]
    if error:
        print("[AVBD_COVERAGE_ERROR] %s" % error)
        print("[AVBD_COVERAGE_COUNTS] requested=%s selected=0 executed=0 completed=0" %
              len(requested_names))
        print("[AVBD_COVERAGE_MATRIX] package=%s config=%s solver=%s result=FAIL" %
              (args.package, args.config, args.solver))
        return avbd_gate.CONFIG_ERROR
    if args.list:
        for item in selected:
            print("%s\t%s\t%s\t%s" % (
                item.get("id"), item.get("snippet"), item.get("case"),
                expected_status_for(item, args.config)))
        print("[AVBD_COVERAGE_COUNTS] requested=%s selected=%s executed=0 completed=0" %
              (len(requested_names), len(selected)))
        return 0

    bin_dir = os.path.abspath(args.bin_dir or avbd_gate.default_bin_dir(ROOT, args.config))
    rows = []
    tool_ok = True
    for spec in selected:
        timeout = args.timeout if args.timeout is not None else float(spec.get("timeout", 300))
        expected_status = expected_status_for(spec, args.config)
        run = None
        if expected_status in ("planned", "invalid-sdk"):
            row = evaluate_run(spec, None, args.config, args.solver)
        elif spec.get("kind") == "smoke":
            run = run_smoke(ROOT, bin_dir, timeout)
            row = evaluate_run(spec, run, args.config, args.solver)
            sys.stdout.write((run.get("text") or "") + (("\n" if run.get("text") else "")))
        else:
            run = run_snippet(bin_dir, spec, args.solver, timeout)
            row = evaluate_run(spec, run, args.config, args.solver)
            sys.stdout.write((run.get("text") or "") + (("\n" if run.get("text") else "")))
        row_ok = not row.get("errors")
        if expected_status in ("planned", "invalid-sdk"):
            row_ok = True
        tool_ok = tool_ok and row_ok
        result_label = "PASS" if row_ok else "FAIL"
        print("[AVBD_COVERAGE] id=%s case=%s expected=%s observed=%s runner=%s" %
              (row.get("id"), row.get("case"), expected_status,
               row.get("observed_status"), result_label), flush=True)
        for item in row.get("errors") or []:
            print("[AVBD_COVERAGE_ERROR] id=%s error=%s" % (row.get("id"), item), flush=True)
        rows.append(row)

    counts = avbd_gate.support_counts(rows)
    if counts["executed"] == 0 and any(
            expected_status_for(spec, args.config) not in ("planned", "invalid-sdk")
            for spec in selected):
        print("[AVBD_COVERAGE_ERROR] empty execution set")
        tool_ok = False
    # Baseline mismatch is separate from measurement validity: a repaired
    # expected-FAIL case can be a fully validated physical PASS. Corrupt output,
    # incomplete frames, runtime failures and crashes still invalidate it.
    measurement_ok = all(
        row.get("physical_valid", False) if row.get("expected_status") in ("pass", "fail")
        else not row.get("errors") for row in rows)
    result, physical_result = matrix_outcome(counts, tool_ok,
                                             args.require_physical_pass, measurement_ok)
    print("[AVBD_COVERAGE_COUNTS] requested=%s selected=%s executed=%s completed=%s "
          "planned=%s unsupported=%s physical_pass=%s physical_fail=%s "
          "support_rate=%s/%s" % (
              counts["requested"], counts["selected"], counts["executed"],
              counts["completed"], counts["planned"], counts["unsupported"],
              counts["physical_pass"], counts["physical_fail"],
              counts["support_rate_numerator"], counts["support_rate_denominator"]))
    print("[AVBD_COVERAGE_MATRIX] package=%s config=%s solver=%s result=%s "
          "physical_result=%s" %
          (args.package, args.config, args.solver, result, physical_result))
    report = {
        "package": args.package,
        "config": args.config,
        "solver": args.solver,
        "bin_dir": bin_dir,
        "counts": counts,
        "result": result,
        "baseline_result": "PASS" if tool_ok else "FAIL",
        "physical_result": physical_result,
        "require_physical_pass": args.require_physical_pass,
        "binaries": hash_binaries(bin_dir, [
            os.path.join(ROOT, "tools", "avbd_smoke_new.exe"),
            os.path.join(ROOT, "tools", "avbd_smoke.exe"),
        ]),
        "cases": rows,
        "generated_unix": int(time.time()),
    }
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("[AVBD_COVERAGE_OUTPUT] %s" % os.path.abspath(args.output))
    if not tool_ok and not args.require_physical_pass:
        return avbd_gate.CONFIG_ERROR if counts["executed"] == 0 else avbd_gate.PHYSICAL_FAIL
    return 0 if result == "PASS" else avbd_gate.PHYSICAL_FAIL


if __name__ == "__main__":
    sys.exit(main())
