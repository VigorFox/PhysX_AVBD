#!/usr/bin/env python3
"""Shared AVBD headless GATE parse, selection, and fail-closed counting.

Scheduling/parse failures must not become a physical PASS. Planned and
unsupported cells stay out of the support-rate numerator.
"""
from __future__ import print_function

import json
import math
import os
import re


CONFIG_ERROR = 2
PHYSICAL_FAIL = 1
OK = 0

FINITE_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")
NON_FINITE_TOKENS = ("nan", "+nan", "-nan", "inf", "+inf", "-inf", "1.#inf", "-1.#inf", "1.#qnan")
SDK_FATAL_RE = re.compile(
    r"(?im)^.*\(\d+\)\s*:\s*(?:invalid operation|invalid parameter|out of memory|internal error|abort)\s*:")


def parse_fields(line):
    fields = {}
    errors = []
    tokens = line.split()
    if not tokens or tokens[0] != "[AVBD_GATE]":
        errors.append("missing [AVBD_GATE] prefix")
        return fields, errors
    for token in tokens[1:]:
        if "=" not in token:
            errors.append("malformed token: %s" % token)
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append("duplicate key: %s" % key)
        fields[key] = value
    return fields, errors


def gate_lines(text):
    return [
        line.strip()
        for line in (text or "").splitlines()
        if line.startswith("[AVBD_GATE] ")
    ]


def is_finite_token(value):
    if value is None:
        return False
    lowered = value.strip().lower()
    if lowered in NON_FINITE_TOKENS:
        return False
    if not FINITE_RE.fullmatch(value.strip()):
        return False
    return math.isfinite(float(value))


def missing_required(fields, required):
    missing = []
    for key in required:
        if key not in fields or fields.get(key) in (None, ""):
            missing.append(key)
    return missing


def non_finite_numeric_fields(fields, numeric_keys):
    bad = []
    for key in numeric_keys:
        if key not in fields:
            continue
        finite = is_finite_token(fields[key])
        if not finite:
            bad.append(key)
    return bad


def parse_case_list(raw):
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_selection(raw, known, aliases=None):
    """Return (error, selected_names).

    selected_names is None when the caller should run the default/all set.
    Empty or unknown requests always error; they must not fall through to PASS.
    """
    aliases = aliases or {}
    text = (raw or "").strip()
    if text in aliases:
        return None, aliases[text]
    names = parse_case_list(text)
    if not names:
        return "empty case selection", []
    unknown = [name for name in names if name not in known]
    if unknown:
        return "unknown cases: %s" % ",".join(unknown), names
    duplicates = sorted(set(name for name in names if names.count(name) > 1))
    if duplicates:
        return "duplicate cases: %s" % ",".join(duplicates), names
    return None, names


def evaluate_gate_text(text, expected):
    """Evaluate one process output against an expected capability record.

    expected keys:
      case, expected_status (pass|fail|unsupported|error),
      expected_exit (optional int), required_fields, numeric_fields,
      capability (optional).
    Returns (ok, record_dict).
    """
    lines = gate_lines(text)
    errors = []
    fields = {}
    if len(lines) == 0:
        errors.append("missing GATE")
    elif len(lines) != 1:
        errors.append("gate count is %s, expected 1" % len(lines))
    else:
        fields, parse_errors = parse_fields(lines[0])
        errors.extend(parse_errors)
        # Older rigid headless protocols publish requestedFrames while the
        # deformable protocols publish frames.  They describe the same launch
        # request; normalize the former at the shared parser boundary and reject
        # an ambiguous record that publishes conflicting values.
        if "requestedFrames" in fields:
            if "frames" in fields and fields["frames"] != fields["requestedFrames"]:
                errors.append("frames=%r disagrees with requestedFrames=%r" %
                              (fields["frames"], fields["requestedFrames"]))
            elif "frames" not in fields:
                fields["frames"] = fields["requestedFrames"]
        want_case = expected.get("case")
        if want_case and fields.get("case") != want_case:
            errors.append("case=%r, expected %r" % (fields.get("case"), want_case))
        if fields.get("schema") != "1":
            errors.append("unsupported GATE schema=%r" % fields.get("schema"))
        for key in ("solver", "snippet", "frames", "completedFrames"):
            if key in expected and fields.get(key) != str(expected[key]):
                errors.append("%s=%r, expected %r" % (key, fields.get(key), str(expected[key])))
        required = list(expected.get("required_fields") or [])
        for key in ("schema", "snippet", "case", "status"):
            if key not in required:
                required.append(key)
        errors.extend("%s missing" % key for key in missing_required(fields, required))
        errors.extend("non-finite %s=%s" % (key, fields.get(key))
                      for key in non_finite_numeric_fields(fields, expected.get("numeric_fields") or []))

    expected_status = expected.get("expected_status", "pass")
    capability = fields.get("capability")
    status = fields.get("status")
    if expected_status == "unsupported":
        if capability != "UNSUPPORTED":
            errors.append("capability=%r, expected UNSUPPORTED" % capability)
        if status not in ("ERROR", "FAIL"):
            errors.append("unsupported status=%r" % status)
    elif expected_status == "pass":
        if status != "PASS":
            errors.append("status=%r, expected PASS" % status)
        if capability == "UNSUPPORTED":
            errors.append("supported case returned UNSUPPORTED")
        if expected.get("capability") and capability != expected.get("capability"):
            errors.append("capability=%r, expected %r" % (capability, expected.get("capability")))
    elif expected_status == "fail":
        if status != "FAIL":
            errors.append("status=%r, expected FAIL" % status)
        if fields.get("reason") not in (None, "physical_contract"):
            errors.append("expected physical failure, reason=%r" % fields.get("reason"))
    elif expected_status == "error":
        if status != "ERROR":
            errors.append("status=%r, expected ERROR" % status)

    if expected_status in ("pass", "fail"):
        if SDK_FATAL_RE.search(text or ""):
            errors.append("invalid physical run: SDK runtime error in process output")
        if fields.get("reason") in ("runtime_error", "initialization_or_fetch"):
            errors.append("invalid physical run: reason=%s" % fields["reason"])
        for key in ("fatalErrors", "nonFinite"):
            if key in fields and (not is_finite_token(fields[key]) or float(fields[key]) != 0):
                errors.append("invalid physical run: %s=%s" % (key, fields[key]))

    record = {
        "ok": not errors,
        "errors": errors,
        "fields": fields,
        "gate_count": len(lines),
        "status": status,
        "capability": capability,
        "expected_status": expected_status,
    }
    return not errors, record


def exit_matches_status(exit_code, expected_status, normalized_nonzero=False):
    """Only the documented launcher normalization may replace exits 2/3 by 1."""
    native = {"pass": 0, "fail": 1, "error": 2, "unsupported": 3}
    expected = native.get(expected_status)
    return expected is not None and (
        exit_code == expected or
        (normalized_nonzero and expected in (2, 3) and exit_code == 1))


def support_counts(rows):
    """rows: iterable of dicts with expected_status, observed_status, ran."""
    requested = len(rows)
    selected = sum(1 for row in rows if row.get("selected"))
    executed = sum(1 for row in rows if row.get("executed"))
    completed = sum(1 for row in rows if row.get("completed"))
    planned = sum(1 for row in rows if row.get("expected_status") == "planned")
    unsupported = sum(1 for row in rows if row.get("expected_status") == "unsupported")
    invalid_sdk = sum(1 for row in rows if row.get("expected_status") == "invalid-sdk")
    physical = [row for row in rows
                if row.get("expected_status") in ("pass", "fail")
                and row.get("selected") and row.get("executed")
                and row.get("completed")
                and row.get("physical_valid", not row.get("errors"))]
    physical_pass = sum(1 for row in physical if row.get("observed_status") == "pass")
    physical_fail = sum(1 for row in physical if row.get("observed_status") == "fail")
    # Support rate excludes planned / unsupported / invalid-sdk / unrun.
    denominator = physical_pass + physical_fail
    return {
        "requested": requested,
        "selected": selected,
        "executed": executed,
        "completed": completed,
        "planned": planned,
        "unsupported": unsupported,
        "invalid_sdk": invalid_sdk,
        "physical_pass": physical_pass,
        "physical_fail": physical_fail,
        "support_rate_numerator": physical_pass,
        "support_rate_denominator": denominator,
    }


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def cases_for_package(manifest, package):
    if package in ("all", None, ""):
        return list(manifest.get("cases") or [])
    selected = []
    for case in manifest.get("cases") or []:
        packages = case.get("packages") or []
        if package in packages:
            selected.append(case)
    return selected


def default_bin_dir(root, config):
    return os.path.join(root, "physx", "bin", "win.x86_64.vc143.md", config)


def file_sha256(path, chunk=1024 * 1024):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()
