#!/usr/bin/env python3
"""Fail-closed tests for AVBD coverage/migration runners.

These tests check scheduler and GATE-parse contracts. They do not grep
implementation source and they do not treat planned/unsupported as physical
support.
"""
from __future__ import print_function

import os
import subprocess
import sys
import unittest
from unittest import mock

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import avbd_gate  # noqa: E402
import run_avbd_contact_coverage as coverage  # noqa: E402

ROOT = os.path.dirname(TOOLS)
MIGRATIONS = os.path.join(TOOLS, "run_snippet_avbd_migrations.py")
COVERAGE = os.path.join(TOOLS, "run_avbd_contact_coverage.py")
MANIFEST = os.path.join(TOOLS, "avbd_contact_coverage_cases.json")


def run_py(script, extra, cwd=None):
    argv = [sys.executable, script] + extra
    return subprocess.run(argv, cwd=cwd or ROOT, text=True, capture_output=True)


class SelectionTests(unittest.TestCase):
    def test_c01_includes_chainmail_original_contracts(self):
        manifest = avbd_gate.load_manifest(MANIFEST)
        error, selected = coverage.select_cases(manifest, "C01", "all")
        self.assertIsNone(error)
        cases = {item["case"]: item for item in selected
                 if item.get("snippet") == "chainmail"}
        self.assertEqual(set(cases), {"topology", "sag", "impact", "projectile"})
        for name, frames in (("topology", 1), ("sag", 180),
                             ("impact", 600), ("projectile", 360)):
            self.assertEqual(cases[name]["frames"], frames)
            self.assertEqual(cases[name]["expected_status"], "pass")
            self.assertIn("maxJointError", cases[name]["required_fields"])
            self.assertIn("maxNetSpeed", cases[name]["numeric_fields"])
            if name in ("impact", "projectile"):
                self.assertIn("maxBallPlanarSeparation", cases[name]["required_fields"])
                self.assertIn("maxBallPlanarSeparation", cases[name]["numeric_fields"])

    def test_unknown_migration_case_is_config_error(self):
        proc = run_py(MIGRATIONS, ["--snippet", "volume", "--cases", "does-not-exist"])
        self.assertEqual(proc.returncode, avbd_gate.CONFIG_ERROR)
        self.assertIn("unknown cases", proc.stdout)
        self.assertIn("result=FAIL", proc.stdout)
        self.assertNotRegex(proc.stdout, r"\[AVBD_MIGRATION_MATRIX\] result=PASS")

    def test_empty_migration_case_is_config_error(self):
        proc = run_py(MIGRATIONS, ["--snippet", "volume", "--cases", ",,,"])
        self.assertEqual(proc.returncode, avbd_gate.CONFIG_ERROR)
        self.assertIn("empty case selection", proc.stdout)
        self.assertIn("result=FAIL", proc.stdout)

    def test_unknown_coverage_case_is_config_error(self):
        proc = run_py(COVERAGE, ["--package", "C00", "--cases", "not-a-real-case", "--list"])
        self.assertEqual(proc.returncode, avbd_gate.CONFIG_ERROR)
        self.assertIn("unknown cases", proc.stdout)
        self.assertIn("result=FAIL", proc.stdout)

    def test_empty_coverage_case_is_config_error(self):
        proc = run_py(COVERAGE, ["--package", "C00", "--cases", "   ", "--list"])
        self.assertEqual(proc.returncode, avbd_gate.CONFIG_ERROR)
        self.assertIn("empty case selection", proc.stdout)


class GateParseTests(unittest.TestCase):
    def test_chainmail_failure_stays_in_physical_denominator(self):
        spec = dict(snippet="chainmail", case="sag", frames=180,
                    expected_status="pass")
        output = ("[AVBD_GATE] schema=1 snippet=SnippetChainmail solver=avbd "
                  "case=sag frames=180 completedFrames=180 status=FAIL "
                  "reason=physical_contract detail=unbounded_or_constraint_error "
                  "nonFinite=0 fatalErrors=0 fetchFailures=0")
        row = coverage.evaluate_run(spec, dict(ran=True, exit=1, text=output),
                                    "checked", "avbd")
        self.assertEqual(row["observed_status"], "fail")
        self.assertTrue(row["errors"])
        self.assertTrue(row["physical_valid"])
        counts = avbd_gate.support_counts([row])
        self.assertEqual(counts["physical_fail"], 1)
        self.assertEqual(counts["support_rate_denominator"], 1)

    def test_identity_and_completion_match_request(self):
        base = dict(schema="1", snippet="SnippetHelloWorld", solver="avbd",
                    case="ground", frames="180", completedFrames="180", status="PASS")
        expected = dict(case="ground", snippet="SnippetHelloWorld", solver="avbd",
                        frames=180, completedFrames=180)
        for key, bad in (("schema", "2"), ("solver", "pgs"),
                         ("snippet", "Other"), ("frames", "1"), ("completedFrames", "179")):
            fields = dict(base)
            fields[key] = bad
            ok, _ = avbd_gate.evaluate_gate_text("[AVBD_GATE] " +
                " ".join("%s=%s" % item for item in fields.items()), expected)
            self.assertFalse(ok, key)

    def test_requested_frames_is_a_strict_frames_alias(self):
        expected = dict(case="ground", snippet="SnippetHelloWorld", solver="avbd",
                        frames=180, completedFrames=180)
        text = ("[AVBD_GATE] schema=1 snippet=SnippetHelloWorld solver=avbd "
                "case=ground requestedFrames=180 completedFrames=180 status=PASS")
        ok, record = avbd_gate.evaluate_gate_text(text, expected)
        self.assertTrue(ok, record)
        self.assertEqual(record["fields"]["frames"], "180")

        conflicting = text.replace("case=ground ",
                                   "case=ground frames=179 ")
        ok, record = avbd_gate.evaluate_gate_text(conflicting, expected)
        self.assertFalse(ok)
        self.assertTrue(any("disagrees" in error for error in record["errors"]))

    def test_unsupported_claim_cannot_be_a_physical_pass(self):
        ok, _ = avbd_gate.evaluate_gate_text(
            "[AVBD_GATE] schema=1 snippet=X case=ground status=PASS capability=UNSUPPORTED",
            {"case": "ground", "expected_status": "pass"})
        self.assertFalse(ok)

    def test_numeric_fields_require_finite_numbers(self):
        for value in ("garbage", "1e999", "-1e999", "NaN", "Infinity", "1.2.3"):
            with self.subTest(value=value):
                ok, _ = avbd_gate.evaluate_gate_text(
                    "[AVBD_GATE] schema=1 snippet=X case=ground status=PASS minY=" + value,
                    {"case": "ground", "numeric_fields": ["minY"]})
                self.assertFalse(ok)
        for value in ("0", "-1.25e-5", "3.40282347e+38"):
            self.assertTrue(avbd_gate.is_finite_token(value))

    def test_expected_failure_rejects_success_and_crash_exits(self):
        for exit_code in (0, 2, 3, 124, -1073741819):
            with self.subTest(exit_code=exit_code):
                row = coverage.evaluate_run(
                    {"case": "ground", "expected_status": "fail"},
                    {"ran": True, "exit": exit_code,
                     "text": "[AVBD_GATE] schema=1 snippet=X case=ground status=FAIL"},
                    "checked", "avbd")
                self.assertTrue(row["errors"])

    def test_documented_nonzero_launcher_normalization(self):
        for status, native in (("error", 2), ("unsupported", 3)):
            self.assertTrue(avbd_gate.exit_matches_status(native, status))
            self.assertFalse(avbd_gate.exit_matches_status(1, status))
            self.assertTrue(avbd_gate.exit_matches_status(1, status, True))
            for code in (0, 124, -1073741819):
                self.assertFalse(avbd_gate.exit_matches_status(code, status, True))

    def test_runtime_failure_is_not_expected_physical_failure(self):
        for extra in ("reason=runtime_error fatalErrors=1", "nonFinite=1", "fatalErrors=nan"):
            with self.subTest(extra=extra):
                ok, _ = avbd_gate.evaluate_gate_text(
                    "[AVBD_GATE] schema=1 snippet=X case=ground status=FAIL " + extra,
                    {"case": "ground", "expected_status": "fail"})
                self.assertFalse(ok)

    def test_missing_gate_fails(self):
        ok, record = avbd_gate.evaluate_gate_text("no gate here\n", {"case": "ground", "expected_status": "pass"})
        self.assertFalse(ok)
        self.assertIn("missing GATE", record["errors"])

    def test_sdk_runtime_error_overrides_a_success_gate(self):
        text = ("D:\\source\\DyAvbdDynamics.cpp (4571) : invalid operation : "
                "AVBD could not verify the local contact/material domain; step not published.\n"
                "[AVBD_GATE] schema=1 snippet=X case=visual-showcase status=PASS reason=none\n")
        row = coverage.evaluate_run(
            {"case": "visual-showcase", "expected_status": "pass"},
            {"ran": True, "exit": 0, "text": text}, "checked", "avbd")
        self.assertFalse(row["physical_valid"])
        self.assertTrue(any("runtime error" in error for error in row["errors"]))

    def test_runtime_reason_cannot_claim_success_without_a_counter(self):
        ok, _ = avbd_gate.evaluate_gate_text(
            "[AVBD_GATE] schema=1 snippet=X case=ground status=PASS reason=runtime_error",
            {"case": "ground", "expected_status": "pass"})
        self.assertFalse(ok)

    def test_duplicate_gate_fails(self):
        text = (
            "[AVBD_GATE] schema=1 snippet=X case=ground status=PASS capability=SUPPORTED\n"
            "[AVBD_GATE] schema=1 snippet=X case=ground status=PASS capability=SUPPORTED\n"
        )
        ok, record = avbd_gate.evaluate_gate_text(text, {"case": "ground", "expected_status": "pass"})
        self.assertFalse(ok)
        self.assertTrue(any("gate count" in item for item in record["errors"]))

    def test_status_fail_is_not_pass(self):
        text = "[AVBD_GATE] schema=1 snippet=X case=ground status=FAIL reason=physical_contract capability=SUPPORTED\n"
        ok, record = avbd_gate.evaluate_gate_text(text, {"case": "ground", "expected_status": "pass"})
        self.assertFalse(ok)
        self.assertEqual(record["status"], "FAIL")

    def test_expected_fail_accepts_fail_status(self):
        text = "[AVBD_GATE] schema=1 snippet=X case=rigid-squeeze status=FAIL reason=physical_contract capability=SUPPORTED minDetF=0.5\n"
        ok, record = avbd_gate.evaluate_gate_text(text, {
            "case": "rigid-squeeze",
            "expected_status": "fail",
            "required_fields": ["minDetF"],
            "numeric_fields": ["minDetF"],
        })
        self.assertTrue(ok)
        self.assertEqual(record["status"], "FAIL")

    def test_unsupported_uses_capability_not_exit_code(self):
        text = "[AVBD_GATE] schema=1 snippet=X case=skinning status=ERROR reason=x capability=UNSUPPORTED\n"
        ok, record = avbd_gate.evaluate_gate_text(text, {
            "case": "skinning",
            "expected_status": "unsupported",
        })
        self.assertTrue(ok)
        self.assertEqual(record["capability"], "UNSUPPORTED")

    def test_non_finite_numeric_fails(self):
        text = "[AVBD_GATE] schema=1 snippet=X case=ground status=PASS minY=nan minDetF=1\n"
        ok, record = avbd_gate.evaluate_gate_text(text, {
            "case": "ground",
            "expected_status": "pass",
            "numeric_fields": ["minY", "minDetF"],
        })
        self.assertFalse(ok)
        self.assertTrue(any("non-finite minY" in item for item in record["errors"]))

    def test_missing_required_field_fails(self):
        text = "[AVBD_GATE] schema=1 snippet=X case=ground status=PASS\n"
        ok, record = avbd_gate.evaluate_gate_text(text, {
            "case": "ground",
            "expected_status": "pass",
            "required_fields": ["firstBoxCoreHitFrame"],
        })
        self.assertFalse(ok)
        self.assertTrue(any("firstBoxCoreHitFrame missing" in item for item in record["errors"]))


class SupportRateTests(unittest.TestCase):
    def test_repaired_expected_failure_passes_only_physical_acceptance(self):
        row = coverage.evaluate_run({"case": "ground", "expected_status": "fail"},
            {"ran": True, "exit": 0, "text":
             "[AVBD_GATE] schema=1 snippet=X solver=avbd case=ground status=PASS reason=none"},
            "checked", "avbd")
        self.assertTrue(row["errors"])
        self.assertTrue(row["physical_valid"])
        counts = avbd_gate.support_counts([row])
        self.assertEqual(coverage.matrix_outcome(counts, False, False, True), ("FAIL", "PASS"))
        self.assertEqual(coverage.matrix_outcome(counts, False, True, True), ("PASS", "PASS"))
        self.assertEqual(coverage.matrix_outcome(counts, False, True, False),
                         ("FAIL", "NOT_VERIFIED"))

    def test_unexpected_physical_fail_is_still_counted(self):
        row = coverage.evaluate_run({"case": "ground", "expected_status": "pass"},
            {"ran": True, "exit": 1, "text":
             "[AVBD_GATE] schema=1 snippet=X solver=avbd case=ground status=FAIL reason=physical_contract"},
            "checked", "avbd")
        self.assertTrue(row["errors"])
        self.assertTrue(row["physical_valid"])
        self.assertEqual(avbd_gate.support_counts([row])["physical_fail"], 1)

    def test_expected_failure_cannot_pass_acceptance(self):
        counts = {"physical_pass": 3, "physical_fail": 1}
        self.assertEqual(coverage.matrix_outcome(counts, True), ("PASS", "FAIL"))
        self.assertEqual(coverage.matrix_outcome(counts, True, True), ("FAIL", "FAIL"))
        self.assertEqual(coverage.matrix_outcome({"physical_pass": 0, "physical_fail": 0},
                                               True, True), ("FAIL", "NOT_VERIFIED"))

    def test_bad_or_unexecuted_record_is_not_physical_support(self):
        for extra in ({"errors": ["crash"]}, {"executed": False}, {"completed": False}):
            row = dict(selected=True, executed=True, completed=True,
                       expected_status="pass", observed_status="pass")
            row.update(extra)
            self.assertEqual(avbd_gate.support_counts([row])["physical_pass"], 0)

    def test_planned_and_unsupported_are_not_support_numerator(self):
        rows = [
            {"selected": True, "expected_status": "pass", "observed_status": "pass", "executed": True, "completed": True},
            {"selected": True, "expected_status": "fail", "observed_status": "fail", "executed": True, "completed": True},
            {"selected": True, "expected_status": "unsupported", "observed_status": "unsupported", "executed": True, "completed": True},
            {"selected": True, "expected_status": "planned", "observed_status": "planned", "executed": False, "completed": False},
        ]
        counts = avbd_gate.support_counts(rows)
        self.assertEqual(counts["support_rate_numerator"], 1)
        self.assertEqual(counts["support_rate_denominator"], 2)
        self.assertEqual(counts["unsupported"], 1)
        self.assertEqual(counts["planned"], 1)

    def test_manifest_lists_planned_without_counting_them_as_pass(self):
        manifest = avbd_gate.load_manifest(MANIFEST)
        error, selected = coverage.select_cases(manifest, "C00", "all")
        self.assertIsNone(error)
        planned = [item for item in selected if item.get("expected_status") in ("planned", "invalid-sdk")]
        runnable = [item for item in selected if item.get("expected_status") not in ("planned", "invalid-sdk")]
        self.assertGreater(len(planned), 0)
        self.assertGreater(len(runnable), 0)
        self.assertTrue(any(item.get("id") == "volume-rigid-squeeze" for item in runnable))
        self.assertTrue(any(item.get("id") == "surface-static-box" for item in runnable))
        self.assertEqual(
            next(item for item in runnable if item["id"] == "volume-rigid-squeeze")["expected_status"],
            "pass",
        )


class InterruptedOutputTests(unittest.TestCase):
    def test_timeout_keeps_evidence_but_never_counts_as_pass(self):
        spec = dict(snippet="volume", case="visual-showcase", frames=360,
                    expected_status="pass")
        # Even a printed PASS cannot qualify a process that did not exit.
        output = b"[AVBD_GATE] schema=1 snippet=X case=visual-showcase status=PASS\n"
        for run in (lambda: coverage.run_snippet(ROOT, spec, "avbd", 600),
                    lambda: coverage.run_smoke(ROOT, ROOT, 600)):
            for stderr in (b"[AVBD_PROGRESS] frame=270\n", "fatal error\n", None):
                with self.subTest(stderr=stderr), \
                        mock.patch.object(coverage.os.path, "isfile", return_value=True), \
                        mock.patch.object(coverage.subprocess, "run", side_effect=
                            subprocess.TimeoutExpired("test", 600, output=output, stderr=stderr)):
                    result = run()
                self.assertIn("[AVBD_GATE]", result["text"])
                if stderr:
                    self.assertIn(stderr.decode() if isinstance(stderr, bytes) else stderr,
                                  result["text"])
                self.assertEqual(result["exit"], 124)
                self.assertEqual(result["timeout_seconds"], 600)
                self.assertGreaterEqual(result["elapsed_seconds"], 0)
                row = coverage.evaluate_run(spec, result, "checked", "avbd")
                self.assertTrue(row["timed_out"])
                self.assertFalse(row["completed"])
                self.assertFalse(row["physical_valid"])
                self.assertEqual(row["process_output"], result["text"])
                self.assertEqual(avbd_gate.support_counts([row])["physical_pass"], 0)

    def test_vertex_embedding_evidence_is_required(self):
        manifest = avbd_gate.load_manifest(MANIFEST)
        for name in ("volume-embedded-sync", "volume-visual-showcase"):
            spec = next(case for case in manifest["cases"] if case["id"] == name)
            fields = {name: "0" for name in spec["required_fields"]}
            fields.update(status="PASS", case=spec["case"], schema="1", snippet="X")
            del fields["embeddingMaxVertexError"]
            ok, record = avbd_gate.evaluate_gate_text("[AVBD_GATE] " +
                " ".join("%s=%s" % item for item in fields.items()), spec)
            self.assertFalse(ok)
            self.assertTrue(any("embeddingMaxVertexError missing" in error
                                for error in record["errors"]))

    def test_truncated_output_is_parse_failure(self):
        ok, record = avbd_gate.evaluate_gate_text("simulate started\n", {"case": "ground", "expected_status": "pass"})
        self.assertFalse(ok)
        self.assertIn("missing GATE", record["errors"])


if __name__ == "__main__":
    unittest.main()
