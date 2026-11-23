"""Quality must qualify each run; differing trajectories must not veto it."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import benchmark_avbd_native_parallel as native
import benchmark_avbd_volume as volume
import run_avbd_parallel_schedule as legacy
import run_avbd_common_step_probe as common

def lab_module(name):
    path = Path(__file__).resolve().parents[1] / "experiments/avbd_contact_lab" / (name + ".py")
    if not path.is_file():
        raise unittest.SkipTest("This check exercises the optional local AVBD contact lab")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_gate(**changes):
    gate = dict(schema="1", snippet="SnippetDeformableSurfaceAVBD", solver="avbd",
                case="free-flight", model="rigid_patch_joint_sheet", status="PASS",
                frames="60", completedFrames="60", nonFinite="0", fetchFailures="0",
                fatalErrors="0", dispatcherThreads="4", averageStepMs="50",
                trajectoryHash="first", maxPenetration="0")
    return dict(gate, **changes)


def line(gate):
    return "[AVBD_GATE] " + " ".join(f"{k}={v}" for k, v in gate.items())


class ParallelAcceptanceTests(unittest.TestCase):
    def test_observer_purity_checks_one_call_and_rejects_mutation(self):
        observation = lab_module("observation_checks")
        import numpy as np
        system = SimpleNamespace(bodies=[SimpleNamespace(pose=np.zeros(7), v=np.zeros(3), w=np.zeros(3))])
        audit = SimpleNamespace(begin=lambda s: None, accept=lambda s: None, finish=lambda s: None)
        counts = observation.guard_observer(audit)
        audit.begin(system);audit.accept(system);audit.finish(system)
        self.assertEqual(counts, dict(begin=1, accept=1, finish=1))
        def corrupt(s):
            s.bodies[0].pose[0] = 1
        bad = SimpleNamespace(begin=lambda s: None, accept=corrupt, finish=lambda s: None)
        observation.guard_observer(bad)
        with self.assertRaisesRegex(AssertionError, 'modified physical state'):
            bad.accept(system)

    def test_publication_quality_separate_from_stationarity_research(self):
        output = ("[AVBD_COMMON_FRAME] bdf=1 stationary=0 status=ACCEPTED poseForcePublications=1\n"
                  "[COMMON_PROGRAM_SCENE] fetchedFinite=PASS\n")
        self.assertTrue(common.evaluate_output(0, False, 0, output))
        self.assertFalse(common.evaluate_output(0, False, 0, output, require_stationary=True))
        for invalid in (output.replace("bdf=1", "bdf=2"), output.replace("fetchedFinite=PASS", "fetchedFinite=FAIL"),
                        output.replace("poseForcePublications=1", "poseForcePublications=0"),
                        output.replace("stationary=0", "stationary=0 bdf=1")):
            self.assertFalse(common.evaluate_output(0, False, 0, invalid))

    def test_scheduler_layout_and_failure_path_are_not_gates(self):
        output = ("[AVBD_COMMON_FRAME] status=ACCEPTED poseForcePublications=1 bdf=1 stationary=1\n"
                  "[COMMON_CHAIN] status=PASS\n[COMMON_PROGRAM_SCENE] fetchedFinite=PASS\n"
                  "[AVBD_PATCH_SWEEP] groups=5 colors=3 maxBlocks=6\n")
        self.assertTrue(common.evaluate_output(6, False, 0, output))
        rejected = ("[AVBD_COMMON_FRAME] status=REJECTED poseForcePublications=0 bdf=0\n"
                    "[COMMON_CONFLICT] unpublished=PASS\nreason=early_infeasibility_proof\n")
        self.assertTrue(common.evaluate_output(7, False, 0, rejected))
        self.assertFalse(common.evaluate_output(7, False, 0, rejected.replace("bdf=0", "bdf=1")))
        overlap = rejected + "np_geometry_or_initial_recovery_uncompiled\n"
        self.assertTrue(common.evaluate_output(0, True, 1, overlap))
        self.assertFalse(common.evaluate_output(0, True, -9, overlap))

    def test_asset_inputs_exclude_derived_diagnostics_but_keep_physics(self):
        capture = lab_module("capture_assets")
        before = dict(schema=1, dt=1/60, initialMechanicalEnergy=1,
                      actors=[dict(mass=2, pose=[0, 0, 0], comWorld=[0, 0, 0])],
                      beams=[dict(ids=[0, 1], stiffness=[100], coordinates=[0], jacobian=[1], elasticEnergy=0)],
                      tetrahedra=[dict(ids=[0, 1, 2, 3], energyScale=1, elasticEnergy=0)],
                      assets=[dict(actorCount=4, mass=8, proxyVolume=1)])
        after = json.loads(json.dumps(before))
        after['initialMechanicalEnergy'] += 1e-12
        after['beams'][0]['jacobian'][0] += 1e-12
        self.assertEqual(capture.authored_inputs(before), capture.authored_inputs(after))
        after['actors'][0]['mass'] += 0.1
        self.assertNotEqual(capture.authored_inputs(before), capture.authored_inputs(after))

    def test_native_independent_physics_not_trajectory_equality(self):
        for gate in (native_gate(), native_gate(trajectoryHash="different", maxPenetration="0.000001")):
            self.assertTrue(native.physical_pass(gate, 0, False, 60, "free-flight", "surface"))
        for change in (dict(status="FAIL"), dict(completedFrames="59"), dict(nonFinite="1"),
                       dict(fetchFailures="1"), dict(fatalErrors="1"), dict(solver="pgs"),
                       dict(model="legacy_fem"), dict(snippet="SnippetJointSoftBody")):
            with self.subTest(change=change):
                self.assertFalse(native.physical_pass(native_gate(**change), 0, False, 60, "free-flight", "surface"))
        self.assertFalse(native.physical_pass(native_gate(), 0, True, 60, "free-flight", "surface"))
        self.assertFalse(native.physical_pass(native_gate(), 1, False, 60, "free-flight", "surface"))

    def test_native_five_body_uses_its_actual_model_and_no_hash_gate(self):
        gate = native_gate(snippet="SnippetJointSoftBody", case="visual-showcase", frames="360",
                           completedFrames="360", model="convex_cell_joint_volume")
        gate.pop("trajectoryHash")
        self.assertTrue(native.physical_pass(gate, 0, False, 360, "visual-showcase", "volume"))
        self.assertFalse(native.physical_pass(dict(gate, model="rigid_cell_joint_volume"),
                                             0, False, 360, "visual-showcase", "volume"))

    def test_native_route_telemetry_and_fail_closed_logs(self):
        route = "[AVBD_SURFACE_ROUTE] model=joint_sheet commonStep=budget workers=4\n"
        schedule = "[AVBD_BOUNDED_SCHEDULE] waves=2 batches=4 tasks=3 sweeps=1\n"
        local = "[AVBD_LOCAL_BUDGET] budget=4 iterations=4 primalBlocks=2304\n"
        good = route + schedule + local + line(native_gate()) + "\n"
        cases = [(good, True), (good.replace("commonStep=budget", "commonStep=bounded"), False),
                 (good.replace("workers=4", "workers=8"), False),
                 (good.replace("dispatcherThreads=4", "dispatcherThreads=8"), False),
                 (good + line(native_gate()), False), (good.rstrip() + " status=PASS", False),
                 (good.replace("averageStepMs=50", "averageStepMs=nan"), False),
                 (good.replace("budget=4 iterations=4", "budget=32 iterations=32"), False),
                 (good.replace("iterations=4", "iterations=5"), False),
                 (good.replace(local, ""), False)]
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(output=Path(folder), diagnostics=True, snippet="surface",
                                      case="free-flight", frames=60, timeout=1)
            for output, expected in cases:
                with self.subTest(output=output):
                    def launch(*unused, stdout, **kwargs):
                        stdout.write(output.encode())
                        return mock.Mock(wait=mock.Mock(return_value=0))
                    with mock.patch.object(native.subprocess, "Popen", side_effect=launch), \
                            mock.patch.object(native, "process_cpu_seconds", return_value=0.5):
                        result = native.run(Path(folder) / "fake.exe", args, 4, 1)
                    self.assertEqual(result["physicalPass"], expected)
                    self.assertEqual(result["schedule"]["tasks"], 3)

    def run_legacy(self, status):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "physx/bin/win.x86_64.vc143.md/checked"
            binary.mkdir(parents=True)
            (binary / "SnippetDeformableVolumeAVBD_64.exe").write_bytes(b"fixture")

            def launch(command, stdout, **unused):
                self.assertIn("--particle3", command)
                workers = int(next(a.split("=")[1] for a in command if a.startswith("--dispatcher-threads=")))
                gate = native_gate(snippet="SnippetDeformableVolumeAVBD", case="visual-showcase",
                                   frames="360", completedFrames="360", status=status,
                                   trajectoryHash=str(workers) if status == "PASS" else "same_failed_run",
                                   maxPenetration=str(workers * 1e-7) if status == "PASS" else "0.01")
                gate.pop("fetchFailures")  # The legacy FEM schema omits this field.
                stats = f"[AVBD_SCHEDULE] workers={workers} peakTasks={workers if workers > 1 else 0}\n"
                stdout.write((stats * 360 + line(gate) + "\n").encode())
                return mock.Mock(wait=mock.Mock(return_value=0 if status == "PASS" else 1))

            argv = ["legacy", "--threads", "1", "2", "--output", str(root / "report")]
            with mock.patch.object(legacy, "ROOT", root), mock.patch("sys.argv", argv), \
                    mock.patch.object(legacy.subprocess, "Popen", side_effect=launch), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = legacy.main()
            return code, json.loads((root / "report/report.json").read_text())

    def test_legacy_different_passing_runs_qualify(self):
        code, report = self.run_legacy("PASS")
        self.assertEqual(code, 0)
        self.assertEqual(report["qualification"], "PASS")
        self.assertFalse(report["runs"][1]["equal"])

    def test_legacy_equal_failed_runs_never_qualify(self):
        code, report = self.run_legacy("FAIL")
        self.assertEqual(code, 1)
        self.assertTrue(report["runs"][1]["equal"])
        self.assertEqual(report["qualification"], "NOT_VERIFIED")

    def run_volume_compare(self, failed_candidate=False, expected_failure=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("baseline", "candidate"):
                (root / (name + ".exe")).write_bytes(name.encode())

            def launch(command, **unused):
                self.assertIn("--particle3", command)
                candidate = "candidate.exe" in command[0]
                status = "FAIL" if expected_failure or (candidate and failed_candidate) else "PASS"
                gate = native_gate(snippet="SnippetDeformableVolumeAVBD", case="ground", frames="180",
                                   completedFrames="180", status=status, trajectoryHash=str(candidate),
                                   maxPenetration="0.000001" if candidate else "0")
                gate.pop("fetchFailures")  # The legacy FEM schema omits this field.
                return subprocess.CompletedProcess(command, 0 if status == "PASS" else 1,
                                                   stdout=line(gate).encode(), stderr=b"")

            argv = ["compare", "--baseline-exe", str(root / "baseline.exe"), "--candidate-exe",
                    str(root / "candidate.exe"), "--cases", "ground", "--repeats", "1",
                    "--output", str(root / "result")]
            if expected_failure:
                argv += ["--expected-failures", "ground"]
            with mock.patch("sys.argv", argv), mock.patch.object(volume.platform, "platform", return_value="test"), \
                    mock.patch.object(volume.platform, "processor", return_value="test"), \
                    mock.patch.object(volume.subprocess, "run", side_effect=launch), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = volume.main()
            return code, json.loads((root / "result/report.json").read_text())

    def test_volume_compare_reports_differences_without_veto(self):
        code, report = self.run_volume_compare()
        self.assertEqual(code, 0)
        self.assertEqual(report["qualification"], "PASS")
        self.assertIn("trajectoryHash", report["cases"]["ground"]["runs"][0]["differences"])

    def test_volume_failed_candidate_remains_failed(self):
        code, report = self.run_volume_compare(failed_candidate=True)
        self.assertEqual(code, 1)
        self.assertEqual(report["qualification"], "NOT_VERIFIED")

    def test_expected_failure_comparison_is_not_quality_pass(self):
        code, report = self.run_volume_compare(expected_failure=True)
        self.assertEqual(code, 0)
        self.assertEqual(report["qualification"], "NOT_VERIFIED")


if __name__ == "__main__":
    unittest.main()
