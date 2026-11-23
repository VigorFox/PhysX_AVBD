"""Fail-closed checks for replay evidence, separate from native physics tests."""
import struct
import io
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import run_avbd_cooked_mesh_replay as replay


class ReplayEvidenceTests(unittest.TestCase):
    def test_input_identity_excludes_derived_float_reductions(self):
        geometry = dict(simVertices="8", simTets="5", collisionVertices="8", collisionTets="5",
                        boundaryFaces="12", totalMass="1", totalVolume="1")
        original = dict(hashes={"0": {"simTopology": "same"}}, metrics={"0": geometry})
        reduced = dict(original, metrics={"0": dict(geometry, totalMass="1.0000001", totalVolume="0.9999999")})
        self.assertEqual(replay.mesh_identity(original, False), replay.mesh_identity(reduced, False))
        different_topology = dict(original, metrics={"0": dict(geometry, simTets="6")})
        self.assertNotEqual(replay.mesh_identity(original, False), replay.mesh_identity(different_topology, False))

    def test_fixture_rejects_truncation_corruption_and_trailing_bytes(self):
        payload = b"NXS\x01DVME" + struct.pack("<I", 4) + b"fixture bytes"
        header = replay.HEADER.pack(b"AVBDMV1\0", 1, 5, 123, replay.fnv(payload), len(payload), 0)
        original = header + payload
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cube.pxvol"
            path.write_bytes(original)
            self.assertEqual(replay.fixture(path)["streamVersion"], 4)
            for bad in (original[:20], original[:-1], original + b"x", original[:-1] + b"x"):
                with self.subTest(bad=bad):
                    path.write_bytes(bad)
                    with self.assertRaises(ValueError):
                        replay.fixture(path)
            # A matching envelope checksum cannot make an incomplete SDK header valid.
            payload = b"short"
            path.write_bytes(replay.HEADER.pack(b"AVBDMV1\0", 1, 5, 123,
                             replay.fnv(payload), len(payload), 0) + payload)
            with self.assertRaises(ValueError):
                replay.fixture(path)

    def test_non_finite_timeout_is_rejected_before_launch(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value), mock.patch("sys.argv", [
                    "replay", "--output-dir", "unused", "--timeout=" + value]), \
                    mock.patch("sys.stderr", new_callable=io.StringIO), \
                    mock.patch.object(replay.subprocess, "run") as launch:
                with self.assertRaises(SystemExit) as error:
                    replay.main()
                self.assertEqual(error.exception.code, 2)
                launch.assert_not_called()

    def test_missing_duplicate_or_unexpected_records_fail(self):
        valid = "[AVBD_MESH_HASH] volume=0 hash=a"
        self.assertIn("0", replay.records(valid, "[AVBD_MESH_HASH]", "volume", ["0"]))
        for bad in ("", valid + "\n" + valid, valid.replace("volume=0", "volume=1"),
                    valid + " volume=0"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    replay.records(bad, "[AVBD_MESH_HASH]", "volume", ["0"])

    def test_short_extended_budget_and_partial_config_runs_never_qualify(self):
        good = dict(physical_valid=True, observed_status="pass", errors=[], timed_out=False)
        spec = dict(frames=360, timeout=600)
        args = dict(checks_passed=True, runs=[good] * 3, configs=replay.CONFIGS,
                    frames=360, timeout=600, spec=spec)
        self.assertTrue(replay.qualification(**args))
        for override in (dict(frames=1), dict(timeout=601), dict(configs=["checked"]),
                         dict(runs=[]), dict(checks_passed=False)):
            self.assertFalse(replay.qualification(**dict(args, **override)))
        for bad in (dict(physical_valid=False), dict(observed_status="fail"),
                    dict(timed_out=True), dict(errors=["SDK runtime error"])):
            runs = [good, good, dict(good, **bad)]
            self.assertFalse(replay.qualification(**dict(args, runs=runs)))


if __name__ == "__main__":
    unittest.main()
