#!/usr/bin/env python3
"""Run the CPU AVBD counterparts of the remaining FEM feature snippets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)

DEMOS = {
    "volume-attachment": (
        "SnippetDeformableVolumeAttachmentAVBD_64.exe",
        "SnippetDeformableVolumeAttachmentAVBD",
        "scene-volume-rigid-attachment",
    ),
    "volume-kinematic": (
        "SnippetDeformableVolumeKinematicAVBD_64.exe",
        "SnippetDeformableVolumeKinematicAVBD",
        "scene-volume-partial-kinematic-target",
    ),
    "volume-skinning": (
        "SnippetDeformableVolumeSkinningAVBD_64.exe",
        "SnippetDeformableVolumeSkinningAVBD",
        "scene-volume-skinning",
    ),
    "surface-skinning": (
        "SnippetDeformableSurfaceSkinningAVBD_64.exe",
        "SnippetDeformableSurfaceSkinningAVBD",
        "surface-skinning",
    ),
}


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def run_demo(
    name: str,
    bin_dir: Path,
    frames: int,
    execution: str,
    timeout: float,
) -> bool:
    executable_name, snippet_name, case_name = DEMOS[name]
    executable = bin_dir / executable_name
    if not executable.is_file():
        print(f"[FAIL] {name}: executable not found: {executable}")
        return False

    argv = [
        str(executable),
        "--headless",
        "--solver=avbd",
        f"--case={case_name}",
        f"--execution={execution}",
        f"--frames={frames}",
        "--dt=0.0166666675",
        "--dispatcher-threads=2",
        "--seed=1",
    ]
    env = os.environ.copy()
    env["PHYSX_SNIPPET_HEADLESS"] = "1"
    result = run_headless_process(
        argv, cwd=bin_dir, env=env, timeout_seconds=timeout
    )
    combined = result.stdout
    if result.stderr:
        combined += ("\n" if combined else "") + result.stderr
    gate_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_GATE] ")
    ]
    skinning_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CPU_SKINNING] ")
    ]

    errors: list[str] = []
    if result.timed_out:
        errors.append("timed out")
    if result.visible_window_detected:
        errors.append("visible window detected")
    if result.returncode != 0:
        errors.append(f"exit code {result.returncode}")
    if len(gate_lines) != 1:
        errors.append(f"gate count {len(gate_lines)}")
    else:
        fields = parse_fields(gate_lines[0])
        expected = {
            "schema": "1",
            "snippet": snippet_name,
            "case": case_name,
            "solver": "avbd",
            "status": "PASS",
            "cleanupComplete": "1",
            "fatalErrors": "0",
        }
        # Surface uses result=PASS; Volume uses status=PASS.
        if name == "surface-skinning":
            expected.pop("status")
            expected["result"] = "PASS"
        for key, value in expected.items():
            if fields.get(key) != value:
                errors.append(
                    f"{key}={fields.get(key)!r}, expected {value!r}"
                )
    if name.endswith("skinning"):
        if len(skinning_lines) != 1:
            errors.append(f"skinning gate count {len(skinning_lines)}")
        else:
            skinning = parse_fields(skinning_lines[0])
            if skinning.get("status") != "PASS":
                errors.append("skinning status is not PASS")
            try:
                vertices = int(skinning.get("vertices", "0"))
                triangles = int(skinning.get("triangles", "0"))
                evaluated = int(
                    skinning.get("evaluatedFrames", "0")
                )
                finite = int(skinning.get("finiteFrames", "0"))
            except ValueError:
                errors.append("invalid numeric skinning fields")
            else:
                if vertices <= 4 or triangles <= 4:
                    errors.append("skinned mesh is unexpectedly small")
                if evaluated != frames or finite != frames:
                    errors.append("skinning frame coverage mismatch")

    if errors:
        print(f"[FAIL] {name}: " + "; ".join(errors))
        if combined:
            print(combined.rstrip())
        return False
    print(f"[PASS] {name}: {frames} frames ({execution})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bin-dir", type=Path, default=DEFAULT_BIN_DIR
    )
    parser.add_argument(
        "--demo", choices=tuple(DEMOS), action="append"
    )
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument(
        "--execution", choices=("parallel", "sequential"),
        default="parallel",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")

    selected = args.demo or list(DEMOS)
    passed = True
    for name in selected:
        passed = (
            run_demo(
                name,
                args.bin_dir.resolve(),
                args.frames,
                args.execution,
                args.timeout,
            )
            and passed
        )
    print(
        f"[AVBD_FEATURE_DEMOS] passed={int(passed)} "
        f"demos={len(selected)}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
