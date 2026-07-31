#!/usr/bin/env python3
"""Fail closed if Scene duplicates canonical soft-body particle ranges."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = (
    ROOT / "physx/source/simulationcontroller/src/ScScene.cpp"
)


def section(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    end_index = text.find(end, start_index + len(start))
    if start_index < 0 or end_index < 0:
        return ""
    return text[start_index:end_index]


def main() -> int:
    scene = SCENE_PATH.read_text(encoding="utf-8")
    entry = section(scene, "struct Entry", "struct StaticShapeEntry")
    errors: list[str] = []

    if not entry:
        errors.append("could not isolate AvbdCpuSoftScene::Entry")
    for declaration in (
        "PxU32\t\t\t\t\t\tparticleStart;",
        "PxU32\t\t\t\t\t\tparticleCount;",
    ):
        if declaration in entry:
            errors.append(
                "Scene Entry duplicates canonical compiled particle range "
                f"{declaration.strip()!r}"
            )

    for direct_access in (
        "entry.particleStart",
        "entry.particleCount",
        "entry->particleStart",
        "entry->particleCount",
        "remainingEntry.particleStart",
        "remainingEntry.particleCount",
    ):
        if direct_access in scene:
            errors.append(
                "Scene still consumes duplicated Entry range "
                f"{direct_access!r}"
            )

    required = (
        "PX_FORCE_INLINE PxU32 getParticleStart(",
        "PX_FORCE_INLINE PxU32 getParticleCount(",
        "const Entry& entry) const",
        "mBodies[entry.bodyIndex].compiled.particleStart",
        "mBodies[entry.bodyIndex].compiled.particleCount",
    )
    for token in required:
        if token not in scene:
            errors.append(
                "canonical compiled particle-range access lost "
                f"{token!r}"
            )

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_SOFT_SCENE_STORAGE_SOURCE_GATE_ERROR] "
                + error
            )
        print(
            "[AVBD_CPU_SOFT_SCENE_STORAGE_SOURCE_GATE] status=FAIL"
        )
        return 1

    print(
        "[AVBD_CPU_SOFT_SCENE_STORAGE_SOURCE_GATE] status=PASS "
        "particleRange=compiled-body-only entryShadow=none"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
