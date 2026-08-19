#!/usr/bin/env python3
"""Lock position-AL contacts out of component momentum finalization."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = (
    ROOT
    / "physx/source/lowleveldynamics/src/DyAvbdSoftBodyComponent.h"
)


def main() -> int:
    source = COMPONENT.read_text(encoding="utf-8")
    start_marker = (
        "PX_NOINLINE inline void "
        "avbdFinalizeSoftComponentVelocities("
    )
    end_marker = "PX_FORCE_INLINE void avbdUpdateSoftContactDual("
    start = source.find(start_marker)
    end = source.find(end_marker, start + len(start_marker))
    errors: list[str] = []
    if start < 0:
        errors.append("missing soft component velocity finalizer")
        finalizer = ""
    elif end < 0:
        errors.append("missing finalizer end marker")
        finalizer = ""
    else:
        finalizer = source[start:end]

    ownership_guard = re.compile(
        r"if\s*\(\s*!target\.valid\s*\|\|\s*"
        r"mode\s*==\s*AvbdSoftComponentFinalizeMode::"
        r"ePOSITION_OWNED\s*\|\|\s*"
        r"mode\s*==\s*AvbdSoftComponentFinalizeMode::"
        r"eUNSUPPORTED\s*\)\s*continue\s*;",
        re.DOTALL,
    )
    if finalizer and ownership_guard.search(finalizer) is None:
        errors.append(
            "position-owned contacts can enter component momentum "
            "finalization"
        )

    forbidden_switches = (
        "_".join(
            ("PHYSX", "AVBD", "POSITION", "OWNED", "COMPONENT", "FINALIZER")
        ),
        "_".join(
            ("PHYSX", "AVBD", "ENABLE", "POSITION", "OWNED", "FINALIZER")
        ),
        "avbdUsePositionOwnedComponentFinalizer",
        "avbdEnablePositionOwnedComponentFinalizer",
    )
    for token in forbidden_switches:
        if token in source:
            errors.append(f"runtime bypass switch was introduced: {token}")

    if errors:
        for error in errors:
            print(
                "[avbd:position-owned-component-finalizer] FAIL "
                f"{error}"
            )
        return 1
    print(
        "[avbd:position-owned-component-finalizer] PASS "
        "position-AL contacts remain local to the particle/contact solve"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
