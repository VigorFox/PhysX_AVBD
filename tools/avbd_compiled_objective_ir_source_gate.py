#!/usr/bin/env python3
"""Fail closed when AVBD objective ownership drifts back to runtime flags."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "physx/source/lowleveldynamics/src"
HEADER = SOURCE_ROOT / "DyAvbdConstraint.h"
SOLVER = SOURCE_ROOT / "DyAvbdSolver.cpp"
JOINT_SOLVER = SOURCE_ROOT / "DyAvbdSolverJointPath.cpp"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require(
    errors: list[str], condition: bool, description: str
) -> None:
    if not condition:
        errors.append(description)


def slice_between(
    errors: list[str],
    text: str,
    start: str,
    end: str | None,
    description: str,
) -> str:
    start_index = text.find(start)
    if start_index < 0:
        errors.append(f"{description}: missing start marker {start!r}")
        return ""
    if end is None:
        return text[start_index:]
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        errors.append(f"{description}: missing end marker {end!r}")
        return ""
    return text[start_index:end_index]


def main() -> int:
    errors: list[str] = []
    header = read(HEADER)
    solver = read(SOLVER)
    joint_solver = read(JOINT_SOLVER)
    all_sources = "\n".join((header, solver, joint_solver))

    owner_enum = re.compile(
        r"enum\s+class\s+AvbdVelocityObjectiveOwner"
        r"\s*:\s*physx::PxU8\s*\{\s*"
        r"PositionAL\s*,\s*"
        r"PointFinalize\s*,\s*"
        r"ManifoldFinalize\s*,\s*"
        r"ComponentFinalize\s*,\s*"
        r"JointFinalize\s*,\s*"
        r"Unsupported\s*\};",
        re.DOTALL,
    )
    require(
        errors,
        owner_enum.search(header) is not None,
        "typed owner enum is missing or is no longer single-valued",
    )

    required_header_fragments = (
        "struct AvbdCompiledVelocityObjective",
        "struct AvbdCompiledContactObjectiveProgram",
        "struct AvbdCompiledJointObjective",
        "struct AvbdCompiledJointObjectiveProgram",
        "findAvbdContactSourceObjective(",
        "findAvbdCompleteManifoldObjective(",
        "findAvbdJointObjectiveForSourceRow(",
        "(compiled.sourceSlotMask & sourceSlotMask) != 0",
        "(compiled.sourceRowMask & sourceRowMask) != 0",
    )
    for fragment in required_header_fragments:
        require(
            errors,
            fragment in header,
            f"compiled objective IR lost required fragment {fragment!r}",
        )

    forbidden_transient_owners = (
        "findAvbdVelocityObjectiveByOwner",
        "eSAME_ARTICULATION_EXTERNAL_SPHERICAL",
        "bodyResponseScale",
        "eCOUPLED_LINEAR_DRIVE_ACTIVE",
        "eLINEAR_POSITION_DRIVE_ACTIVE",
        "eANGULAR_AXIS_VELOCITY_DRIVE_ACTIVE",
        "eSLERP_VELOCITY_DRIVE_ACTIVE",
        "eANGULAR_AXIS_POSITION_DRIVE_ACTIVE",
        "eSLERP_POSITION_DRIVE_ACTIVE",
        "eCOUPLED_ANGULAR_POSITION_DRIVE_ACTIVE",
        "eCOUPLED_SPATIAL_TENDON_ACTIVE",
        "eNATIVE_PASSIVE_REACTION_ACTIVE",
        "eCOUPLED_LINEAR_POSITION_DRIVE_ACTIVE",
        "eVELOCITY_TANGENT_TARGET_OWNER",
        "eVELOCITY_TANGENT_TARGET_MANIFOLD_OWNER",
        "eDEFORMABLE_POSITION_TANGENT_OWNER",
        "eVELOCITY_PASSIVE_FRICTION_MANIFOLD_OWNER",
        "eVELOCITY_PASSIVE_FRICTION_COMPONENT_OWNER",
    )
    for token in forbidden_transient_owners:
        require(
            errors,
            token not in all_sources,
            f"runtime/transient ownership token was reintroduced: {token}",
        )

    require(
        errors,
        header.count("compileAvbdOrdinaryRigidContactObjectives(") == 1,
        "ordinary rigid-contact compiler must have exactly one definition",
    )
    require(
        errors,
        solver.count("compileAvbdOrdinaryRigidContactObjectives(") == 1,
        "contact-only path must call the shared ordinary compiler once",
    )
    require(
        errors,
        joint_solver.count(
            "compileAvbdOrdinaryRigidContactObjectives("
        )
        == 1,
        "joint path must call the shared ordinary compiler once",
    )

    local_solve = slice_between(
        errors,
        joint_solver,
        "void AvbdSolver::solveLocalSystemWithJoints(",
        "// Solver with Joint Constraints",
        "local joint solve",
    )
    require(
        errors,
        "compiledConeObjective" in local_solve
        and "eJOINT_SOURCE_ANGULAR_CONE" in local_solve,
        "local joint solve must route the cone through compiled source rows",
    )
    require(
        errors,
        "eD6_LEGACY_CONE_LIMIT_ACTIVE" not in local_solve
        and "eSPHERICAL_ELLIPTICAL_CONE_LIMIT_ACTIVE"
        not in local_solve,
        "local joint solve must not select cone ownership from source flags",
    )

    iterative_solve = slice_between(
        errors,
        joint_solver,
        "// Stage 5: Main solver loop",
        None,
        "iterative joint solve",
    )
    require(
        errors,
        "compiledConeObjective" in iterative_solve
        and "eJOINT_SOURCE_ANGULAR_CONE" in iterative_solve,
        "iterative joint solve must route the cone through compiled IR",
    )
    require(
        errors,
        "sourceFlags" not in iterative_solve,
        "main iterative/finalize solve stages must not read sourceFlags",
    )

    locked_projection = slice_between(
        errors,
        solver,
        "static void projectBodyStaticLockedD6LinearVelocities(",
        "// Suppress pose-solve bounce only on fast normal approach",
        "body-static locked-D6 derivative closure",
    )
    require(
        errors,
        "findAvbdJointObjectiveForSourceRow(" in locked_projection
        and "AvbdVelocityObjectiveOwner::PositionAL"
        in locked_projection,
        "locked-D6 derivative closure must consume exact compiled rows",
    )
    require(
        errors,
        "isLinearDriveEnabled" not in locked_projection
        and "driveFlags" not in locked_projection,
        "locked-D6 derivative closure must not infer ownership from drives",
    )

    if errors:
        for error in errors:
            print(f"[avbd:compiled-objective-ir-source] FAIL {error}")
        return 1

    print(
        "[avbd:compiled-objective-ir-source] PASS "
        "typedOwner=1 contactProgram=1 jointProgram=1 "
        "sharedOrdinaryCompiler=1 runtimeSourceFlagDispatch=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
