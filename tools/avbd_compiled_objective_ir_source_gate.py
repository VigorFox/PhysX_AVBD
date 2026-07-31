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
SOFT_COMPONENT = SOURCE_ROOT / "DyAvbdSoftBodyComponent.h"
SOFT_INTERNAL = SOURCE_ROOT / "DyAvbdSoftBody.h"


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
    soft_component = read(SOFT_COMPONENT)
    soft_internal = read(SOFT_INTERNAL)
    all_sources = "\n".join(
        (header, solver, joint_solver, soft_component, soft_internal)
    )

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
    require(
        errors,
        all_sources.count("enum class AvbdVelocityObjectiveOwner") == 1
        and "enum class AvbdVelocityObjectiveOwner" not in soft_component,
        "velocity owner must have exactly one canonical declaration",
    )
    require(
        errors,
        '#include "DyAvbdConstraint.h"' in soft_component,
        "soft component no longer consumes the canonical owner declaration",
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

    soft_owner_enum = re.compile(
        r"enum\s+class\s+AvbdSoftObjectiveOwner\s*:\s*PxU8"
        r"\s*\{\s*"
        r"eKINEMATIC_PIN_POSITION_AL\s*,\s*"
        r"eDEFORMABLE_KINEMATIC_POSITION_AL\s*,\s*"
        r"eKINEMATIC_ATTACHMENT_POSITION_AL\s*,\s*"
        r"eRIGID_ATTACHMENT_POSITION_AL\s*,\s*"
        r"eARTICULATION_ATTACHMENT_POSITION_AL\s*,\s*"
        r"eSOFT_PAIR_ATTACHMENT_POSITION_AL\s*,\s*"
        r"eUNSUPPORTED\s*\};",
        re.DOTALL,
    )
    require(
        errors,
        soft_owner_enum.search(soft_component) is not None,
        "soft objective owner enum is missing or no longer explicit",
    )
    required_soft_fragments = (
        "enum class AvbdSoftContactTargetKind",
        "enum class AvbdSoftPinTargetKind",
        "enum class AvbdSoftAttachmentTargetKind",
        "eWORLD_STATIC",
        "eWORLD_FIXED",
        "eKINEMATIC_RIGID",
        "eDEFORMABLE_SURFACE",
        "eRIGID_BODY",
        "struct AvbdCompiledSoftObjective",
        "compiledObjectives",
        "compileObjectiveProgram(",
        "isObjectiveProgramCurrent(",
        "avbdGetPinObjectiveOwner(",
        "eDEFORMABLE_KINEMATIC_POSITION_AL",
        "eKINEMATIC_ATTACHMENT_POSITION_AL",
        "eARTICULATION_ATTACHMENT_POSITION_AL",
        "eSOFT_PAIR_ATTACHMENT_POSITION_AL",
        "targetPoint",
        "previousWorldTarget",
        "avbdEvaluateContactParticleBlockAtSurfacePoint(",
        "avbdAddDynamicSoftRigidContactContributions_rigid(",
    )
    soft_sources = "\n".join(
        (soft_component, soft_internal, solver, joint_solver)
    )
    for fragment in required_soft_fragments:
        require(
            errors,
            fragment in soft_sources,
            f"soft compiled objective IR lost fragment {fragment!r}",
        )
    soft_finalize_mode = re.compile(
        r"enum\s+class\s+AvbdSoftComponentFinalizeMode\s*:\s*PxU8"
        r"\s*\{\s*"
        r"eMOMENTUM\s*,\s*"
        r"eKINEMATIC_CONTACT\s*,\s*"
        r"ePOSITION_OWNED\s*,\s*"
        r"eUNSUPPORTED\s*\};",
        re.DOTALL,
    )
    require(
        errors,
        soft_finalize_mode.search(soft_component) is not None,
        "soft component finalizer mode is missing or became combinable flags",
    )
    required_soft_velocity_fragments = (
        "AvbdVelocityObjectiveOwner velocityOwner;",
        "struct AvbdCompiledSoftVelocityObjective",
        "PxArray<AvbdCompiledSoftVelocityObjective>",
        "compiledVelocityObjectives",
        "compileVelocityObjectives",
        "avbdFinalizeSoftComponentVelocities(",
        "AvbdVelocityObjectiveOwner::ComponentFinalize",
        "AvbdSoftContactTargetKind::eKINEMATIC_RIGID",
        "kinematicSurfacePointPrevious",
        "previousSurfacePoint",
        "if(!positionOwned && !componentOwned)",
        "No solve stage",
        "is allowed to reinterpret target kind or flags later",
    )
    for fragment in required_soft_velocity_fragments:
        require(
            errors,
            fragment in soft_component,
            "soft velocity objective IR lost fragment "
            f"{fragment!r}",
        )
    minimal_velocity_ir = slice_between(
        errors,
        soft_component,
        "struct AvbdCompiledSoftVelocityObjective",
        "};",
        "minimal soft velocity objective IR",
    )
    for fragment in (
        "AvbdVelocityObjectiveOwner owner;",
        "AvbdSoftContactSource source;",
        "PxU32 bodyIndex;",
        "PxU32 particleIndex;",
        "PxU32 queryParticleIndices[3];",
        "PxReal queryWeights[3];",
        "PxVec3 normal;",
        "PxVec3 surfacePoint;",
        "PxVec3 previousSurfacePoint;",
    ):
        require(
            errors,
            fragment in minimal_velocity_ir,
            f"minimal soft velocity IR lost field {fragment!r}",
        )
    require(
        errors,
        all(
            token not in minimal_velocity_ir
            for token in (
                "AvbdSoftContact ",
                "AvbdSoftContactState",
                "normalLambda",
                "tangentLambda",
                "penalty",
                "frictionStick",
            )
        ),
        "compiled soft velocity IR carries position-AL/contact state",
    )
    require(
        errors,
        re.search(
            r"PxArray\s*<\s*AvbdSoftContact\s*>\s*"
            r"\w*[Vv]elocity",
            soft_component,
        )
        is None,
        "compiled velocity program regressed to an AvbdSoftContact array",
    )
    require(
        errors,
        "attachmentIndices" not in soft_sources
        and "pinIndices" not in soft_sources,
        "soft solve reintroduced owner-specific adjacency arrays",
    )
    require(
        errors,
        "geometry.rigidBodyIdx" not in soft_sources
        and ".geometry.rigidBodyIdx" not in soft_sources,
        "soft contact target ownership regressed to the overloaded "
        "rigidBodyIdx encoding",
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

    soft_particle_solve = slice_between(
        errors,
        joint_solver,
        "void AvbdSolver::solveSoftParticle(",
        "void AvbdSolver::updateSoftDual(",
        "soft particle solve",
    )
    require(
        errors,
        "compiledObjectives" in soft_particle_solve
        and "switch (objective.owner)" in soft_particle_solve,
        "soft particle primal must dispatch the compiled owner program",
    )
    require(
        errors,
        "avbdCollectSoftContactParticleIndices(" in joint_solver
        and "contactRef.jacobianScale" in soft_particle_solve
        and "avbdEvaluateContactParticleBlockAtSurfacePoint("
        in soft_particle_solve
        and "avbdEvaluateContactParticleBlock(" in soft_particle_solve,
        "low-level soft particle primal must consume every incident "
        "soft-contact block through the shared evaluator",
    )
    require(
        errors,
        "surfaceParticleIndices" in soft_component
        and "surfaceWeights" in soft_component
        and "hasDeformableSurfaceTarget()" in soft_component,
        "prepared soft-soft geometry lost its target triangle incidence",
    )
    required_soft_target_fragments = (
        "enum class AvbdSoftContactTargetKind",
        "eWORLD_STATIC",
        "eDEFORMABLE_SURFACE",
        "eRIGID_BODY",
        "targetKind",
        "targetIndex",
        "hasRigidBodyTarget()",
        "avbdAddDynamicSoftRigidContactContribution_rigid(",
    )
    for fragment in required_soft_target_fragments:
        require(
            errors,
            fragment in soft_sources,
            f"soft contact target IR lost fragment {fragment!r}",
        )
    require(
        errors,
        "geometry.rigidBodyIdx" not in soft_sources,
        "soft contact target type was re-encoded through rigidBodyIdx",
    )
    body_contact_rows = slice_between(
        errors,
        solver,
        "void AvbdSolver::accumulateBodyContactRows(",
        "void AvbdSolver::solveLocalSystem(",
        "rigid body contact rows",
    )
    require(
        errors,
        "avbdAddDynamicSoftRigidContactContribution_rigid("
        in body_contact_rows,
        "rigid local solve does not consume the dynamic rigid-soft block",
    )
    require(
        errors,
        "avbdGetRigidContactSurfacePoint(" in soft_particle_solve
        and "avbdEvaluateContactParticleBlockAtSurfacePoint("
        in soft_particle_solve,
        "soft particle primal does not evaluate the current rigid target pose",
    )
    soft_dual = slice_between(
        errors,
        joint_solver,
        "void AvbdSolver::updateSoftDual(",
        None,
        "soft dual update",
    )
    require(
        errors,
        "compiledObjectives" in soft_dual
        and "switch (objective.owner)" in soft_dual,
        "soft dual must dispatch the same compiled owner program",
    )
    require(
        errors,
        "avbdUpdateSoftContactDualAtSurfacePoint(" in soft_dual
        and "avbdGetRigidContactSurfacePoint(" in soft_dual,
        "dynamic rigid-soft dual does not consume the current rigid target",
    )
    require(
        errors,
        all(
            helper not in soft_sources
            for helper in (
                "avbdEvaluateAttachmentForceHessian_particle",
                "avbdAddAttachmentContribution_rigid",
                "avbdAddAttachmentContributions_rigid",
            )
        ),
        "rigid attachment objective regained a one-body primal helper",
    )
    coupled_attachment_solve = slice_between(
        errors,
        joint_solver,
        "void AvbdSolver::solveSoftRigidAttachmentsCoupled(",
        "void AvbdSolver::updateSoftDual(",
        "coupled rigid-soft attachment solve",
    )
    required_coupled_attachment_fragments = (
        "compiledObjectives",
        "AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL",
        "avbdEvaluateSoftRigidAttachmentCoupledStep(",
        "endpoint < objective.point.particleCount",
        "objective.point.particleIndices[endpoint]].position +=",
        "step.particleCorrections[endpoint]",
        "rigidBody.position += step.rigidLinearCorrection",
        "rigidBody.rotation =",
    )
    coupled_attachment_kernel = slice_between(
        errors,
        soft_internal,
        "PX_FORCE_INLINE bool "
        "avbdEvaluateSoftRigidAttachmentCoupledStep(",
        "// =============================================================================\n"
        "// Internal-only: rigid-soft contact detection stub",
        "coupled rigid-soft attachment kernel",
    )
    required_coupled_kernel_fragments = (
        "avbdGetSoftPointPosition(point, particles) - worldAnchor",
        "avbdGetSoftPointInverseMass(",
        "rigidLinearInverse -",
        "skew * rigidAngularInverse * skew",
        "step.particleCorrections[i]",
        "step.rigidLinearCorrection",
        "step.rigidAngularCorrection",
    )
    for fragment in required_coupled_attachment_fragments:
        require(
            errors,
            fragment in coupled_attachment_solve,
            "coupled rigid-soft attachment owner lost fragment "
            f"{fragment!r}",
        )
    for fragment in required_coupled_kernel_fragments:
        require(
            errors,
            fragment in coupled_attachment_kernel,
            "coupled rigid-soft attachment kernel lost fragment "
            f"{fragment!r}",
        )
    require(
        errors,
        joint_solver.count("solveSoftRigidAttachmentsCoupled(") == 2,
        "coupled rigid-soft attachment block must have one declaration call "
        "and one definition",
    )
    require(
        errors,
        "numSoftContacts == 0 || softContacts" in solver,
        "objective-only soft islands no longer route through solveWithJoints",
    )
    require(
        errors,
        "positionOwnedAngularBodies" in joint_solver
        and "AvbdSoftObjectiveOwner::"
        in joint_solver
        and "eRIGID_ATTACHMENT_POSITION_AL" in joint_solver
        and "positionOwnedAngularBodies" in solver
        and "unconstrainedAngularMotion" in solver,
        "position-owned rigid attachment lost angular derivative closure",
    )
    require(
        errors,
        "avbdAddDynamicSoftRigidContactContribution_rigid("
        in solver,
        "rigid local solve does not consume dynamic rigid-soft contacts",
    )
    require(
        errors,
        "targetKind == AvbdSoftContactTargetKind::eRIGID_BODY"
        in soft_component
        and "This Scene-external component has no rigid 6x6 block"
        in soft_component,
        "Scene-external component no longer fails closed on rigid targets",
    )
    require(
        errors,
        "AVBD.softBodyLevel6x6" not in iterative_solve
        and "bodyComPred" not in iterative_solve
        and "bodyAccumTheta" not in iterative_solve,
        "soft contact objective is duplicated by the legacy body-level 6x6 "
        "aggregate owner",
    )
    soft_component_step = slice_between(
        errors,
        soft_component,
        "inline void avbdStepSoftBodies(",
        "} // namespace Dy",
        "Scene-external soft component step",
    )
    required_soft_convergence_fragments = (
        "AvbdSoftSweepConvergenceObservation",
        "AvbdSoftResidualConvergenceTracker",
        "avbdLimitTetDisplacementObserved(",
        "avbdLimitTetDisplacementFromLinearizations(",
        "avbdEvaluateNeoHookeanForceHessianPrepared(",
        "avbdSolveSymmetric33(",
        "AvbdSoftResidualConvergenceTracker residualConvergence(\n"
        "\t\t\t1e-8f, 2)",
        "residualConvergence.observe(sweepObservation)",
        "unsafeAppliedConvergenceCandidates",
        "if(residualPolicyConverged)",
        "residualConvergedOuterIterations++",
    )
    for fragment in required_soft_convergence_fragments:
        require(
            errors,
            fragment in soft_component,
            "soft strict residual authority lost fragment "
            f"{fragment!r}",
        )
    required_self_ogc_fragments = (
        "inline void avbdComputeSafetyBounds(",
        "PxArray<PxReal> vertexMinimums;",
        "PxArray<PxReal> triangleMinimums;",
        "PxArray<PxReal> edgeMinimums;",
        "avbdClosestPointsOnSegments(",
        "inline void avbdDetectSelfCollisionOGC(",
        "queryParticleIndices[1] = q1;",
        "previousQueryClosest - previousTargetClosest",
        "PxArray<PxReal> selfCollisionSafetyBounds;",
        "selfCollisionSafetyBounds[pi]",
        "avbdTruncateDisplacement(",
    )
    for fragment in required_self_ogc_fragments:
        require(
            errors,
            fragment in soft_component,
            "full self-OGC path lost fragment "
            f"{fragment!r}",
        )
    applied_candidate_block = slice_between(
        errors,
        soft_component_step,
        "if(appliedDisplacementConverged &&",
        "if(residualPolicyConverged)",
        "soft applied-displacement convergence candidate",
    )
    require(
        errors,
        "break;" not in applied_candidate_block,
        "soft component regressed to applied-displacement early-stop "
        "authority",
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
        "softProgram=1 sharedOrdinaryCompiler=1 "
        "runtimeSourceFlagDispatch=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
