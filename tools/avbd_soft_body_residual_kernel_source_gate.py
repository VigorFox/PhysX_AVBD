#!/usr/bin/env python3
"""Fail closed if CPU AVBD soft residual authority or tet kernel regresses."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "physx/source/lowleveldynamics/src"
COMPONENT = SOURCE_ROOT / "DyAvbdSoftBodyComponent.h"
JOINT_PATH = SOURCE_ROOT / "DyAvbdSolverJointPath.cpp"
TEST = ROOT / "physx/snippets/snippetsoftbodyavbd/SnippetSoftBodyAVBD.cpp"
RUNNER = ROOT / "tools/run_snippet_soft_body_avbd_headless.py"


def require(errors: list[str], condition: bool, description: str) -> None:
    if not condition:
        errors.append(description)


def main() -> int:
    errors: list[str] = []
    component = COMPONENT.read_text(encoding="utf-8")
    joint_path = JOINT_PATH.read_text(encoding="utf-8")
    test = TEST.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    residual_fragments = (
        "struct AvbdSoftTetDisplacementLimitResult",
        "struct AvbdSoftSweepConvergenceObservation",
        "maxLocalSolveDisplacementSq",
        "positiveJRejectedSteps",
        "isResidualConverged(",
        "residualConvergedOuterIterations",
        "unsafeAppliedConvergenceCandidates",
    )
    for fragment in residual_fragments:
        require(
            errors,
            fragment in component,
            f"soft residual authority lost fragment {fragment!r}",
        )
    require(
        errors,
        "if (maxDxSq < 1e-12f)" not in component
        and "if(maxDxSq < 1e-12f)" not in component,
        "soft solve regressed to post-limiter displacement termination",
    )

    kernel_fragments = (
        "deformationGradientWeights[4]",
        "shapeGradientNormSq[4]",
        "inverseRestDeterminant",
        "avbdEvaluateNeoHookeanForceHessianPrepared(",
        "neoHookeanAlpha",
        "currentFaceGradient",
        "AvbdTetVertexLinearization",
        "avbdLimitTetDisplacementFromLinearizations(",
        "tetLinearizationCacheFallbackParticleSteps",
        "avbdSolveSymmetric33(",
    )
    for fragment in kernel_fragments:
        require(
            errors,
            fragment in component,
            f"prepared tet kernel lost fragment {fragment!r}",
        )
    require(
        errors,
        "avbdEvaluateNeoHookeanForceHessianPrepared(" in joint_path
        and "sb.material.neoHookeanAlpha" in joint_path,
        "Scene low-level soft path no longer consumes the prepared tet kernel",
    )

    prepared_start = component.find(
        "avbdEvaluateNeoHookeanForceHessianPrepared("
    )
    wrapper_start = component.find(
        "avbdEvaluateNeoHookeanForceHessian(", prepared_start + 1
    )
    prepared_body = (
        component[prepared_start:wrapper_start]
        if prepared_start >= 0 and wrapper_start > prepared_start
        else ""
    )
    require(
        errors,
        "PxMat33 F = Ds * tet.DmInv" not in prepared_body
        and "PxMat33 F=Ds*tet.DmInv" not in prepared_body,
        "prepared tet kernel rebuilt the generic deformation matrix",
    )

    require(
        errors,
        "--- Test 25: Soft Sweep Residual Authority ---" in test,
        "Test25 residual-authority counterexample is missing",
    )
    require(
        errors,
        "--- Test 26: Neo-Hookean Kernel Equivalence ---" in test,
        "Test26 tet-kernel equivalence gate is missing",
    )
    require(
        errors,
        "--- Test 27: Positive-J Limiter Kernel Equivalence ---" in test,
        "Test27 positive-J limiter equivalence gate is missing",
    )
    require(
        errors,
        "--- Test 28: Residual Convergence Tracker ---" in test,
        "Test28 residual convergence state-machine gate is missing",
    )
    require(
        errors,
        "--- Test 29: Symmetric Particle Block Solve ---" in test,
        "Test29 symmetric particle-block solve gate is missing",
    )
    require(
        errors,
        "H.getInverse() * f" not in component
        and "H3.getInverse() * f3" not in joint_path,
        "soft particle solve regressed to materializing a full 3x3 inverse",
    )
    require(
        errors,
        "AvbdSoftResidualConvergenceTracker residualConvergence("
        in component
        and "1e-8f, 2" in component,
        "active residual policy is no longer 1e-4 for two sweeps",
    )
    require(
        errors,
        "currentDetF + determinantGradient.dot(displacement)"
        in component,
        "positive-J limiter rebuilt current/proposed deformation matrices",
    )
    require(
        errors,
        "tuple(range(1, 37))" in runner
        and "choices=range(1, 37)" in runner,
        "soft-body headless runner no longer requires Tests 1..36",
    )

    if errors:
        for error in errors:
            print(f"[AVBD_SOFT_RESIDUAL_KERNEL_SOURCE_GATE_ERROR] {error}")
        print("[AVBD_SOFT_RESIDUAL_KERNEL_SOURCE_GATE] status=FAIL")
        return 1
    print(
        "[AVBD_SOFT_RESIDUAL_KERNEL_SOURCE_GATE] "
        "status=PASS tests=25,26,27,28,29 "
        "convergence=localSolveResidualConsecutive "
        "tetKernel=prepared"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
