#!/usr/bin/env python3
"""Fail closed if the rigid AVBD scene-iteration contract regresses."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAG = "AVBD_SCENE_ITERATION_POLICY_GATE"


def read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise AssertionError(f"missing file: {relative}")
    return path.read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def code_only(text: str) -> str:
    """Remove comments so a retired contract cannot satisfy a source check."""

    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\r\n]*", "", text)


def struct_body(text: str, name: str) -> str:
    match = re.search(rf"\bstruct\s+{re.escape(name)}\s*\{{", text)
    require(match is not None, f"missing struct {name}")
    assert match is not None

    opening = text.find("{", match.start())
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1:index]
    raise AssertionError(f"unterminated struct {name}")


def require_scene_desc_contract() -> None:
    scene_desc_raw = read("physx/include/PxSceneDesc.h")
    scene_desc = code_only(scene_desc_raw)

    require(
        re.search(r"\bPxU32\s+avbdIterations\s*;", scene_desc) is not None,
        "PxSceneDesc.h: missing PxSceneDesc::avbdIterations",
    )
    require(
        re.search(r"\bavbdIterations\s*\(\s*4[uU]?\s*\)", scene_desc)
        is not None,
        "PxSceneDesc.h: avbdIterations must default to 4",
    )
    require(
        re.search(
            r"if\s*\(\s*"
            r"(?=[^)]*\bavbdIterations\s*<\s*1[uU]?\b)"
            r"(?=[^)]*\bavbdIterations\s*>\s*255[uU]?\b)"
            r"[^)]*\)",
            scene_desc,
        )
        is not None,
        "PxSceneDesc.h: avbdIterations validity range must be [1, 255]",
    )
    require(
        re.search(
            r"\bPxU32\s+avbdJointIterationOverride\s*;", scene_desc
        )
        is not None,
        "PxSceneDesc.h: missing PxSceneDesc::avbdJointIterationOverride",
    )
    require(
        re.search(
            r"\bavbdJointIterationOverride\s*\(\s*8[uU]?\s*\)",
            scene_desc,
        )
        is not None,
        "PxSceneDesc.h: avbdJointIterationOverride must default to 8",
    )
    require(
        re.search(
            r"\bavbdJointIterationOverride\s*>\s*255[uU]?\b",
            scene_desc,
        )
        is not None,
        "PxSceneDesc.h: joint override must reject values above 255",
    )
    require(
        re.search(r"\bbool\s+avbdEnableEarlyStop\s*;", scene_desc)
        is not None,
        "PxSceneDesc.h: missing PxSceneDesc::avbdEnableEarlyStop",
    )
    require(
        re.search(
            r"\bavbdEnableEarlyStop\s*\(\s*true\s*\)", scene_desc
        )
        is not None,
        "PxSceneDesc.h: avbdEnableEarlyStop must default to true",
    )


def require_scene_to_solver_wiring() -> None:
    scene = code_only(read("physx/source/simulationcontroller/src/ScScene.cpp"))
    dynamics_header = code_only(
        read("physx/source/lowleveldynamics/src/DyAvbdDynamics.h")
    )
    dynamics = code_only(
        read("physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp")
    )

    require(
        re.search(
            r"createAVBDDynamicsContext\s*\("
            r"(?:(?!\);).)*\bdesc\.avbdIterations\s*,\s*"
            r"desc\.avbdJointIterationOverride\s*,\s*"
            r"desc\.avbdEnableEarlyStop\s*,\s*mPublicFlags\s*\)",
            scene,
            flags=re.DOTALL,
        )
        is not None,
        "ScScene.cpp: PxSceneDesc::avbdIterations is not passed to the AVBD factory",
    )
    require(
        re.search(
            r"createAVBDDynamicsContext\s*\("
            r"(?:(?!\);).)*\bPxU32\s+avbdIterations\s*,\s*"
            r"PxU32\s+avbdJointIterationOverride\s*,\s*"
            r"bool\s+avbdEnableEarlyStop\s*,\s*PxSceneFlags",
            dynamics_header,
            flags=re.DOTALL,
        )
        is not None,
        "DyAvbdDynamics.h: AVBD factory does not accept avbdIterations",
    )
    require(
        "mAvbdIterations(avbdIterations)" in re.sub(r"\s+", "", dynamics),
        "DyAvbdDynamics.cpp: context does not retain the scene iteration budget",
    )
    require(
        "mAvbdJointIterationOverride(avbdJointIterationOverride)"
        in re.sub(r"\s+", "", dynamics),
        "DyAvbdDynamics.cpp: context does not retain the joint override",
    )
    require(
        "mAvbdEnableEarlyStop(avbdEnableEarlyStop)"
        in re.sub(r"\s+", "", dynamics),
        "DyAvbdDynamics.cpp: context does not retain the early-stop switch",
    )
    require(
        re.search(
            r"\bconfig\.iterations\s*=\s*mAvbdIterations\s*;", dynamics
        )
        is not None,
        "DyAvbdDynamics.cpp: scene iteration budget does not initialize the rigid solver",
    )
    require(
        re.search(
            r"\bconfig\.jointIterationOverride\s*=\s*"
            r"mAvbdJointIterationOverride\s*;",
            dynamics,
        )
        is not None,
        "DyAvbdDynamics.cpp: joint override does not initialize the rigid solver",
    )
    require(
        re.search(
            r"\bconfig\.enableEarlyStop\s*=\s*mAvbdEnableEarlyStop\s*;",
            dynamics,
        )
        is not None,
        "DyAvbdDynamics.cpp: early-stop switch does not initialize the rigid solver",
    )
    require(
        re.search(
            r"\bconfig\.positionTolerance\s*\*=\s*config\.lengthScale\s*;",
            dynamics,
        )
        is not None,
        "DyAvbdDynamics.cpp: positionTolerance must scale with Scene lengthScale",
    )


def require_rigid_iteration_config() -> None:
    types_raw = read("physx/source/lowleveldynamics/src/DyAvbdTypes.h")
    types = code_only(types_raw)
    config = struct_body(types, "AvbdSolverConfig")

    iteration_fields = [
        field
        for field in re.findall(
            r"\b(?:physx::)?PxU32\s+([A-Za-z_]\w*)\s*;", config
        )
        if "iteration" in field.lower()
    ]
    require(
        iteration_fields == ["iterations", "jointIterationOverride"],
        "DyAvbdTypes.h: rigid AvbdSolverConfig must expose scene and joint "
        f"iteration controls (found {iteration_fields!r})",
    )
    require(
        re.search(r"\biterations\s*\(\s*4[uU]?\s*\)", config) is not None,
        "DyAvbdTypes.h: rigid AvbdSolverConfig::iterations must default to 4",
    )
    require(
        re.search(
            r"\bjointIterationOverride\s*\(\s*8[uU]?\s*\)", config
        )
        is not None,
        "DyAvbdTypes.h: jointIterationOverride must default to 8",
    )
    require(
        re.search(r"\bbool\s+enableEarlyStop\s*;", config) is not None,
        "DyAvbdTypes.h: AvbdSolverConfig is missing enableEarlyStop",
    )
    require(
        re.search(r"\benableEarlyStop\s*\(\s*true\s*\)", config)
        is not None,
        "DyAvbdTypes.h: enableEarlyStop must default to true",
    )
    require(
        "outerIterations" not in config and "innerIterations" not in config,
        "DyAvbdTypes.h: rigid AvbdSolverConfig retained outer/inner iteration semantics",
    )


def require_solver_iteration_policy() -> None:
    paths = {
        "contact": "physx/source/lowleveldynamics/src/DyAvbdSolver.cpp",
        "joint": "physx/source/lowleveldynamics/src/DyAvbdSolverJointPath.cpp",
    }
    sources = {name: code_only(read(path)) for name, path in paths.items()}
    maximum = re.compile(
        r"\b(?:physx::)?PxMax\s*\(\s*mConfig\.iterations\s*,\s*"
        r"iterationOverride\s*\)"
    )

    for name, source in sources.items():
        require(
            maximum.search(source) is not None,
            f"{paths[name]}: {name} path must use max(config.iterations, "
            "iterationOverride)",
        )
        require(
            re.search(r"\biterationOverride\s*>\s*0\s*\?", source) is None,
            f"{paths[name]}: positive override still replaces rather than raises "
            "the scene budget",
        )

    require(
        "jointIterationOverride" not in sources["contact"],
        "DyAvbdSolver.cpp: contact-only path must not apply the joint override",
    )

    joint = sources["joint"]
    require(
        re.search(
            r"(?:physx::)?PxMax\s*\(\s*(?:base|joint)Iterations\s*,\s*"
            r"mConfig\.jointIterationOverride\s*\)",
            joint,
        )
        is not None,
        "DyAvbdSolverJointPath.cpp: explicit joint override does not raise the budget",
    )
    require(
        re.search(
            r"(?:numD6\s*>\s*0|numGear\s*>\s*0)", joint
        )
        is not None
        and re.search(
            r"jointIterationOverride\s*>\s*0", joint
        )
        is not None,
        "DyAvbdSolverJointPath.cpp: joint override must be gated by actual "
        "joint constraints and zero must disable it",
    )
    require(
        re.search(
            r"(?:physx::)?PxMin\s*\(\s*jointIterations\s*,\s*"
            r"mConfig\.jointIterationOverride\s*\)",
            joint,
        )
        is not None,
        "DyAvbdSolverJointPath.cpp: enabled joint override must also be the "
        "early-stop minimum-iteration floor",
    )
    require(
        re.search(
            r"(?:physx::)?PxMax\s*\([^;]*\b8[uU]?\b", joint
        )
        is None,
        "DyAvbdSolverJointPath.cpp: hidden literal joint floor of 8 remains",
    )

    for name, source in sources.items():
        require(
            re.search(
                r"\bmConfig\.enableEarlyStop\s*&&", source
            )
            is not None,
            f"{paths[name]}: early stop is not controlled by the scene switch",
        )
        require(
            re.search(
                r"(?:physx::)?PxMin\s*\([^;]*,\s*"
                r"(?:physx::)?PxU32\s*\(\s*4\s*\)\s*\)",
                source,
            )
            is not None,
            f"{paths[name]}: early stop must wait for min(budget, 4)",
        )
        for token in ("consecutiveConverged", "computeMaxPoseDeltas"):
            require(
                token in source,
                f"{paths[name]}: early-stop convergence token {token} is missing",
            )
        require(
            re.search(
                r"(?:iters|jointIterations)\s*-\s*"
                r"(?:iterationState\.)?minIterations\s*>\s*1",
                source,
            )
            is not None,
            f"{paths[name]}: early-stop tracking must require enough budget "
            "for two convergence observations after the floor",
        )

    require(
        re.search(
            r"\benableEarlyStop\s*=\s*mConfig\.enableEarlyStop\s*&&"
            r"(?:(?!;).)*!\s*hasCompleteSoftSelection\b",
            joint,
            flags=re.DOTALL,
        )
        is not None,
        "DyAvbdSolverJointPath.cpp: complete soft selections must disable "
        "rigid pose-delta early stop",
    )

    rigid_policy = "\n".join(
        (
            code_only(read("physx/source/lowleveldynamics/src/DyAvbdTypes.h")),
            code_only(read("physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp")),
            sources["contact"],
            sources["joint"],
        )
    )
    for token in (
        "AVBD_MIN_INNER_ITERS_BODY_VS_STATIC",
        "contactOnlyIters",
    ):
        require(
            token not in rigid_policy,
            f"rigid AVBD policy retains forbidden hidden iteration path {token}",
        )
    require(
        re.search(
            r"\bcontact\w*Iter\w*\s*=\s*"
            r"(?:physx::)?PxMax\s*\([^;]*\b16[uU]?\b",
            rigid_policy,
            flags=re.IGNORECASE,
        )
        is None,
        "rigid AVBD policy retains a contact-only iteration floor of 16",
    )


def require_metadata_contract() -> None:
    names = code_only(
        read(
            "physx/source/physxmetadata/core/include/"
            "PxAutoGeneratedMetaDataObjectNames.h"
        )
    )
    objects = code_only(
        read(
            "physx/source/physxmetadata/core/include/"
            "PxAutoGeneratedMetaDataObjects.h"
        )
    )
    implementation = code_only(
        read(
            "physx/source/physxmetadata/core/src/"
            "PxAutoGeneratedMetaDataObjects.cpp"
        )
    )

    for property_name in (
        "AvbdIterations",
        "AvbdJointIterationOverride",
        "AvbdEnableEarlyStop",
    ):
        require(
            f"PxSceneDesc_{property_name}" in names,
            f"metadata names: missing PxSceneDesc_{property_name}",
        )
    for fragment in (
        "PxU32 AvbdIterations;",
        "PxU32 AvbdJointIterationOverride;",
        "DEFINE_PROPERTY_TO_VALUE_STRUCT_MAP( PxSceneDesc, AvbdIterations",
        "DEFINE_PROPERTY_TO_VALUE_STRUCT_MAP( PxSceneDesc, AvbdJointIterationOverride",
        "DEFINE_PROPERTY_TO_VALUE_STRUCT_MAP( PxSceneDesc, AvbdEnableEarlyStop",
        "PxSceneDesc_AvbdIterations, PxSceneDesc, PxU32",
        "PxSceneDesc_AvbdJointIterationOverride, PxSceneDesc, PxU32",
        "inOperator( AvbdIterations",
        "inOperator( AvbdJointIterationOverride",
        "inOperator( AvbdEnableEarlyStop",
    ):
        require(
            fragment in objects,
            f"metadata objects: missing AvbdIterations fragment {fragment!r}",
        )
    require(
        re.search(r"\b(?:_Bool|bool)\s+AvbdEnableEarlyStop\s*;", objects)
        is not None,
        "metadata objects: missing AvbdEnableEarlyStop value field",
    )
    require(
        re.search(
            r"PxSceneDesc_AvbdEnableEarlyStop\s*,\s*PxSceneDesc\s*,\s*"
            r"(?:_Bool|bool)",
            objects,
        )
        is not None,
        "metadata objects: missing AvbdEnableEarlyStop property",
    )
    for fragment in (
        "getPxSceneDescAvbdIterations",
        "setPxSceneDescAvbdIterations",
        "getPxSceneDescAvbdJointIterationOverride",
        "setPxSceneDescAvbdJointIterationOverride",
        "getPxSceneDescAvbdEnableEarlyStop",
        "setPxSceneDescAvbdEnableEarlyStop",
        'AvbdIterations( "AvbdIterations"',
        'AvbdJointIterationOverride( "AvbdJointIterationOverride"',
        'AvbdEnableEarlyStop( "AvbdEnableEarlyStop"',
        "AvbdIterations( inSource->avbdIterations )",
        "AvbdJointIterationOverride( inSource->avbdJointIterationOverride )",
        "AvbdEnableEarlyStop( inSource->avbdEnableEarlyStop )",
    ):
        require(
            fragment in implementation,
            f"metadata implementation: missing AvbdIterations fragment {fragment!r}",
        )


def require_soft_body_iteration_model_preserved() -> None:
    soft_body = code_only(
        read("physx/source/lowleveldynamics/src/DyAvbdSoftBodyComponent.h")
    )
    require(
        re.search(r"\bPxU32\s+outerIterations\b", soft_body) is not None
        and re.search(r"\bPxU32\s+innerIterations\b", soft_body) is not None,
        "DyAvbdSoftBodyComponent.h: genuine soft-body outer/inner iteration "
        "mechanism was removed by the rigid-policy cleanup",
    )


def require_benchmark_contract() -> None:
    snippet = code_only(
        read("physx/snippets/snippethelloworld/SnippetHelloWorld.cpp")
    )
    authority = code_only(read("tools/run_avbd_rigid_stress_baseline.py"))
    matrix = code_only(read("tools/run_avbd_cpu_performance_matrix.py"))

    for prefix in ("[AVBD_GATE]", "[AVBD_RIGID_PERF]", "[AVBD_RIGID_WORK]"):
        require(
            re.search(
                rf"{re.escape(prefix)}\s+schema=4\b", snippet
            )
            is not None,
            f"SnippetHelloWorld.cpp: {prefix} must use authority schema 4",
        )
    for token in (
        "avbdIterationSemantics=budgeted-complete-primal-dual-stiffness",
        "avbdJointIterationOverrideSource=",
        "avbdJointIterationOverrideActive=0",
        "avbdEarlyStopSource=",
        "avbdEarlyStopEnabled=",
        "avbdEarlyStopActive=",
    ):
        require(
            token in snippet,
            f"SnippetHelloWorld.cpp: missing telemetry token {token!r}",
        )
    require(
        re.search(
            r"--avbd-joint-iteration-override=.*?parseU32\s*\("
            r".*?,\s*0[uU]?\s*,\s*255[uU]?\s*,",
            snippet,
            flags=re.DOTALL,
        )
        is not None,
        "SnippetHelloWorld.cpp: joint override CLI range must be [0, 255]",
    )
    require(
        "--avbd-early-stop=" in snippet
        and 'equalsIgnoreCase(value, "on")' in snippet
        and 'equalsIgnoreCase(value, "off")' in snippet,
        "SnippetHelloWorld.cpp: early-stop CLI must accept on|off",
    )
    require(
        "getRigidStressAvbdEarlyStopActive" in snippet
        and re.search(
            r"gRigidStressSceneAvbdIterations\s*-\s*earlyStopFloor\s*>\s*1",
            snippet,
        )
        is not None,
        "SnippetHelloWorld.cpp: early-stop Active telemetry must reflect "
        "actual rigid-stress eligibility, not only the enabled setting",
    )
    for condition, assignment in (
        (
            "gRigidStressAvbdIterationsExplicit",
            "sceneDesc.avbdIterations",
        ),
        (
            "gRigidStressAvbdJointIterationOverrideExplicit",
            "sceneDesc.avbdJointIterationOverride",
        ),
        (
            "gRigidStressAvbdEarlyStopExplicit",
            "sceneDesc.avbdEnableEarlyStop",
        ),
    ):
        require(
            re.search(
                rf"if\s*\(\s*{condition}\s*\)\s*{assignment}\s*=",
                snippet,
            )
            is not None,
            f"SnippetHelloWorld.cpp: {assignment} must only be written for "
            "an explicit CLI override",
        )

    for fragment in (
        'SCHEMA = "4"',
        "AVBD_ITERATIONS = 4",
        "AVBD_JOINT_ITERATION_OVERRIDE = 8",
        'AVBD_EARLY_STOP_SOURCE = "default"',
        "AVBD_EARLY_STOP_ENABLED = 1",
        "AVBD_EARLY_STOP_ACTIVE = 0",
        'AVBD_ITERATION_SEMANTICS = "budgeted-complete-primal-dual-stiffness"',
    ):
        require(
            fragment in authority,
            f"rigid-stress authority: missing contract fragment {fragment!r}",
        )
    for fragment in (
        "AUTHORITY_SCHEMA = 4",
        "MATRIX_SCHEMA = 3",
        "AVBD_ITERATIONS = 4",
        "AVBD_JOINT_ITERATION_OVERRIDE = 8",
        "AVBD_EARLY_STOP_ENABLED = True",
        "AVBD_EARLY_STOP_ACTIVE = False",
        'AVBD_ITERATION_SEMANTICS = "budgeted-complete-primal-dual-stiffness"',
    ):
        require(
            fragment in matrix,
            f"CPU matrix: missing contract fragment {fragment!r}",
        )
    required_fields = (
        "avbdJointIterationOverrideSource",
        "avbdJointIterationOverride",
        "avbdJointIterationOverrideActive",
        "avbdEarlyStopSource",
        "avbdEarlyStopEnabled",
        "avbdEarlyStopActive",
    )
    for runner_name, runner in (("authority", authority), ("matrix", matrix)):
        for field in required_fields:
            require(
                f'"{field}"' in runner,
                f"formal {runner_name} runner does not validate {field}",
            )
        require(
            '"avbdEarlyStopActiveForFixture"' in runner
            and '"avbdEarlyStopActiveSolver"' not in runner,
            f"formal {runner_name} runner must distinguish the enabled "
            "setting from fixture-level early-stop eligibility",
        )
    for runner_name, runner in (("authority", authority), ("matrix", matrix)):
        for option in (
            "--avbd-iterations=",
            "--avbd-joint-iteration-override=",
            "--avbd-early-stop=",
        ):
            require(
                option not in runner,
                f"formal {runner_name} runner must exercise SceneDesc defaults; "
                f"found {option}",
            )


def main() -> int:
    require_scene_desc_contract()
    require_scene_to_solver_wiring()
    require_rigid_iteration_config()
    require_solver_iteration_policy()
    require_metadata_contract()
    require_soft_body_iteration_model_preserved()
    require_benchmark_contract()

    print(
        f"[{TAG}] status=PASS sceneDefault=4 range=1..255 "
        "jointOverrideDefault=8 jointOverrideRange=0..255 "
        "rigidPolicy=budgeted-complete-iteration "
        "overridePolicy=joint-only hiddenContact16=absent "
        "hiddenJoint8=absent earlyStopDefault=on earlyStopSwitch=present "
        "metadata=present authoritySchema=4 matrixSchema=3 "
        "softOuterInner=preserved"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"[{TAG}] status=FAIL error={error}")
        raise SystemExit(1)
