#!/usr/bin/env python3
"""Fail closed if the AVBD CPU colored-backend contract regresses.

This is a source-contract gate, not a performance or numerical authority.  It
keeps the new non-deterministic CPU body-color path separate from the ordered
scene backend and from the explicitly installed GPU owner-wave backend.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAG = "AVBD_CPU_COLORED_BACKEND_GATE"
GROUPS = (
    "sceneRouting",
    "orderedAdmission",
    "colorContext",
    "strictEdges",
    "completeCoverage",
    "csrValidation",
    "colorGrain",
    "dualFanout",
    "gpuSplit",
    "mixedExcluded",
)


class Gate:
    def __init__(self) -> None:
        self.groups = {name: True for name in GROUPS}
        self.errors: list[str] = []

    def fail(self, groups: str | tuple[str, ...], message: str) -> None:
        if isinstance(groups, str):
            groups = (groups,)
        for group in groups:
            self.groups[group] = False
        self.errors.append(message)

    def check(
        self,
        groups: str | tuple[str, ...],
        condition: bool,
        message: str,
    ) -> None:
        if not condition:
            self.fail(groups, message)


def code_only(text: str) -> str:
    """Remove comments so retired contracts cannot satisfy the gate."""

    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\r\n]*", "", text)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def read_code(gate: Gate, relative: str, groups: tuple[str, ...]) -> str:
    path = ROOT / relative
    if not path.is_file():
        gate.fail(groups, f"missing file: {relative}")
        return ""
    try:
        return code_only(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as error:
        gate.fail(groups, f"cannot read {relative}: {error}")
        return ""


def block_after(
    gate: Gate,
    text: str,
    pattern: str,
    label: str,
    groups: str | tuple[str, ...],
    start: int = 0,
) -> str:
    """Return the brace-balanced block following a declaration/statement."""

    match = re.search(pattern, text[start:], flags=re.DOTALL)
    if match is None:
        gate.fail(groups, f"missing {label}")
        return ""
    absolute_start = start + match.start()
    opening = text.find("{", absolute_start)
    if opening < 0:
        gate.fail(groups, f"missing opening brace for {label}")
        return ""

    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]

    gate.fail(groups, f"unterminated block for {label}")
    return ""


def statement_after(
    gate: Gate,
    text: str,
    anchor: str,
    label: str,
    groups: str | tuple[str, ...],
) -> str:
    start = text.find(anchor)
    if start < 0:
        gate.fail(groups, f"missing {label}")
        return ""
    end = text.find(";", start)
    if end < 0:
        gate.fail(groups, f"unterminated {label}")
        return ""
    return text[start : end + 1]


def require_scene_and_admission_contracts(
    gate: Gate, types: str, dynamics: str, tasks: str
) -> None:
    config = block_after(
        gate,
        types,
        r"\bstruct\s+AvbdSolverConfig\s*\{",
        "AvbdSolverConfig",
        ("sceneRouting", "orderedAdmission"),
    )
    ordered = block_after(
        gate,
        config,
        r"\bvoid\s+enableOrderedBackend\s*\(\s*\)",
        "AvbdSolverConfig::enableOrderedBackend",
        "sceneRouting",
    )
    requires = block_after(
        gate,
        config,
        r"\bbool\s+requiresOrderedBackend\s*\(\s*\)\s*const",
        "AvbdSolverConfig::requiresOrderedBackend",
        "orderedAdmission",
    )
    update = block_after(
        gate,
        dynamics,
        r"\bvoid\s+AvbdDynamicsContext::update\s*\(",
        "AvbdDynamicsContext::update",
        ("sceneRouting", "orderedAdmission"),
    )
    admission = block_after(
        gate,
        tasks,
        r"\bbool\s+AvbdSolveIslandTask::canUseRigidWaveTasks\s*"
        r"\(\s*\)\s*const",
        "AvbdSolveIslandTask::canUseRigidWaveTasks",
        ("orderedAdmission", "mixedExcluded"),
    )
    deferred = statement_after(
        gate,
        update,
        "const bool canDeferContactPreparation",
        "P280 deferred-contact admission",
        "orderedAdmission",
    )

    gate.check(
        "sceneRouting",
        re.search(r"\bbool\s+useOrderedBackend\s*;", config) is not None,
        "DyAvbdTypes.h: missing AvbdSolverConfig::useOrderedBackend",
    )
    gate.check(
        "sceneRouting",
        re.search(r"\buseOrderedBackend\s*\(\s*false\s*\)", config)
        is not None,
        "DyAvbdTypes.h: ordered backend must default off",
    )
    ordered_compact = compact(ordered)
    gate.check(
        "sceneRouting",
        "useOrderedBackend=true;" in ordered_compact
        and "enableParallelization=false;" in ordered_compact,
        "DyAvbdTypes.h: enableOrderedBackend must select ordered execution "
        "and disable parallelization",
    )
    gate.check(
        "orderedAdmission",
        "returnuseOrderedBackend||isDeterministic();" in compact(requires),
        "DyAvbdTypes.h: requiresOrderedBackend must cover both ordered and "
        "deterministic solver modes",
    )
    gate.check(
        "sceneRouting",
        re.search(
            r"if\s*\(\s*isEnhancedDeterminismEnabled\s*\(\s*\)\s*\)\s*"
            r"(?:\{\s*)?config\.enableOrderedBackend\s*\(\s*\)\s*;",
            update,
        )
        is not None,
        "DyAvbdDynamics.cpp: enhanced determinism is not routed to "
        "enableOrderedBackend",
    )
    gate.check(
        "orderedAdmission",
        "!mSolver.getConfig().requiresOrderedBackend()" in compact(deferred),
        "DyAvbdDynamics.cpp: P280 deferred contact preparation does not fail "
        "closed on requiresOrderedBackend",
    )
    gate.check(
        "orderedAdmission",
        "isDeterministic()" not in deferred,
        "DyAvbdDynamics.cpp: P280 admission bypasses the unified ordered "
        "backend predicate",
    )
    admission_compact = compact(admission)
    gate.check(
        "orderedAdmission",
        "returnconfig.enableParallelization&&!config.requiresOrderedBackend();"
        in admission_compact,
        "DyAvbdTasks.cpp: colored CPU admission must require parallelization "
        "and reject ordered backends",
    )
    gate.check(
        "orderedAdmission",
        "isDeterministic()" not in admission,
        "DyAvbdTasks.cpp: colored admission bypasses "
        "requiresOrderedBackend",
    )

    for token, description in (
        ("mBatch.hasArticulationBodies", "articulation bodies"),
        ("mBatch.numD6 != 0", "D6 joints"),
        ("mBatch.numGear != 0", "gear joints"),
        ("mBatch.numSoftParticles != 0", "soft particles"),
        ("mBatch.numSoftBodies != 0", "soft bodies"),
        ("mBatch.numSoftContacts != 0", "soft contacts"),
    ):
        gate.check(
            "mixedExcluded",
            token in admission,
            f"DyAvbdTasks.cpp: colored admission does not exclude {description}",
        )


def require_color_context(gate: Gate, solver_header: str) -> None:
    context = block_after(
        gate,
        solver_header,
        r"\bstruct\s+AvbdRigidSolveContext\s*\{",
        "AvbdRigidSolveContext",
        "colorContext",
    )
    for field in ("bodyColorOffsets", "bodyColorBodies"):
        gate.check(
            "colorContext",
            re.search(
                rf"\bphysx::PxArray\s*<\s*physx::PxU32\s*>\s+"
                rf"{field}\s*;",
                context,
            )
            is not None,
            f"DyAvbdSolver.h: AvbdRigidSolveContext is missing {field}",
        )
    for field in ("bodyColorCount", "maxBodyColorWidth"):
        gate.check(
            "colorContext",
            re.search(rf"\bphysx::PxU32\s+{field}\s*;", context) is not None,
            f"DyAvbdSolver.h: AvbdRigidSolveContext is missing {field}",
        )
        gate.check(
            "colorContext",
            re.search(rf"\b{field}\s*\(\s*0[uU]?\s*\)", context)
            is not None,
            f"DyAvbdSolver.h: {field} must initialize to zero",
        )


def require_color_plan(gate: Gate, solver: str) -> None:
    builder = block_after(
        gate,
        solver,
        r"\bbool\s+AvbdSolver::buildRigidBodyColorPlan\s*\(",
        "AvbdSolver::buildRigidBodyColorPlan",
        ("strictEdges", "completeCoverage"),
    )
    builder_compact = compact(builder)

    for token in (
        "context.bodyColorOffsets.clear();",
        "context.bodyColorBodies.clear();",
        "context.bodyColorCount=0;",
        "context.maxBodyColorWidth=0;",
    ):
        gate.check(
            "completeCoverage",
            token in builder_compact,
            f"DyAvbdSolver.cpp: color plan does not reset {token}",
        )

    for token, message in (
        ("contactMap->numBodies != numBodies", "map body count"),
        ("!contactMap->constraintOffsets", "map offsets"),
        ("!contactMap->constraintCounts", "map counts"),
        ("!contactMap->constraintIndices", "map indices"),
        (
            "contactMap->constraintOffsets[numBodies] !=",
            "terminal CSR offset",
        ),
        ("contactMap->totalConstraintRefs", "total CSR reference count"),
        ("bodies[body].nodeIndex != body", "island-local body index"),
        ("contactMap->getBodyConstraints", "per-body CSR traversal"),
        ("contactIndex >= numContacts", "CSR contact index bounds"),
        ("bodyA != body && bodyB != body", "CSR endpoint ownership"),
        ("contactMap->constraintOffsets[0] != 0", "zero CSR origin"),
        ("begin > end", "monotonic CSR offsets"),
        ("end > contactMap->totalConstraintRefs", "CSR range bounds"),
        ("contactMap->constraintCounts[body] != end - begin", "CSR row count"),
    ):
        gate.check(
            "completeCoverage",
            token in builder,
            f"DyAvbdSolver.cpp: color plan does not validate {message}",
        )

    for token in (
        "contactMap->constraintOffsets[0] != 0",
        "begin > end",
        "end > contactMap->totalConstraintRefs",
        "contactMap->constraintCounts[body] != end - begin",
    ):
        gate.check(
            "csrValidation",
            token in builder,
            f"DyAvbdSolver.cpp: fail-closed CSR validation is missing {token}",
        )

    dynamic_start = builder.find("physx::PxU32 dynamicBodyCount = 0")
    gate.check(
        "completeCoverage",
        dynamic_start >= 0,
        "DyAvbdSolver.cpp: color plan is missing dynamic-body census",
    )
    dynamic_loop = ""
    if dynamic_start >= 0:
        dynamic_loop = block_after(
            gate,
            builder,
            r"for\s*\(\s*(?:physx::)?PxU32\s+body\s*=\s*0\s*;\s*"
            r"body\s*<\s*numBodies\s*;\s*\+\+body\s*\)",
            "dynamic-body coloring loop",
            "completeCoverage",
            start=dynamic_start,
        )
    dynamic_compact = compact(dynamic_loop)
    dynamic_fragments = (
        "if(bodies[body].invMass<=0.0f)continue;",
        "++dynamicBodyCount;",
        "bodyColors[body]=color;",
    )
    for fragment in dynamic_fragments:
        gate.check(
            "completeCoverage",
            fragment in dynamic_compact,
            f"DyAvbdSolver.cpp: dynamic coloring loop is missing {fragment}",
        )
    if all(fragment in dynamic_compact for fragment in dynamic_fragments):
        gate.check(
            "completeCoverage",
            dynamic_compact.index(dynamic_fragments[0])
            < dynamic_compact.index(dynamic_fragments[1])
            < dynamic_compact.index(dynamic_fragments[2]),
            "DyAvbdSolver.cpp: dynamic census/color assignment order is unsafe",
        )

    edge_match = re.search(
        r"for\s*\(\s*(?:physx::)?PxU32\s+contactIndex\s*=\s*0\s*;\s*"
        r"contactIndex\s*<\s*numContacts\s*;\s*\+\+contactIndex\s*\)",
        builder,
    )
    edge_start = edge_match.start() if edge_match is not None else -1
    edge_loop = ""
    if edge_start < 0:
        gate.fail("strictEdges", "DyAvbdSolver.cpp: missing full contact-edge validation loop")
    else:
        edge_loop = block_after(
            gate,
            builder,
            r"for\s*\(\s*(?:physx::)?PxU32\s+contactIndex\s*=\s*0\s*;\s*"
            r"contactIndex\s*<\s*numContacts\s*;\s*\+\+contactIndex\s*\)",
            "full contact-edge validation loop",
            "strictEdges",
        )
    edge_compact = compact(edge_loop)
    for fragment, description in (
        (
            "bodyA=contacts[contactIndex].header.bodyIndexA;",
            "body A source endpoint",
        ),
        (
            "bodyB=contacts[contactIndex].header.bodyIndexB;",
            "body B source endpoint",
        ),
        ("bodyA>=numBodies", "body A bounds"),
        ("bodyB>=numBodies", "body B bounds"),
        ("bodyA==bodyB", "self edge"),
        ("bodies[bodyA].invMass<=0.0f", "static body A exclusion"),
        ("bodies[bodyB].invMass<=0.0f", "static body B exclusion"),
        ("bodyColors[bodyA]==PX_MAX_U32", "uncolored body A"),
        ("bodyColors[bodyB]==PX_MAX_U32", "uncolored body B"),
        ("bodyColors[bodyA]==bodyColors[bodyB]", "same-color edge"),
        ("returnfalse;", "fail-closed rejection"),
    ):
        gate.check(
            "strictEdges",
            fragment in edge_compact,
            f"DyAvbdSolver.cpp: strict edge pass is missing {description}",
        )
    offsets_start = builder.find("context.bodyColorOffsets.resize(colorCount + 1u)")
    gate.check(
        "strictEdges",
        edge_start >= 0
        and offsets_start >= 0
        and builder.find("bodyColors[body] = color") < edge_start < offsets_start,
        "DyAvbdSolver.cpp: source-edge validation must run after coloring and "
        "before publishing compact ranges",
    )

    for fragment, description in (
        ("context.bodyColorOffsets.resize(colorCount+1u);", "color offsets"),
        (
            "context.bodyColorOffsets[colorCount]!=dynamicBodyCount",
            "complete dynamic-body count",
        ),
        ("context.bodyColorBodies.resize(dynamicBodyCount);", "packed body array"),
        (
            "context.bodyColorBodies[writeOffsets[color]++]=body;",
            "one packed entry per colored body",
        ),
        ("context.bodyColorCount=colorCount;", "published color count"),
        ("returntrue;", "successful complete plan"),
    ):
        gate.check(
            "completeCoverage",
            fragment in builder_compact,
            f"DyAvbdSolver.cpp: color plan is missing {description}",
        )


def require_dispatch_split(gate: Gate, tasks: str) -> None:
    policy = block_after(
        gate,
        tasks,
        r"\bstruct\s+AvbdRigidExecutionPolicy\s*\{",
        "AvbdRigidExecutionPolicy",
        ("colorGrain", "gpuSplit"),
    )
    submit_wave = block_after(
        gate,
        tasks,
        r"\bvoid\s+AvbdSolveIslandTask::submitRigidWave\s*\(\s*\)",
        "AvbdSolveIslandTask::submitRigidWave",
        "gpuSplit",
    )
    submit_color = block_after(
        gate,
        tasks,
        r"\bvoid\s+AvbdSolveIslandTask::submitRigidColor\s*\(\s*\)",
        "AvbdSolveIslandTask::submitRigidColor",
        ("colorGrain", "gpuSplit"),
    )
    run = block_after(
        gate,
        tasks,
        r"\bvoid\s+AvbdSolveIslandTask::run\s*\(\s*\)",
        "AvbdSolveIslandTask::run",
        ("colorGrain", "gpuSplit"),
    )

    gate.check(
        "colorGrain",
        re.search(r"\beCOLOR_TASK_GRAIN_BODIES\s*=\s*64[uU]?\b", policy)
        is not None,
        "DyAvbdTasks.cpp: colored CPU task grain must remain 64 bodies",
    )
    gate.check(
        "gpuSplit",
        re.search(r"\beTASK_GRAIN_BODIES\s*=\s*256[uU]?\b", policy)
        is not None,
        "DyAvbdTasks.cpp: exact owner-wave task grain must remain separate",
    )
    color_compact = compact(submit_color)
    for fragment, description in (
        (
            "context.bodyColorOffsets[mCurrentColor]",
            "current color range start",
        ),
        (
            "context.bodyColorOffsets[mCurrentColor+1u]",
            "current color range end",
        ),
        (
            "AvbdRigidExecutionPolicy::eCOLOR_TASK_GRAIN_BODIES",
            "colored task grain",
        ),
        ("mRigidContext.bodyColorBodies.begin()", "colored owner list"),
        ("solveRigidBodyRange(", "inline colored solve"),
        ("createRigidBodyRangeTask(", "parallel colored solve"),
    ):
        gate.check(
            "colorGrain",
            fragment in color_compact,
            f"DyAvbdTasks.cpp: submitRigidColor is missing {description}",
        )
    gate.check(
        "colorGrain",
        color_compact.count("mRigidContext.bodyColorBodies.begin()") >= 2,
        "DyAvbdTasks.cpp: both inline and child colored ranges must use the "
        "compact body list",
    )

    for forbidden in (
        "dependencyWaveBodies",
        "getRigidGpuWaveBackend",
        "solveRigidOwnerWave",
        "eTASK_GRAIN_BODIES",
    ):
        gate.check(
            "gpuSplit",
            forbidden not in submit_color,
            f"DyAvbdTasks.cpp: CPU colored submission leaks GPU/wave token "
            f"{forbidden}",
        )
    for token, description in (
        ("dependencyWaveOffsets", "dependency-wave ranges"),
        ("dependencyWaveBodies.begin()", "dependency-wave owner list"),
        ("eTASK_GRAIN_BODIES", "owner-wave task grain"),
        ("getRigidGpuWaveBackend", "explicit GPU backend lookup"),
        ("solveRigidOwnerWave", "GPU owner-wave transaction"),
    ):
        gate.check(
            "gpuSplit",
            token in submit_wave,
            f"DyAvbdTasks.cpp: submitRigidWave is missing {description}",
        )

    run_compact = compact(run)
    for fragment, description in (
        ("getRigidGpuWaveBackend()", "explicit GPU backend selection"),
        (
            "mRigidUsesBodyColors=!(gpuBackend&&gpuBackend->isAvailable());",
            "CPU-color/GPU-wave selector",
        ),
        ("buildRigidBodyColorPlan(mRigidContext)", "CPU color-plan build"),
        ("buildRigidDependencyWaves(mRigidContext)", "GPU wave-plan build"),
        (
            "maxBodyColorWidth<=AvbdRigidExecutionPolicy::"
            "eCOLOR_TASK_GRAIN_BODIES",
            "small-color synchronous fallback",
        ),
        (
            "mRigidUsesBodyColors&&mCurrentColor<"
            "mRigidContext.bodyColorCount",
            "colored iteration admission",
        ),
        ("submitRigidColor();", "colored range dispatch"),
        (
            "!mRigidUsesBodyColors&&mCurrentWave<"
            "mRigidContext.dependencyWaveCount",
            "owner-wave iteration admission",
        ),
        ("submitRigidWave();", "owner-wave dispatch"),
    ):
        gate.check(
            "gpuSplit",
            fragment in run_compact,
            f"DyAvbdTasks.cpp: run path is missing {description}",
        )
    color_build = run.find("buildRigidBodyColorPlan")
    wave_build = run.find("buildRigidDependencyWaves")
    color_dispatch = run.find("submitRigidColor")
    wave_dispatch = run.find("submitRigidWave")
    gate.check(
        "gpuSplit",
        min(color_build, wave_build, color_dispatch, wave_dispatch) >= 0
        and color_build < wave_build < color_dispatch < wave_dispatch,
        "DyAvbdTasks.cpp: CPU color and GPU owner-wave phases are not kept in "
        "their expected branches",
    )


def require_dual_fanout(
    gate: Gate, solver_header: str, solver: str, tasks_header: str, tasks: str
) -> None:
    state = block_after(
        gate,
        solver_header,
        r"\bstruct\s+AvbdRigidSolveIterationState\s*\{",
        "AvbdRigidSolveIterationState",
        "dualFanout",
    )
    dual_task = block_after(
        gate,
        tasks_header,
        r"\bclass\s+AvbdRigidDualRangeTask\s*:\s*public\s+AvbdTask\s*\{",
        "AvbdRigidDualRangeTask",
        "dualFanout",
    )
    range_kernel = block_after(
        gate,
        solver,
        r"\bvoid\s+AvbdSolver::solveRigidDualRange\s*\(",
        "AvbdSolver::solveRigidDualRange",
        "dualFanout",
    )
    complete = block_after(
        gate,
        solver,
        r"\bbool\s+AvbdSolver::completeRigidSolveIteration\s*\(",
        "AvbdSolver::completeRigidSolveIteration",
        "dualFanout",
    )
    policy = block_after(
        gate,
        tasks,
        r"\bstruct\s+AvbdRigidExecutionPolicy\s*\{",
        "AvbdRigidExecutionPolicy",
        "dualFanout",
    )
    submit = block_after(
        gate,
        tasks,
        r"\bbool\s+AvbdSolveIslandTask::submitRigidDual\s*\(\s*\)",
        "AvbdSolveIslandTask::submitRigidDual",
        "dualFanout",
    )
    run = block_after(
        gate,
        tasks,
        r"\bvoid\s+AvbdSolveIslandTask::run\s*\(\s*\)",
        "AvbdSolveIslandTask::run",
        "dualFanout",
    )

    for text, fragment, message in (
        (state, "bool parallelDualComplete;", "iteration completion latch"),
        (state, "parallelDualComplete(false)", "false latch default"),
        (dual_task, "PxU32 mBegin;", "task range begin"),
        (dual_task, "PxU32 mEnd;", "task range end"),
        (range_kernel, "state.contacts + begin", "disjoint contact base"),
        (range_kernel, "end - begin", "disjoint contact count"),
        (range_kernel, "end - begin > 4u", "dual branch-preserving range"),
        (complete, "if (!state.parallelDualComplete)", "scalar fallback"),
        (complete, "state.parallelDualComplete = false;", "latch reset"),
        (policy, "eDUAL_TASK_GRAIN_CONTACTS = 256u", "dual task grain"),
        (
            submit,
            "count / AvbdRigidExecutionPolicy::eDUAL_TASK_GRAIN_CONTACTS",
            "coarse admission",
        ),
        (submit, "createRigidDualRangeTask", "dual task factory"),
        (submit, "task->setContinuation(this)", "dual fan-in"),
        (run, "mRigidUsesBodyColors && submitRigidDual()", "CPU-only dual dispatch"),
        (
            run,
            "mRigidContext.iteration.parallelDualComplete = true;",
            "dual completion publication",
        ),
        (run, "mRigidPhase = eRIGID_POST_DUAL;", "post-dual phase transition"),
    ):
        gate.check(
            "dualFanout",
            fragment in text,
            f"AVBD CPU dual fan-out is missing {message}",
        )
    gate.check(
        "dualFanout",
        "updateLagrangianMultipliers(" in complete,
        "ordered/scalar dual authority was removed",
    )
    gate.check(
        "dualFanout",
        "#ifPX_AVBD_ENABLE_SOLVER_PROFILEreturnfalse;#else"
        in compact(submit),
        "AVBD CPU dual fan-out must fail closed when the solver-profile "
        "payload requires a scalar global reduction",
    )


def main() -> int:
    gate = Gate()
    types = read_code(
        gate,
        "physx/source/lowleveldynamics/src/DyAvbdTypes.h",
        ("sceneRouting", "orderedAdmission"),
    )
    dynamics = read_code(
        gate,
        "physx/source/lowleveldynamics/src/DyAvbdDynamics.cpp",
        ("sceneRouting", "orderedAdmission"),
    )
    solver_header = read_code(
        gate,
        "physx/source/lowleveldynamics/src/DyAvbdSolver.h",
        (
            "colorContext",
            "strictEdges",
            "completeCoverage",
            "dualFanout",
        ),
    )
    solver = read_code(
        gate,
        "physx/source/lowleveldynamics/src/DyAvbdSolver.cpp",
        (
            "strictEdges",
            "completeCoverage",
            "csrValidation",
            "dualFanout",
        ),
    )
    tasks = read_code(
        gate,
        "physx/source/lowleveldynamics/src/DyAvbdTasks.cpp",
        (
            "orderedAdmission",
            "colorGrain",
            "dualFanout",
            "gpuSplit",
            "mixedExcluded",
        ),
    )
    tasks_header = read_code(
        gate,
        "physx/source/lowleveldynamics/src/DyAvbdTasks.h",
        ("dualFanout", "mixedExcluded"),
    )

    require_scene_and_admission_contracts(gate, types, dynamics, tasks)
    require_color_context(gate, solver_header)
    require_color_plan(gate, solver)
    require_dispatch_split(gate, tasks)
    require_dual_fanout(gate, solver_header, solver, tasks_header, tasks)

    status = "PASS" if not gate.errors else "FAIL"
    summary = " ".join(
        f"{name}={'present' if gate.groups[name] else 'missing'}"
        for name in GROUPS
    )
    print(f"[{TAG}] {summary} status={status}")
    for error in gate.errors:
        print(f"- {error}")
    return 0 if not gate.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
