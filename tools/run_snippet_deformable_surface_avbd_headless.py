#!/usr/bin/env python3
"""Run public CPU AVBD deformable-surface gates without a window."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

from snippet_headless_process import run_headless_process


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = (
    REPO_ROOT / "physx" / "bin" / "win.x86_64.vc143.md" / "checked"
)
EXECUTABLE = "SnippetDeformableSurfaceAVBD_64.exe"
CASES = (
    "surface-lifecycle",
    "surface-performance-dense-no-contact",
    "surface-ogc-box-edge",
    "surface-ground",
    "surface-sleep-wake",
    "surface-buffer-mutation",
    "surface-dynamic-box",
    "surface-dynamic-sphere",
    "surface-dynamic-capsule",
    "surface-dynamic-convex",
    "surface-kinematic-box",
    "surface-kinematic-sphere",
    "surface-kinematic-capsule",
    "surface-kinematic-convex",
    "surface-kinematic-triangle-mesh",
    "surface-kinematic-heightfield",
    "surface-soft-soft-wake",
    "surface-soft-soft-swept-ccd",
    "surface-volume-wake",
    "surface-surface-attachment",
    "surface-volume-attachment",
    "surface-self-collision",
    "surface-self-collision-filter",
    "surface-self-collision-swept-ccd",
    "surface-material-friction",
    "surface-world-pin",
    "surface-world-element-attachment",
    "surface-rigid-attachment",
    "surface-rigid-element-attachment",
    "surface-static-attachment",
    "surface-static-element-attachment",
    "surface-kinematic-attachment",
    "surface-kinematic-element-attachment",
    "surface-articulation-attachment",
    "surface-articulation-element-attachment",
    "surface-element-filter",
    "surface-partial-element-filter",
    "surface-soft-soft-element-filter",
    "surface-volume-element-filter",
    "volume-volume-element-filter",
    "surface-bending",
    "surface-flattening",
    "surface-motion-controls",
    "surface-max-depenetration-velocity",
    "surface-speculative-ccd",
    "surface-plane-speculative-ccd",
    "surface-sphere-speculative-ccd",
    "surface-capsule-speculative-ccd",
    "surface-convex-speculative-ccd",
    "surface-moving-kinematic-sphere-speculative-ccd",
    "surface-moving-kinematic-capsule-speculative-ccd",
    "surface-rotating-kinematic-capsule-speculative-ccd",
    "surface-moving-kinematic-convex-speculative-ccd",
    "surface-rotating-kinematic-convex-speculative-ccd",
    "surface-dynamic-sphere-relative-swept-ccd",
    "surface-dynamic-capsule-relative-swept-ccd",
    "surface-dynamic-rotating-capsule-relative-swept-ccd",
    "surface-dynamic-convex-relative-swept-ccd",
    "surface-dynamic-rotating-convex-relative-swept-ccd",
    "surface-static-sphere-reverse-swept-ccd",
    "surface-kinematic-sphere-reverse-swept-ccd",
    "surface-dynamic-sphere-reverse-swept-ccd",
    "surface-static-capsule-reverse-swept-ccd",
    "surface-kinematic-capsule-reverse-swept-ccd",
    "surface-dynamic-capsule-reverse-swept-ccd",
    "surface-rotating-kinematic-capsule-reverse-swept-ccd",
    "surface-dynamic-rotating-capsule-reverse-swept-ccd",
    "surface-static-convex-reverse-swept-ccd",
    "surface-kinematic-convex-reverse-swept-ccd",
    "surface-dynamic-convex-reverse-swept-ccd",
    "surface-rotating-kinematic-convex-reverse-swept-ccd",
    "surface-dynamic-rotating-convex-reverse-swept-ccd",
    "surface-deforming-sphere-reverse-swept-ccd",
    "surface-deforming-capsule-reverse-swept-ccd",
    "surface-deforming-convex-reverse-swept-ccd",
    "surface-deforming-triangle-mesh-reverse-swept-ccd",
    "surface-deforming-heightfield-reverse-swept-ccd",
    "surface-static-triangle-mesh-speculative-ccd",
    "surface-kinematic-triangle-mesh-speculative-ccd",
    "surface-static-heightfield-speculative-ccd",
    "surface-kinematic-heightfield-speculative-ccd",
    "surface-static-triangle-mesh-reverse-swept-ccd",
    "surface-kinematic-triangle-mesh-reverse-swept-ccd",
    "surface-static-heightfield-reverse-swept-ccd",
    "surface-kinematic-heightfield-reverse-swept-ccd",
    "surface-rotating-kinematic-triangle-mesh-speculative-ccd",
    "surface-rotating-kinematic-heightfield-speculative-ccd",
    "surface-rotating-kinematic-triangle-mesh-reverse-swept-ccd",
    "surface-rotating-kinematic-heightfield-reverse-swept-ccd",
    "surface-sphere-reverse-feature",
    "surface-capsule-reverse-feature",
    "surface-convex-reverse-feature",
    "surface-triangle-mesh-reverse-feature",
    "surface-heightfield-reverse-feature",
    "surface-skinning",
)
REQUIRED_ONES = (
    "actorCreated",
    "shapeAttached",
    "hostBuffersInitialized",
    "actorAdded",
    "actorRemoved",
    "actorReadded",
    "pinnedStable",
    "dynamicMoved",
    "boundsFinite",
    "cleanupComplete",
)
REQUIRED_ZEROES = (
    "fetchFailures",
    "nonFiniteSamples",
    "fatalErrors",
    "warningErrors",
)
DETERMINISM_KEYS = (
    "maxPinnedDrift",
    "maxDynamicDisplacement",
    "initialDynamicCentroidY",
    "finalDynamicCentroidY",
    "minY",
    "finalMinY",
    "maxSpeed",
    "finalMaxSpeed",
    "surfaceSlept",
    "maxWakeCentroidRise",
    "bufferPinnedDrift",
    "bufferRestoredDisplacement",
    "dynamicBoxFinalY",
    "dynamicBoxMaxLinearSpeed",
    "dynamicBoxFinalLinearSpeed",
    "dynamicBoxMaxAngularSpeed",
    "dynamicBoxFinalAngularSpeed",
    "dynamicBoxFinalSleeping",
    "kinematicMaxPoseError",
    "kinematicSurfaceDisplacement",
    "kinematicFinalY",
    "secondSurfaceFinalCentroidY",
    "secondSurfaceMaxDisplacement",
    "secondSurfaceMinY",
    "secondSurfaceFinalMinY",
    "secondSurfaceMaxSpeed",
    "secondSurfaceFinalMaxSpeed",
    "secondSurfaceFinalSleeping",
    "mixedVolumeFinalCentroidY",
    "mixedVolumeMaxDisplacement",
    "mixedVolumeMinY",
    "mixedVolumeFinalMinY",
    "mixedVolumeMaxSpeed",
    "mixedVolumeFinalMaxSpeed",
    "mixedVolumeFinalSleeping",
    "selfCollisionMinEnabledSeparation",
    "selfCollisionMinDisabledSeparation",
    "selfCollisionFilterMinSeparation",
    "materialFrictionLowDisplacement",
    "materialFrictionHighDisplacement",
    "materialFrictionHighFinalSpeed",
    "attachmentPinMaxDrift",
    "attachmentReleasedMaxDisplacement",
    "rigidAttachmentMaxDrift",
    "rigidAttachmentMaxRigidDisplacement",
    "rigidAttachmentMaxRigidSpeed",
    "rigidAttachmentMaxAngularDisplacement",
    "rigidAttachmentMaxAngularSpeed",
    "rigidAttachmentReleasedSeparation",
    "articulationRootMaxDisplacement",
    "articulationChildMaxForbiddenDisplacement",
    "articulationChildMaxAngularDisplacement",
    "elementFilterMinY",
    "elementFilterFinalMinY",
    "partialFilterFilteredMinY",
    "partialFilterUnfilteredMinY",
    "bendingInitialPlaneError",
    "bendingFinalPlaneError",
    "bendingZeroControlDisplacement",
    "bendingStiffDisplacement",
    "bendingMaxEdgeStrain",
    "flatteningInitialPlaneError",
    "flatteningMinimumPlaneError",
    "flatteningFinalPlaneError",
    "flatteningControlDisplacement",
    "flatteningTargetDisplacement",
    "flatteningMaxEdgeStrain",
    "motionMaxVelocityBounded",
    "motionSettlingApplied",
    "motionSettlingSlept",
    "motionControlStayedAwake",
    "motionMaxVelocityFirstStepDisplacement",
    "motionMaxVelocityFirstStepSpeed",
    "motionSettlingFinalSpeed",
    "motionControlFinalSpeed",
    "depenetrationLimitApplied",
    "depenetrationFirstStepBounded",
    "depenetrationControlSeparated",
    "depenetrationGradualRecovery",
    "depenetrationLimitedFirstStepRise",
    "depenetrationControlFirstStepRise",
    "depenetrationLimitedFinalRise",
    "depenetrationLimitedMaxSpeed",
    "speculativeCcdFlagApplied",
    "speculativeCcdPreventedTunneling",
    "speculativeCcdNegativeControlTunneled",
    "speculativeCcdPositiveMinY",
    "speculativeCcdPositiveMinSeparation",
    "speculativeCcdNegativeMaxY",
)


def parse_gate(line: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in line.split()[1:]:
        if "=" not in token:
            errors.append(f"malformed gate token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append(f"duplicate gate key: {key}")
        fields[key] = value
    return fields, errors


def require_int(
    fields: dict[str, str], key: str, expected: int, errors: list[str]
) -> None:
    try:
        value = int(fields[key])
    except (KeyError, ValueError):
        errors.append(f"{key} is missing or not an integer")
        return
    if value != expected:
        errors.append(f"{key}={value}, expected {expected}")


def require_finite_float(
    fields: dict[str, str], key: str, errors: list[str]
) -> float | None:
    try:
        value = float(fields[key])
    except (KeyError, ValueError):
        errors.append(f"{key} is missing or not a float")
        return None
    if not math.isfinite(value):
        errors.append(f"{key} is not finite")
        return None
    return value


def run_once(
    bin_dir: Path,
    frames: int,
    timeout: float,
    execution: str,
    case_name: str,
    repeat_index: int,
) -> tuple[bool, tuple[str, ...] | None]:
    executable = bin_dir / EXECUTABLE
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
    env["PHYSX_SNIPPET_SOLVER"] = "avbd"
    env["PHYSX_SNIPPET_FRAME_COUNT"] = str(frames)
    if execution == "sequential":
        env["PHYSX_AVBD_TASKGRAPH_SERIAL"] = "1"
        env["PHYSX_AVBD_SOFT_FAST_PATH"] = "0"
    else:
        env.pop("PHYSX_AVBD_TASKGRAPH_SERIAL", None)
        # Parallel acceptance validates the relaxed fast path by its physical
        # gates, not against the scalar trajectory.
        env.setdefault("PHYSX_AVBD_SOFT_FAST_PATH", "1")
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
    ogc_box_edge_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_OGC_BOX_EDGE] ")
    ]
    sphere_reverse_feature_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_SPHERE_REVERSE_FEATURE] ")
    ]
    sphere_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_SPHERE_REVERSE_SWEPT] ")
    ]
    capsule_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CAPSULE_REVERSE_SWEPT] ")
    ]
    capsule_rotational_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(
            "[AVBD_CAPSULE_ROTATIONAL_REVERSE_SWEPT] "
        )
    ]
    capsule_rotational_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CAPSULE_ROTATIONAL_SWEPT] ")
    ]
    capsule_dynamic_rotational_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(
            "[AVBD_CAPSULE_DYNAMIC_ROTATIONAL_SWEPT] "
        )
    ]
    convex_rotational_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(
            "[AVBD_CONVEX_ROTATIONAL_REVERSE_SWEPT] "
        )
    ]
    convex_rotational_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CONVEX_ROTATIONAL_SWEPT] ")
    ]
    convex_dynamic_rotational_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(
            "[AVBD_CONVEX_DYNAMIC_ROTATIONAL_SWEPT] "
        )
    ]
    convex_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CONVEX_REVERSE_SWEPT] ")
    ]
    triangle_surface_forward_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_TRIANGLE_SURFACE_FORWARD_SWEPT] ")
    ]
    triangle_surface_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_TRIANGLE_SURFACE_REVERSE_SWEPT] ")
    ]
    triangle_surface_rotational_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(
            "[AVBD_TRIANGLE_SURFACE_ROTATIONAL_SWEPT] "
        )
    ]
    deforming_soft_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(
            "[AVBD_DEFORMING_SOFT_REVERSE_SWEPT] "
        )
    ]
    deforming_soft_triangle_surface_reverse_swept_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith(
            "[AVBD_DEFORMING_SOFT_TRIANGLE_SURFACE_REVERSE_SWEPT] "
        )
    ]
    capsule_reverse_feature_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CAPSULE_REVERSE_FEATURE] ")
    ]
    convex_reverse_feature_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_CONVEX_REVERSE_FEATURE] ")
    ]
    triangle_mesh_reverse_feature_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_TRIANGLE_MESH_REVERSE_FEATURE] ")
    ]
    heightfield_reverse_feature_lines = [
        line.strip()
        for line in combined.splitlines()
        if line.startswith("[AVBD_HEIGHTFIELD_REVERSE_FEATURE] ")
    ]

    errors: list[str] = []
    if result.timed_out:
        errors.append("timed out")
    if result.visible_window_detected:
        errors.append(
            "visible window detected: "
            + ", ".join(result.visible_window_titles)
        )
    if result.returncode != 0:
        errors.append(f"exit code is {result.returncode}, expected 0")
    if len(gate_lines) != 1:
        errors.append(f"gate count is {len(gate_lines)}, expected exactly 1")
        fields: dict[str, str] = {}
    else:
        fields, parse_errors = parse_gate(gate_lines[0])
        errors.extend(parse_errors)

    if fields:
        kinematic_attachment_case = case_name in (
            "surface-kinematic-attachment",
            "surface-kinematic-element-attachment",
        )
        static_attachment_case = case_name in (
            "surface-static-attachment",
            "surface-static-element-attachment",
        )
        articulation_attachment_case = case_name in (
            "surface-articulation-attachment",
            "surface-articulation-element-attachment",
        )
        if fields.get("schema") != "1":
            errors.append("schema is not 1")
        if fields.get("snippet") != "SnippetDeformableSurfaceAVBD":
            errors.append("unexpected snippet name")
        if fields.get("solver") != "avbd":
            errors.append("solver is not avbd")
        if fields.get("case") != case_name:
            errors.append("unexpected case")
        if fields.get("execution") != execution:
            errors.append("execution mode does not match request")
        if fields.get("result") != "PASS":
            errors.append("snippet result is not PASS")
        if case_name == "surface-ogc-box-edge":
            if len(ogc_box_edge_lines) != 1:
                errors.append(
                    "OGC box-edge gate count is "
                    f"{len(ogc_box_edge_lines)}, expected exactly 1"
                )
                ogc_fields: dict[str, str] = {}
            else:
                ogc_fields, parse_errors = parse_gate(
                    ogc_box_edge_lines[0]
                )
                errors.extend(parse_errors)
            require_int(fields, "fatalErrors", 0, errors)
            require_int(fields, "warningErrors", 0, errors)
            require_int(fields, "cleanupComplete", 1, errors)
            if ogc_fields:
                require_int(ogc_fields, "frames", frames, errors)
                require_int(
                    ogc_fields, "maxInteriorEdgeHits", 0, errors
                )
                require_int(
                    ogc_fields,
                    "maxSelfEdgeIntersections",
                    0,
                    errors,
                )
                require_int(
                    ogc_fields, "nonFiniteFrames", 0, errors
                )
                for key, maximum in (
                    ("maxBoxTopHeightRange", 0.10),
                    ("maxBoxTopHeightStdDev", 0.03),
                ):
                    raw_value = ogc_fields.get(key)
                    if raw_value is None:
                        errors.append(f"missing {key}")
                        continue
                    try:
                        value = float(raw_value)
                    except ValueError:
                        errors.append(
                            f"{key} is not a float: {raw_value!r}"
                        )
                        continue
                    if value > maximum:
                        errors.append(
                            f"{key}={value:.6f}, expected <= "
                            f"{maximum:.6f}"
                        )
                if ogc_fields.get("result") != "PASS":
                    errors.append("OGC box-edge result is not PASS")
            label = f"{case_name}-r{repeat_index}"
            signature = (
                (
                    ogc_fields.get(
                        "maxInteriorEdgeHits", "<missing>"
                    ),
                    ogc_fields.get(
                        "maxSelfEdgeIntersections", "<missing>"
                    ),
                    ogc_fields.get(
                        "maxBoxTopHeightRange", "<missing>"
                    ),
                    ogc_fields.get(
                        "maxBoxTopHeightStdDev", "<missing>"
                    ),
                    ogc_fields.get(
                        "nonFiniteFrames", "<missing>"
                    ),
                )
                if ogc_fields
                else None
            )
            if errors:
                print(f"[FAIL] {label}: " + "; ".join(errors))
                if combined:
                    print(combined.rstrip())
                return False, signature
            print(
                f"[PASS] {label}: "
                "maxInteriorEdgeHits="
                f"{ogc_fields['maxInteriorEdgeHits']} "
                "maxSelfEdgeIntersections="
                f"{ogc_fields['maxSelfEdgeIntersections']} "
                "maxBoxTopHeightRange="
                f"{ogc_fields['maxBoxTopHeightRange']} "
                "maxBoxTopHeightStdDev="
                f"{ogc_fields['maxBoxTopHeightStdDev']} "
                "nonFiniteFrames="
                f"{ogc_fields['nonFiniteFrames']}"
            )
            return True, signature
        if case_name in (
            "surface-deforming-triangle-mesh-reverse-swept-ccd",
            "surface-deforming-heightfield-reverse-swept-ccd",
            "surface-static-triangle-mesh-speculative-ccd",
            "surface-kinematic-triangle-mesh-speculative-ccd",
            "surface-static-heightfield-speculative-ccd",
            "surface-kinematic-heightfield-speculative-ccd",
            "surface-static-triangle-mesh-reverse-swept-ccd",
            "surface-kinematic-triangle-mesh-reverse-swept-ccd",
            "surface-static-heightfield-reverse-swept-ccd",
            "surface-kinematic-heightfield-reverse-swept-ccd",
            "surface-rotating-kinematic-triangle-mesh-speculative-ccd",
            "surface-rotating-kinematic-heightfield-speculative-ccd",
            "surface-rotating-kinematic-triangle-mesh-reverse-swept-ccd",
            "surface-rotating-kinematic-heightfield-reverse-swept-ccd",
        ):
            rotational_case = "rotating-kinematic" in case_name
            reverse_case = "reverse-swept" in case_name
            expected_target = (
                "static"
                if (
                    case_name.startswith("surface-static-")
                    or case_name.startswith("surface-deforming-")
                )
                else "kinematic"
            )
            expected_geometry = (
                "heightfield"
                if "heightfield" in case_name
                else "triangle-mesh"
            )
            swept_lines = (
                triangle_surface_reverse_swept_lines
                if reverse_case
                else triangle_surface_forward_swept_lines
            )
            if len(swept_lines) != 1:
                errors.append(
                    "triangle-surface swept gate count is "
                    f"{len(swept_lines)}, expected exactly 1"
                )
                swept_fields: dict[str, str] = {}
            else:
                swept_fields, parse_errors = parse_gate(
                    swept_lines[0]
                )
                errors.extend(parse_errors)
            require_int(fields, "speculativeCcdFlagApplied", 1, errors)
            require_int(fields, "fatalErrors", 0, errors)
            require_int(fields, "warningErrors", 0, errors)
            require_int(fields, "cleanupComplete", 1, errors)
            if swept_fields:
                require_int(swept_fields, "frames", frames, errors)
                require_int(
                    swept_fields, "responseObserved", 1, errors
                )
                require_int(
                    swept_fields, "negativeControlPassed", 1, errors
                )
                require_int(
                    swept_fields, "vertexSweepExcluded", 1, errors
                )
                require_int(
                    swept_fields, "nonFiniteSamples", 0, errors
                )
                if swept_fields.get("target") != expected_target:
                    errors.append(
                        "triangle-surface swept target="
                        f"{swept_fields.get('target')!r}, "
                        f"expected {expected_target!r}"
                    )
                if swept_fields.get("geometry") != expected_geometry:
                    errors.append(
                        "triangle-surface swept geometry="
                        f"{swept_fields.get('geometry')!r}, "
                        f"expected {expected_geometry!r}"
                    )
                if swept_fields.get("result") != "PASS":
                    errors.append(
                        "triangle-surface swept result is not PASS"
                    )
            positive_displacement = require_finite_float(
                swept_fields, "positiveDisplacement", errors
            )
            negative_displacement = require_finite_float(
                swept_fields, "negativeDisplacement", errors
            )
            positive_drop = require_finite_float(
                swept_fields, "positiveDrop", errors
            )
            negative_drop = require_finite_float(
                swept_fields, "negativeDrop", errors
            )
            vertex_separation = require_finite_float(
                swept_fields,
                "minimumVertexSweepSeparation",
                errors,
            )
            if (
                positive_displacement is not None
                and positive_displacement
                <= (0.01 if reverse_case else 0.02)
            ):
                errors.append(
                    "triangle-surface swept response was too small"
                )
            if case_name.startswith("surface-deforming-"):
                if negative_drop is not None and negative_drop <= 0.25:
                    errors.append(
                        "flag-off deforming triangle-surface control "
                        "did not cross"
                    )
                if (
                    len(
                        deforming_soft_triangle_surface_reverse_swept_lines
                    )
                    != 1
                ):
                    errors.append(
                        "deforming triangle-surface reverse-swept gate "
                        "count is "
                        f"{len(deforming_soft_triangle_surface_reverse_swept_lines)}, "
                        "expected exactly 1"
                    )
                    deforming_fields: dict[str, str] = {}
                else:
                    deforming_fields, parse_errors = parse_gate(
                        deforming_soft_triangle_surface_reverse_swept_lines[
                            0
                        ]
                    )
                    errors.extend(parse_errors)
                if deforming_fields:
                    for key in (
                        "responseObserved",
                        "negativeControlPassed",
                        "vertexSweepExcluded",
                    ):
                        require_int(deforming_fields, key, 1, errors)
                    if deforming_fields.get("target") != "static":
                        errors.append(
                            "deforming triangle-surface target is not "
                            "static"
                        )
                    if deforming_fields.get("owner") != "reverse":
                        errors.append(
                            "deforming triangle-surface owner is not "
                            "reverse"
                        )
                    if (
                        deforming_fields.get("geometry")
                        != expected_geometry
                    ):
                        errors.append(
                            "deforming triangle-surface geometry="
                            f"{deforming_fields.get('geometry')!r}, "
                            f"expected {expected_geometry!r}"
                        )
                    if deforming_fields.get("result") != "PASS":
                        errors.append(
                            "deforming triangle-surface result is not "
                            "PASS"
                        )
                    endpoint_separation = require_finite_float(
                        deforming_fields,
                        "endpointMinSeparation",
                        errors,
                    )
                    mid_separation = require_finite_float(
                        deforming_fields,
                        "midSweepMinSeparation",
                        errors,
                    )
                    response_delta = require_finite_float(
                        deforming_fields, "responseDelta", errors
                    )
                    if (
                        endpoint_separation is not None
                        and endpoint_separation <= 0.10
                    ):
                        errors.append(
                            "deforming triangle-surface endpoints "
                            "overlap"
                        )
                    if (
                        mid_separation is not None
                        and mid_separation >= 0.01
                    ):
                        errors.append(
                            "deforming triangle-surface mid-sweep did "
                            "not hit"
                        )
                    if (
                        response_delta is not None
                        and response_delta <= 0.01
                    ):
                        errors.append(
                            "deforming triangle-surface controls did "
                            "not separate"
                        )
            elif case_name.startswith("surface-static-"):
                if negative_drop is not None and negative_drop <= 1.5:
                    errors.append(
                        "flag-off triangle-surface control did not tunnel"
                    )
                if (
                    positive_drop is not None
                    and negative_drop is not None
                    and positive_drop
                    + (0.02 if reverse_case else 0.10)
                    >= negative_drop
                ):
                    errors.append(
                        "static triangle-surface sweep did not "
                        "separate controls"
                    )
            elif (
                negative_displacement is not None
                and negative_displacement >= 0.005
            ):
                errors.append(
                    "flag-off kinematic triangle-surface control moved"
                )
            if (
                reverse_case
                and vertex_separation is not None
                and vertex_separation <= 0.10
            ):
                errors.append(
                    "forward vertex sweep was not geometrically excluded"
                )
            rotational_fields: dict[str, str] = {}
            if rotational_case:
                if len(triangle_surface_rotational_swept_lines) != 1:
                    errors.append(
                        "triangle-surface rotational swept gate count is "
                        f"{len(triangle_surface_rotational_swept_lines)}, "
                        "expected exactly 1"
                    )
                else:
                    rotational_fields, parse_errors = parse_gate(
                        triangle_surface_rotational_swept_lines[0]
                    )
                    errors.extend(parse_errors)
                if rotational_fields:
                    require_int(
                        rotational_fields, "frames", frames, errors
                    )
                    require_int(
                        rotational_fields,
                        "responseObserved",
                        1,
                        errors,
                    )
                    require_int(
                        rotational_fields,
                        "negativeControlPassed",
                        1,
                        errors,
                    )
                    require_int(
                        rotational_fields,
                        "vertexSweepExcluded",
                        1,
                        errors,
                    )
                    if rotational_fields.get("target") != "kinematic":
                        errors.append(
                            "triangle-surface rotational target is not "
                            "kinematic"
                        )
                    expected_owner = (
                        "reverse" if reverse_case else "forward"
                    )
                    if (
                        rotational_fields.get("owner")
                        != expected_owner
                    ):
                        errors.append(
                            "triangle-surface rotational owner="
                            f"{rotational_fields.get('owner')!r}, "
                            f"expected {expected_owner!r}"
                        )
                    if (
                        rotational_fields.get("geometry")
                        != expected_geometry
                    ):
                        errors.append(
                            "triangle-surface rotational geometry="
                            f"{rotational_fields.get('geometry')!r}, "
                            f"expected {expected_geometry!r}"
                        )
                    if rotational_fields.get("result") != "PASS":
                        errors.append(
                            "triangle-surface rotational result is not PASS"
                        )
                endpoint_separation = require_finite_float(
                    rotational_fields,
                    "endpointMinSeparation",
                    errors,
                )
                mid_separation = require_finite_float(
                    rotational_fields,
                    "midSweepMinSeparation",
                    errors,
                )
                positive_angular_travel = require_finite_float(
                    rotational_fields,
                    "positiveAngularTravel",
                    errors,
                )
                negative_angular_travel = require_finite_float(
                    rotational_fields,
                    "negativeAngularTravel",
                    errors,
                )
                if (
                    endpoint_separation is not None
                    and endpoint_separation <= 0.10
                ):
                    errors.append(
                        "triangle-surface rotational endpoints overlap "
                        "the soft target"
                    )
                if mid_separation is not None and (
                    mid_separation >= (-0.05 if reverse_case else 0.01)
                ):
                    errors.append(
                        "triangle-surface rotational arc did not hit "
                        "between endpoints"
                    )
                expected_travel = 2.0 * math.pi / 3.0
                for key, value in (
                    (
                        "positiveAngularTravel",
                        positive_angular_travel,
                    ),
                    (
                        "negativeAngularTravel",
                        negative_angular_travel,
                    ),
                ):
                    if (
                        value is not None
                        and abs(value - expected_travel) > 0.002
                    ):
                        errors.append(
                            f"triangle-surface {key}={value} did not "
                            f"reach {expected_travel}"
                        )
            label = f"{case_name}-r{repeat_index}"
            signature = tuple(
                swept_fields.get(key, "<missing>")
                for key in (
                    "positiveDisplacement",
                    "negativeDisplacement",
                    "positiveDrop",
                    "negativeDrop",
                    "minimumVertexSweepSeparation",
                )
            )
            if rotational_case:
                signature += tuple(
                    rotational_fields.get(key, "<missing>")
                    for key in (
                        "endpointMinSeparation",
                        "midSweepMinSeparation",
                        "positiveAngularTravel",
                        "negativeAngularTravel",
                    )
                )
            if errors:
                print(f"[FAIL] {label}: " + "; ".join(errors))
                if combined:
                    print(combined.rstrip())
                return False, signature
            print(
                f"[PASS] {label}: positiveDisplacement="
                f"{positive_displacement} negativeDisplacement="
                f"{negative_displacement} positiveDrop={positive_drop} "
                f"negativeDrop={negative_drop} "
                f"minimumVertexSweepSeparation={vertex_separation}"
            )
            return True, signature
        if case_name in (
            "surface-static-sphere-reverse-swept-ccd",
            "surface-kinematic-sphere-reverse-swept-ccd",
            "surface-dynamic-sphere-reverse-swept-ccd",
            "surface-static-capsule-reverse-swept-ccd",
            "surface-kinematic-capsule-reverse-swept-ccd",
            "surface-dynamic-capsule-reverse-swept-ccd",
            "surface-rotating-kinematic-capsule-reverse-swept-ccd",
            "surface-dynamic-rotating-capsule-reverse-swept-ccd",
            "surface-static-convex-reverse-swept-ccd",
            "surface-kinematic-convex-reverse-swept-ccd",
            "surface-dynamic-convex-reverse-swept-ccd",
            "surface-rotating-kinematic-convex-reverse-swept-ccd",
            "surface-dynamic-rotating-convex-reverse-swept-ccd",
            "surface-deforming-sphere-reverse-swept-ccd",
            "surface-deforming-capsule-reverse-swept-ccd",
            "surface-deforming-convex-reverse-swept-ccd",
        ):
            capsule_case = "-capsule-" in case_name
            convex_case = "-convex-" in case_name
            deforming_case = "surface-deforming-" in case_name
            rotational_reverse_case = (
                "-rotating-" in case_name
                and "-reverse-swept-ccd" in case_name
            )
            reverse_swept_lines = (
                convex_reverse_swept_lines
                if convex_case
                else (
                    capsule_reverse_swept_lines
                    if capsule_case
                    else sphere_reverse_swept_lines
                )
            )
            geometry_name = (
                "convex"
                if convex_case
                else ("capsule" if capsule_case else "sphere")
            )
            if len(reverse_swept_lines) != 1:
                errors.append(
                    f"{geometry_name} reverse-swept gate count is "
                    f"{len(reverse_swept_lines)}, "
                    "expected exactly 1"
                )
                reverse_fields: dict[str, str] = {}
            else:
                reverse_fields, parse_errors = parse_gate(
                    reverse_swept_lines[0]
                )
                errors.extend(parse_errors)
            require_int(fields, "speculativeCcdFlagApplied", 1, errors)
            require_int(fields, "fatalErrors", 0, errors)
            require_int(fields, "warningErrors", 0, errors)
            require_int(fields, "cleanupComplete", 1, errors)
            if reverse_fields:
                require_int(reverse_fields, "frames", frames, errors)
                require_int(
                    reverse_fields, "responseObserved", 1, errors
                )
                require_int(
                    reverse_fields, "negativeControlPassed", 1, errors
                )
                require_int(
                    reverse_fields,
                    "twoSidedResponseObserved",
                    1,
                    errors,
                )
                require_int(
                    reverse_fields, "vertexSweepExcluded", 1, errors
                )
                require_int(
                    reverse_fields, "nonFiniteSamples", 0, errors
                )
                expected_target = (
                    "static"
                    if (
                        case_name.startswith("surface-static-")
                        or deforming_case
                    )
                    else (
                        "kinematic"
                        if "kinematic" in case_name
                        else "dynamic"
                    )
                )
                if reverse_fields.get("target") != expected_target:
                    errors.append(
                        f"{geometry_name} reverse-swept target="
                        f"{reverse_fields.get('target')!r}, "
                        f"expected {expected_target!r}"
                    )
                if reverse_fields.get("result") != "PASS":
                    errors.append(
                        f"{geometry_name} reverse-swept result is not PASS"
                    )
            positive_displacement = require_finite_float(
                reverse_fields, "positiveDisplacement", errors
            )
            negative_displacement = require_finite_float(
                reverse_fields, "negativeDisplacement", errors
            )
            positive_drop = require_finite_float(
                reverse_fields, "positiveDrop", errors
            )
            negative_drop = require_finite_float(
                reverse_fields, "negativeDrop", errors
            )
            positive_rigid_drop = require_finite_float(
                reverse_fields, "positiveRigidDrop", errors
            )
            negative_rigid_drop = require_finite_float(
                reverse_fields, "negativeRigidDrop", errors
            )
            face_separation = require_finite_float(
                reverse_fields, "faceSeparation", errors
            )
            vertex_sweep_separation = require_finite_float(
                reverse_fields,
                "minimumVertexSweepSeparation",
                errors,
            )
            if (
                vertex_sweep_separation is not None
                and vertex_sweep_separation <= 0.10
            ):
                errors.append(
                    "forward vertex sweep was not geometrically excluded"
                )
            if (
                face_separation is not None
                and face_separation <= -0.15
            ):
                errors.append(
                    f"{geometry_name} crossed the swept soft feature"
                )
            if deforming_case:
                if (
                    negative_drop is not None
                    and negative_drop <= 0.25
                ):
                    errors.append(
                        "flag-off deforming soft face did not cross "
                        f"{geometry_name}"
                    )
            elif case_name.startswith("surface-static-"):
                minimum_negative_drop = (
                    1.3 if capsule_case or convex_case else 1.5
                )
                if (
                    negative_drop is not None
                    and negative_drop <= minimum_negative_drop
                ):
                    errors.append(
                        "flag-off soft face did not tunnel through "
                        f"{geometry_name}"
                    )
                if (
                    positive_drop is not None
                    and negative_drop is not None
                    and positive_drop + 0.10 >= negative_drop
                ):
                    errors.append(
                        "static reverse sweep did not separate controls"
                    )
            else:
                if (
                    positive_displacement is not None
                    and positive_displacement <= 0.02
                ):
                    errors.append(
                        f"moving {geometry_name} did not move the swept "
                        "soft face"
                    )
                if (
                    negative_displacement is not None
                    and negative_displacement >= 0.005
                ):
                    errors.append(
                        "flag-off reverse-swept soft control moved"
                    )
            if case_name.startswith("surface-dynamic-"):
                if (
                    negative_rigid_drop is not None
                    and negative_rigid_drop
                    <= (0.8 if rotational_reverse_case else 1.5)
                ):
                    errors.append(
                        f"flag-off dynamic {geometry_name} did not tunnel"
                    )
                if (
                    positive_rigid_drop is not None
                    and negative_rigid_drop is not None
                    and positive_rigid_drop + 0.05
                    >= negative_rigid_drop
                ):
                    errors.append(
                        "dynamic reverse sweep lacked two-sided response"
                    )
            rotational_fields: dict[str, str] = {}
            if rotational_reverse_case:
                rotational_lines = (
                    convex_rotational_reverse_swept_lines
                    if convex_case
                    else capsule_rotational_reverse_swept_lines
                )
                if len(rotational_lines) != 1:
                    errors.append(
                        f"{geometry_name} rotational reverse-swept gate "
                        f"count is {len(rotational_lines)}, "
                        "expected exactly 1"
                    )
                else:
                    rotational_fields, parse_errors = parse_gate(
                        rotational_lines[0]
                    )
                    errors.extend(parse_errors)
                if rotational_fields:
                    require_int(
                        rotational_fields, "frames", frames, errors
                    )
                    for key in (
                        "responseObserved",
                        "negativeControlPassed",
                        "twoSidedResponseObserved",
                        "vertexSweepExcluded",
                    ):
                        require_int(rotational_fields, key, 1, errors)
                    expected_target = (
                        "kinematic"
                        if "kinematic" in case_name
                        else "dynamic"
                    )
                    if (
                        rotational_fields.get("target")
                        != expected_target
                    ):
                        errors.append(
                            f"{geometry_name} rotational reverse-swept "
                            "target="
                            f"{rotational_fields.get('target')!r}, "
                            f"expected {expected_target!r}"
                        )
                    if rotational_fields.get("owner") != "reverse":
                        errors.append(
                            f"{geometry_name} rotational swept owner is "
                            "not reverse"
                        )
                    if rotational_fields.get("result") != "PASS":
                        errors.append(
                            f"{geometry_name} rotational reverse-swept result "
                            "is not PASS"
                        )
                    rotational_values = {
                        key: require_finite_float(
                            rotational_fields, key, errors
                        )
                        for key in (
                            "endpointMinSeparation",
                            "midSweepMinSeparation",
                            "positiveDisplacement",
                            "negativeDisplacement",
                            "positiveAngularTravel",
                            "negativeAngularTravel",
                        )
                    }
                    endpoint_separation = rotational_values[
                        "endpointMinSeparation"
                    ]
                    mid_separation = rotational_values[
                        "midSweepMinSeparation"
                    ]
                    rotational_positive_displacement = rotational_values[
                        "positiveDisplacement"
                    ]
                    rotational_negative_displacement = rotational_values[
                        "negativeDisplacement"
                    ]
                    positive_angular_travel = rotational_values[
                        "positiveAngularTravel"
                    ]
                    negative_angular_travel = rotational_values[
                        "negativeAngularTravel"
                    ]
                    if (
                        endpoint_separation is not None
                        and endpoint_separation <= 0.05
                    ):
                        errors.append(
                            f"{geometry_name} rotational sweep endpoints "
                            "overlap "
                            "the soft face"
                        )
                    if (
                        mid_separation is not None
                        and mid_separation >= -0.05
                    ):
                        errors.append(
                            f"{geometry_name} rotational sweep does not "
                            "cross the "
                            "soft face between endpoints"
                        )
                    if (
                        rotational_positive_displacement is not None
                        and rotational_positive_displacement <= 0.02
                    ):
                        errors.append(
                            f"{geometry_name} rotational reverse owner "
                            "did not move "
                            "the positive soft face"
                        )
                    if (
                        rotational_negative_displacement is not None
                        and rotational_negative_displacement >= 0.005
                    ):
                        errors.append(
                            f"flag-off {geometry_name} rotational reverse "
                            "control "
                            "moved"
                        )
                    if expected_target == "kinematic":
                        expected_travel = 2.0 * math.pi / 3.0
                        for key, value in (
                            (
                                "positiveAngularTravel",
                                positive_angular_travel,
                            ),
                            (
                                "negativeAngularTravel",
                                negative_angular_travel,
                            ),
                        ):
                            if (
                                value is not None
                                and abs(value - expected_travel) > 0.002
                            ):
                                errors.append(
                                    f"kinematic {key}={value} did not "
                                    f"reach target {expected_travel}"
                                )
                    elif (
                        negative_angular_travel is not None
                        and negative_angular_travel <= 0.8
                    ):
                        errors.append(
                            f"flag-off dynamic {geometry_name} did not rotate "
                            "ballistically"
                        )
                    elif (
                        positive_angular_travel is not None
                        and negative_angular_travel is not None
                        and positive_angular_travel + 0.05
                        >= negative_angular_travel
                    ):
                        errors.append(
                            f"dynamic {geometry_name} rotational reverse "
                            "sweep "
                            "lacked two-sided angular response"
                        )
            deforming_fields: dict[str, str] = {}
            if deforming_case:
                if len(deforming_soft_reverse_swept_lines) != 1:
                    errors.append(
                        "deforming soft reverse-swept gate count is "
                        f"{len(deforming_soft_reverse_swept_lines)}, "
                        "expected exactly 1"
                    )
                else:
                    deforming_fields, parse_errors = parse_gate(
                        deforming_soft_reverse_swept_lines[0]
                    )
                    errors.extend(parse_errors)
                if deforming_fields:
                    for key in (
                        "responseObserved",
                        "negativeControlPassed",
                        "vertexSweepExcluded",
                    ):
                        require_int(deforming_fields, key, 1, errors)
                    if deforming_fields.get("geometry") != geometry_name:
                        errors.append(
                            "deforming reverse geometry="
                            f"{deforming_fields.get('geometry')!r}, "
                            f"expected {geometry_name!r}"
                        )
                    if deforming_fields.get("target") != "static":
                        errors.append(
                            "deforming reverse target is not static"
                        )
                    if deforming_fields.get("owner") != "reverse":
                        errors.append(
                            "deforming soft swept owner is not reverse"
                        )
                    if deforming_fields.get("result") != "PASS":
                        errors.append(
                            "deforming soft reverse-swept result is not PASS"
                        )
                    deforming_values = {
                        key: require_finite_float(
                            deforming_fields, key, errors
                        )
                        for key in (
                            "endpointMinSeparation",
                            "midSweepMinSeparation",
                            "minimumVertexSweepSeparation",
                            "responseDelta",
                        )
                    }
                    if (
                        deforming_values["endpointMinSeparation"]
                        is not None
                        and deforming_values["endpointMinSeparation"]
                        <= 0.10
                    ):
                        errors.append(
                            "deforming reverse endpoints are not separated"
                        )
                    expected_mid_limit = (
                        0.01 if convex_case else -0.02
                    )
                    if (
                        deforming_values["midSweepMinSeparation"]
                        is not None
                        and deforming_values["midSweepMinSeparation"]
                        >= expected_mid_limit
                    ):
                        errors.append(
                            "deforming soft feature did not cross "
                            f"{geometry_name} between endpoints"
                        )
                    if (
                        deforming_values["minimumVertexSweepSeparation"]
                        is not None
                        and deforming_values[
                            "minimumVertexSweepSeparation"
                        ]
                        <= 0.10
                    ):
                        errors.append(
                            "deforming reverse fixture did not exclude "
                            "all soft vertex sweeps"
                        )
                    if (
                        deforming_values["responseDelta"] is not None
                        and deforming_values["responseDelta"] <= 0.01
                    ):
                        errors.append(
                            "deforming reverse response did not separate "
                            "flag-on/off controls"
                        )
            label = f"{case_name}-r{repeat_index}"
            signature = tuple(
                reverse_fields.get(key, "<missing>")
                for key in (
                    "positiveDisplacement",
                    "negativeDisplacement",
                    "positiveDrop",
                    "negativeDrop",
                    "positiveRigidDrop",
                    "negativeRigidDrop",
                    "faceSeparation",
                    "minimumVertexSweepSeparation",
                )
            )
            if rotational_reverse_case:
                signature += tuple(
                    rotational_fields.get(key, "<missing>")
                    for key in (
                        "endpointMinSeparation",
                        "midSweepMinSeparation",
                        "positiveDisplacement",
                        "negativeDisplacement",
                        "positiveAngularTravel",
                        "negativeAngularTravel",
                    )
                )
            if deforming_case:
                signature += tuple(
                    deforming_fields.get(key, "<missing>")
                    for key in (
                        "endpointMinSeparation",
                        "midSweepMinSeparation",
                        "minimumVertexSweepSeparation",
                        "responseDelta",
                    )
                )
            if errors:
                print(f"[FAIL] {label}: " + "; ".join(errors))
                if combined:
                    print(combined.rstrip())
                return False, signature
            print(
                f"[PASS] {label}: positiveDisplacement="
                f"{positive_displacement} negativeDisplacement="
                f"{negative_displacement} positiveDrop={positive_drop} "
                f"negativeDrop={negative_drop} positiveRigidDrop="
                f"{positive_rigid_drop} negativeRigidDrop="
                f"{negative_rigid_drop} faceSeparation="
                f"{face_separation} minimumVertexSweepSeparation="
                f"{vertex_sweep_separation}"
            )
            return True, signature
        if case_name in (
            "surface-sphere-reverse-feature",
            "surface-capsule-reverse-feature",
            "surface-convex-reverse-feature",
            "surface-triangle-mesh-reverse-feature",
            "surface-heightfield-reverse-feature",
        ):
            if case_name == "surface-triangle-mesh-reverse-feature":
                reverse_feature_lines = (
                    triangle_mesh_reverse_feature_lines
                )
                geometry_name = "triangle-mesh"
            elif case_name == "surface-heightfield-reverse-feature":
                reverse_feature_lines = heightfield_reverse_feature_lines
                geometry_name = "heightfield"
            elif case_name == "surface-convex-reverse-feature":
                reverse_feature_lines = convex_reverse_feature_lines
                geometry_name = "convex"
            elif case_name == "surface-capsule-reverse-feature":
                reverse_feature_lines = capsule_reverse_feature_lines
                geometry_name = "capsule"
            else:
                reverse_feature_lines = sphere_reverse_feature_lines
                geometry_name = "sphere"
            if len(reverse_feature_lines) != 1:
                errors.append(
                    f"{geometry_name} reverse-feature gate count is "
                    f"{len(reverse_feature_lines)}, "
                    "expected exactly 1"
                )
                reverse_fields: dict[str, str] = {}
            else:
                reverse_fields, parse_errors = parse_gate(
                    reverse_feature_lines[0]
                )
                errors.extend(parse_errors)
            require_int(fields, "fatalErrors", 0, errors)
            require_int(fields, "warningErrors", 0, errors)
            require_int(fields, "cleanupComplete", 1, errors)
            if reverse_fields:
                require_int(reverse_fields, "frames", frames, errors)
                require_int(
                    reverse_fields, "faceResponseObserved", 1, errors
                )
                require_int(
                    reverse_fields, "vertexSdfExcluded", 1, errors
                )
                require_int(
                    reverse_fields, "negativeControlPassed", 1, errors
                )
                require_int(
                    reverse_fields, "nonFiniteSamples", 0, errors
                )
                if reverse_fields.get("result") != "PASS":
                    errors.append(
                        f"{geometry_name} reverse-feature result is not PASS"
                    )
            positive_displacement = require_finite_float(
                reverse_fields, "positiveDisplacement", errors
            )
            positive_drop = require_finite_float(
                reverse_fields, "positiveDrop", errors
            )
            negative_drop = require_finite_float(
                reverse_fields, "negativeDrop", errors
            )
            face_separation = require_finite_float(
                reverse_fields, "faceSeparation", errors
            )
            vertex_separation = require_finite_float(
                reverse_fields, "minimumVertexSeparation", errors
            )
            if (
                positive_displacement is not None
                and positive_displacement <= 0.001
            ):
                errors.append("reverse feature did not move the surface")
            if negative_drop is not None and negative_drop <= 0.02:
                errors.append("free surface control did not move")
            if (
                positive_drop is not None
                and negative_drop is not None
                and positive_drop + 0.01 >= negative_drop
            ):
                errors.append(
                    "reverse feature did not separate from free control"
                )
            if face_separation is not None and face_separation <= 0.02:
                errors.append(
                    f"{geometry_name} crossed the soft edge/face"
                )
            if vertex_separation is not None and vertex_separation <= 0.10:
                errors.append("vertex SDF was not geometrically excluded")
            label = f"{case_name}-r{repeat_index}"
            signature = tuple(
                reverse_fields.get(key, "<missing>")
                for key in (
                    "positiveDisplacement",
                    "positiveDrop",
                    "negativeDrop",
                    "faceSeparation",
                    "minimumVertexSeparation",
                )
            )
            if errors:
                print(f"[FAIL] {label}: " + "; ".join(errors))
                if combined:
                    print(combined.rstrip())
                return False, signature
            print(
                f"[PASS] {label}: positiveDisplacement="
                f"{positive_displacement} positiveDrop={positive_drop} "
                f"negativeDrop={negative_drop} "
                f"faceSeparation={face_separation} "
                f"minimumVertexSeparation={vertex_separation}"
            )
            return True, signature
        if case_name == "surface-max-depenetration-velocity":
            for key in (
                "depenetrationLimitApplied",
                "depenetrationFirstStepBounded",
                "depenetrationControlSeparated",
                "depenetrationGradualRecovery",
                "cleanupComplete",
            ):
                require_int(fields, key, 1, errors)
            for key in (
                "fetchFailures",
                "nonFiniteSamples",
                "fatalErrors",
                "warningErrors",
            ):
                require_int(fields, key, 0, errors)
            limited_rise = require_finite_float(
                fields, "depenetrationLimitedFirstStepRise", errors
            )
            control_rise = require_finite_float(
                fields, "depenetrationControlFirstStepRise", errors
            )
            final_rise = require_finite_float(
                fields, "depenetrationLimitedFinalRise", errors
            )
            limited_speed = require_finite_float(
                fields, "depenetrationLimitedMaxSpeed", errors
            )
            step_limit = 0.12 * 0.0166666675
            if limited_rise is not None and not (
                -1.0e-6 <= limited_rise <= step_limit * 1.05
            ):
                errors.append(
                    "limited first-step rise escaped the public cap"
                )
            if (
                limited_rise is not None
                and control_rise is not None
                and control_rise <= limited_rise + 5.0e-3
            ):
                errors.append(
                    "unlimited control did not separate from limited actor"
                )
            if (
                limited_rise is not None
                and final_rise is not None
                and final_rise <= limited_rise + 4.0e-3
            ):
                errors.append("limited actor did not recover gradually")
            if limited_speed is not None and limited_speed > 0.12 * 1.05:
                errors.append(
                    "limited actor exceeded max depenetration speed"
                )
            label = f"{case_name}-r{repeat_index}"
            signature = tuple(
                fields.get(key, "<missing>")
                for key in DETERMINISM_KEYS
            )
            if errors:
                print(f"[FAIL] {label}: " + "; ".join(errors))
                if combined:
                    print(combined.rstrip())
                return False, signature
            print(
                f"[PASS] {label}: limitedRise={limited_rise} "
                f"controlRise={control_rise} finalRise={final_rise} "
                f"limitedMaxSpeed={limited_speed}"
            )
            return True, signature
        if case_name in (
            "surface-dynamic-sphere-relative-swept-ccd",
            "surface-dynamic-capsule-relative-swept-ccd",
            "surface-dynamic-rotating-capsule-relative-swept-ccd",
            "surface-dynamic-convex-relative-swept-ccd",
            "surface-dynamic-rotating-convex-relative-swept-ccd",
            "surface-static-sphere-reverse-swept-ccd",
            "surface-kinematic-sphere-reverse-swept-ccd",
            "surface-dynamic-sphere-reverse-swept-ccd",
            "surface-static-capsule-reverse-swept-ccd",
            "surface-kinematic-capsule-reverse-swept-ccd",
            "surface-dynamic-capsule-reverse-swept-ccd",
        ):
            rotational_dynamic_case = (
                case_name
                in (
                    "surface-dynamic-rotating-capsule-relative-swept-ccd",
                    "surface-dynamic-rotating-convex-relative-swept-ccd",
                )
            )
            rotational_convex_case = (
                case_name
                == "surface-dynamic-rotating-convex-relative-swept-ccd"
            )
            for key in (
                "speculativeCcdFlagApplied",
                "dynamicSphereSweepLaunched",
                "dynamicSphereSweepResponseObserved",
                "dynamicSphereSweepNegativeControlTunneled",
                "dynamicSphereSweepTwoSidedResponseObserved",
                "cleanupComplete",
            ):
                require_int(fields, key, 1, errors)
            for key in (
                "fetchFailures",
                "nonFiniteSamples",
                "fatalErrors",
                "warningErrors",
            ):
                require_int(fields, key, 0, errors)
            positive_soft_displacement = require_finite_float(
                fields,
                "dynamicSphereSweepPositiveSoftDisplacement",
                errors,
            )
            negative_soft_displacement = require_finite_float(
                fields,
                "dynamicSphereSweepNegativeSoftDisplacement",
                errors,
            )
            positive_rigid_drop = require_finite_float(
                fields, "dynamicSphereSweepPositiveRigidDrop", errors
            )
            negative_rigid_drop = require_finite_float(
                fields, "dynamicSphereSweepNegativeRigidDrop", errors
            )
            min_separation = require_finite_float(
                fields,
                "dynamicSphereSweepPositiveMinSeparation",
                errors,
            )
            if (
                positive_soft_displacement is not None
                and positive_soft_displacement <= 0.02
            ):
                errors.append(
                    "dynamic finite-geometry sweep did not move the "
                    "positive surface"
                )
            if (
                negative_soft_displacement is not None
                and negative_soft_displacement >= 0.005
            ):
                errors.append(
                    "flag-off dynamic finite-geometry surface control moved"
                )
            if (
                negative_rigid_drop is not None
                and negative_rigid_drop
                <= (0.2 if rotational_dynamic_case else 1.5)
            ):
                errors.append(
                    "flag-off dynamic finite geometry did not advance "
                    "through the test arc"
                )
            if (
                positive_rigid_drop is not None
                and negative_rigid_drop is not None
                and (
                    abs(positive_rigid_drop - negative_rigid_drop) <= 0.05
                    if rotational_dynamic_case
                    else positive_rigid_drop + 0.05
                    >= negative_rigid_drop
                )
            ):
                errors.append(
                    "swept contact did not produce a distinguishable "
                    "two-sided rigid response"
                )
            if min_separation is not None and (
                min_separation >= 1.0e30 or min_separation <= -0.15
            ):
                errors.append(
                    "dynamic finite-geometry swept response missed or crossed"
                )
            label = f"{case_name}-r{repeat_index}"
            signature = (
                fields.get(
                    "dynamicSphereSweepPositiveSoftDisplacement",
                    "<missing>",
                ),
                fields.get(
                    "dynamicSphereSweepNegativeSoftDisplacement",
                    "<missing>",
                ),
                fields.get(
                    "dynamicSphereSweepPositiveRigidDrop", "<missing>"
                ),
                fields.get(
                    "dynamicSphereSweepNegativeRigidDrop", "<missing>"
                ),
                fields.get(
                    "dynamicSphereSweepPositiveMinSeparation",
                    "<missing>",
                ),
            )
            if rotational_dynamic_case:
                rotational_fields: dict[str, str] = {}
                rotational_lines = (
                    convex_dynamic_rotational_swept_lines
                    if rotational_convex_case
                    else capsule_dynamic_rotational_swept_lines
                )
                geometry_name = (
                    "convex" if rotational_convex_case else "capsule"
                )
                if len(rotational_lines) != 1:
                    errors.append(
                        f"dynamic {geometry_name} rotational swept gate "
                        f"count is {len(rotational_lines)}, "
                        "expected exactly 1"
                    )
                else:
                    rotational_fields, parse_errors = parse_gate(
                        rotational_lines[0]
                    )
                    errors.extend(parse_errors)
                if rotational_fields:
                    require_int(rotational_fields, "frames", frames, errors)
                    require_int(
                        rotational_fields, "responseObserved", 1, errors
                    )
                    require_int(
                        rotational_fields,
                        "negativeControlPassed",
                        1,
                        errors,
                    )
                    require_int(
                        rotational_fields,
                        "twoSidedResponseObserved",
                        1,
                        errors,
                    )
                    if rotational_fields.get("target") != "dynamic":
                        errors.append(
                            f"dynamic {geometry_name} rotational target "
                            "is not "
                            "dynamic"
                        )
                    if rotational_fields.get("owner") != "forward":
                        errors.append(
                            f"dynamic {geometry_name} rotational owner "
                            "is not "
                            "forward"
                        )
                    if rotational_fields.get("result") != "PASS":
                        errors.append(
                            f"dynamic {geometry_name} rotational result "
                            "is not PASS"
                        )
                    rotational_values = {
                        key: require_finite_float(
                            rotational_fields, key, errors
                        )
                        for key in (
                            "endpointMinSeparation",
                            "midSweepMinSeparation",
                            "positiveDisplacement",
                            "negativeDisplacement",
                            "positiveAngularTravel",
                            "negativeAngularTravel",
                        )
                    }
                    endpoint_separation = rotational_values[
                        "endpointMinSeparation"
                    ]
                    mid_separation = rotational_values[
                        "midSweepMinSeparation"
                    ]
                    rotational_positive = rotational_values[
                        "positiveDisplacement"
                    ]
                    rotational_negative = rotational_values[
                        "negativeDisplacement"
                    ]
                    positive_angular = rotational_values[
                        "positiveAngularTravel"
                    ]
                    negative_angular = rotational_values[
                        "negativeAngularTravel"
                    ]
                    if (
                        endpoint_separation is not None
                        and endpoint_separation <= 0.05
                    ):
                        errors.append(
                            f"dynamic {geometry_name} rotational endpoints "
                            "are "
                            "not separated"
                        )
                    if mid_separation is not None and (
                        mid_separation
                        >= (-0.05 if not rotational_convex_case else 1.0e-5)
                    ):
                        errors.append(
                            f"dynamic {geometry_name} rotational fixture "
                            "does not "
                            "isolate an interior arc hit"
                        )
                    if (
                        rotational_positive is not None
                        and rotational_positive <= 0.02
                    ):
                        errors.append(
                            f"dynamic {geometry_name} rotational flag-on "
                            "response "
                            "was not observed"
                        )
                    if (
                        rotational_negative is not None
                        and rotational_negative >= 0.005
                    ):
                        errors.append(
                            f"dynamic {geometry_name} rotational flag-off "
                            "control "
                            "moved"
                        )
                    if (
                        negative_angular is not None
                        and negative_angular <= 0.2
                    ):
                        errors.append(
                            f"dynamic {geometry_name} flag-off angular "
                            "control did not advance through the test arc"
                        )
                    if (
                        positive_angular is not None
                        and negative_angular is not None
                        and abs(positive_angular - negative_angular) <= 0.05
                    ):
                        errors.append(
                            f"dynamic {geometry_name} rotational contact "
                            "did not produce a distinguishable two-sided "
                            "angular response"
                        )
                    signature += tuple(
                        rotational_fields.get(key, "<missing>")
                        for key in (
                            "endpointMinSeparation",
                            "midSweepMinSeparation",
                            "positiveDisplacement",
                            "negativeDisplacement",
                            "positiveAngularTravel",
                            "negativeAngularTravel",
                        )
                    )
            if errors:
                print(f"[FAIL] {label}: " + "; ".join(errors))
                if combined:
                    print(combined.rstrip())
                return False, signature
            print(
                f"[PASS] {label}: positiveSoftDisplacement="
                f"{positive_soft_displacement} negativeSoftDisplacement="
                f"{negative_soft_displacement} positiveRigidDrop="
                f"{positive_rigid_drop} negativeRigidDrop="
                f"{negative_rigid_drop} minSeparation={min_separation}"
            )
            return True, signature
        if case_name in (
            "surface-moving-kinematic-sphere-speculative-ccd",
            "surface-moving-kinematic-capsule-speculative-ccd",
            "surface-rotating-kinematic-capsule-speculative-ccd",
            "surface-moving-kinematic-convex-speculative-ccd",
            "surface-rotating-kinematic-convex-speculative-ccd",
            "surface-rotating-kinematic-convex-speculative-ccd",
        ):
            for key in (
                "speculativeCcdFlagApplied",
                "movingSphereTargetIssued",
                "movingSphereCcdResponseObserved",
                "movingSphereNegativeControlHeld",
                "cleanupComplete",
            ):
                require_int(fields, key, 1, errors)
            for key in (
                "fetchFailures",
                "nonFiniteSamples",
                "fatalErrors",
                "warningErrors",
            ):
                require_int(fields, key, 0, errors)
            positive_displacement = require_finite_float(
                fields, "movingSpherePositiveDisplacement", errors
            )
            negative_displacement = require_finite_float(
                fields, "movingSphereNegativeDisplacement", errors
            )
            min_separation = require_finite_float(
                fields, "movingSpherePositiveMinSeparation", errors
            )
            if (
                positive_displacement is not None
                and positive_displacement <= 0.02
            ):
                errors.append(
                    "moving kinematic finite geometry did not produce "
                    "a CCD response"
                )
            if (
                negative_displacement is not None
                and negative_displacement >= 0.005
            ):
                errors.append(
                    "flag-off moving finite-geometry control unexpectedly "
                    "moved"
                )
            if min_separation is not None and (
                min_separation >= 1.0e30 or min_separation <= -0.10
            ):
                errors.append(
                    "moving finite-geometry response missed or crossed "
                    "the geometry"
                )
            label = f"{case_name}-r{repeat_index}"
            signature = (
                fields.get("movingSpherePositiveDisplacement", "<missing>"),
                fields.get("movingSphereNegativeDisplacement", "<missing>"),
                fields.get("movingSpherePositiveMinSeparation", "<missing>"),
            )
            rotational_fields: dict[str, str] = {}
            rotational_kinematic_case = case_name in (
                "surface-rotating-kinematic-capsule-speculative-ccd",
                "surface-rotating-kinematic-convex-speculative-ccd",
            )
            if rotational_kinematic_case:
                rotational_convex_case = (
                    case_name
                    == "surface-rotating-kinematic-convex-speculative-ccd"
                )
                rotational_lines = (
                    convex_rotational_swept_lines
                    if rotational_convex_case
                    else capsule_rotational_swept_lines
                )
                geometry_name = (
                    "convex" if rotational_convex_case else "capsule"
                )
                if len(rotational_lines) != 1:
                    errors.append(
                        f"{geometry_name} rotational swept gate count is "
                        f"{len(rotational_lines)}, "
                        "expected exactly 1"
                    )
                else:
                    rotational_fields, parse_errors = parse_gate(
                        rotational_lines[0]
                    )
                    errors.extend(parse_errors)
                if rotational_fields:
                    require_int(rotational_fields, "frames", frames, errors)
                    require_int(
                        rotational_fields, "responseObserved", 1, errors
                    )
                    require_int(
                        rotational_fields,
                        "negativeControlPassed",
                        1,
                        errors,
                    )
                    if rotational_fields.get("target") != "kinematic":
                        errors.append(
                            f"{geometry_name} rotational swept target "
                            "is not "
                            "kinematic"
                        )
                    if rotational_fields.get("owner") != "forward":
                        errors.append(
                            f"{geometry_name} rotational swept owner is "
                            "not forward"
                        )
                    if rotational_fields.get("result") != "PASS":
                        errors.append(
                            f"{geometry_name} rotational swept result "
                            "is not PASS"
                        )
                    endpoint_min_separation = require_finite_float(
                        rotational_fields,
                        "endpointMinSeparation",
                        errors,
                    )
                    mid_sweep_min_separation = require_finite_float(
                        rotational_fields,
                        "midSweepMinSeparation",
                        errors,
                    )
                    rotational_positive_displacement = require_finite_float(
                        rotational_fields,
                        "positiveDisplacement",
                        errors,
                    )
                    rotational_negative_displacement = require_finite_float(
                        rotational_fields,
                        "negativeDisplacement",
                        errors,
                    )
                    if (
                        endpoint_min_separation is not None
                        and endpoint_min_separation <= 0.05
                    ):
                        errors.append(
                            f"{geometry_name} rotational swept fixture "
                            "endpoints "
                            "are not separated"
                        )
                    if mid_sweep_min_separation is not None and (
                        mid_sweep_min_separation
                        >= (-0.05 if not rotational_convex_case else 1.0e-5)
                    ):
                        errors.append(
                            f"{geometry_name} rotational swept fixture "
                            "does not "
                            "isolate an intermediate arc hit"
                        )
                    if (
                        rotational_positive_displacement is not None
                        and rotational_positive_displacement <= 0.02
                    ):
                        errors.append(
                            f"{geometry_name} rotational swept flag-on "
                            "response "
                            "was not observed"
                        )
                    if (
                        rotational_negative_displacement is not None
                        and rotational_negative_displacement >= 0.005
                    ):
                        errors.append(
                            f"{geometry_name} rotational swept flag-off "
                            "control "
                            "unexpectedly moved"
                        )
                    signature += (
                        rotational_fields.get(
                            "endpointMinSeparation", "<missing>"
                        ),
                        rotational_fields.get(
                            "midSweepMinSeparation", "<missing>"
                        ),
                        rotational_fields.get(
                            "positiveDisplacement", "<missing>"
                        ),
                        rotational_fields.get(
                            "negativeDisplacement", "<missing>"
                        ),
                    )
            if errors:
                print(f"[FAIL] {label}: " + "; ".join(errors))
                if combined:
                    print(combined.rstrip())
                return False, signature
            print(
                f"[PASS] {label}: positiveDisplacement="
                f"{positive_displacement} negativeDisplacement="
                f"{negative_displacement} minSeparation={min_separation}"
            )
            return True, signature
        if case_name in (
            "surface-speculative-ccd",
            "surface-plane-speculative-ccd",
            "surface-sphere-speculative-ccd",
            "surface-capsule-speculative-ccd",
            "surface-convex-speculative-ccd",
        ):
            for key in (
                "speculativeCcdFlagApplied",
                "speculativeCcdPreventedTunneling",
                "cleanupComplete",
            ):
                require_int(fields, key, 1, errors)
            plane_case = case_name == "surface-plane-speculative-ccd"
            finite_geometry_case = case_name in (
                "surface-sphere-speculative-ccd",
                "surface-capsule-speculative-ccd",
                "surface-convex-speculative-ccd",
            )
            if not plane_case:
                require_int(
                    fields,
                    "speculativeCcdNegativeControlTunneled",
                    1,
                    errors,
                )
            for key in (
                "fetchFailures",
                "nonFiniteSamples",
                "fatalErrors",
                "warningErrors",
            ):
                require_int(fields, key, 0, errors)
            positive_min_y = require_finite_float(
                fields, "speculativeCcdPositiveMinY", errors
            )
            positive_min_separation = require_finite_float(
                fields,
                "speculativeCcdPositiveMinSeparation",
                errors,
            )
            negative_max_y = require_finite_float(
                fields, "speculativeCcdNegativeMaxY", errors
            )
            positive_floor = 0.49 if plane_case else 0.54
            if (
                not finite_geometry_case
                and
                positive_min_y is not None
                and positive_min_y < positive_floor
            ):
                errors.append(
                    "speculative actor crossed the collision boundary"
                )
            if (
                finite_geometry_case
                and positive_min_separation is not None
                and (
                    positive_min_separation >= 1.0e30
                    or positive_min_separation < -0.05
                )
            ):
                errors.append(
                    "speculative actor finite-geometry separation was missing "
                    "or penetrated the boundary"
                )
            if (
                not plane_case
                and
                negative_max_y is not None
                and negative_max_y > 0.44
            ):
                errors.append(
                    "discrete negative control did not tunnel"
                )
            label = f"{case_name}-r{repeat_index}"
            signature = tuple(
                fields.get(key, "<missing>")
                for key in DETERMINISM_KEYS
            )
            if errors:
                print(f"[FAIL] {label}: " + "; ".join(errors))
                if combined:
                    print(combined.rstrip())
                return False, signature
            print(
                f"[PASS] {label}: positiveMinY={positive_min_y} "
                f"positiveMinSeparation={positive_min_separation} "
                f"negativeMaxY={negative_max_y}"
            )
            return True, signature
        for key in REQUIRED_ONES:
            require_int(fields, key, 1, errors)
        for key in REQUIRED_ZEROES:
            require_int(fields, key, 0, errors)
        pinned_drift = require_finite_float(
            fields, "maxPinnedDrift", errors
        )
        dynamic_displacement = require_finite_float(
            fields, "maxDynamicDisplacement", errors
        )
        initial_centroid = require_finite_float(
            fields, "initialDynamicCentroidY", errors
        )
        final_centroid = require_finite_float(
            fields, "finalDynamicCentroidY", errors
        )
        if pinned_drift is not None and pinned_drift > 1.0e-4:
            errors.append(f"maxPinnedDrift={pinned_drift} exceeds 1e-4")
        if (
            dynamic_displacement is not None
            and dynamic_displacement < 1.0e-2
        ):
            errors.append(
                "maxDynamicDisplacement is below the movement threshold"
            )
        if (
            initial_centroid is not None
            and final_centroid is not None
            and case_name not in (
                "surface-kinematic-box",
                "surface-kinematic-sphere",
                "surface-kinematic-capsule",
                "surface-kinematic-convex",
                "surface-kinematic-triangle-mesh",
                "surface-kinematic-heightfield",
                "surface-rigid-attachment",
                "surface-rigid-element-attachment",
                "surface-static-attachment",
                "surface-static-element-attachment",
                "surface-kinematic-attachment",
                "surface-kinematic-element-attachment",
                "surface-articulation-attachment",
                "surface-articulation-element-attachment",
                "surface-surface-attachment",
                "surface-volume-attachment",
                "surface-soft-soft-element-filter",
                "surface-volume-element-filter",
                "volume-volume-element-filter",
                "surface-flattening",
                "surface-motion-controls",
                "surface-material-friction",
            )
            and final_centroid >= initial_centroid - 1.0e-3
        ):
            errors.append("dynamic centroid did not move downward")
        if case_name in ("surface-ground", "surface-sleep-wake"):
            require_int(fields, "groundAdded", 1, errors)
            require_int(fields, "groundContactObserved", 1, errors)
            require_int(fields, "groundPenetrationBounded", 1, errors)
            require_int(fields, "groundSettled", 1, errors)
            require_int(fields, "surfaceSlept", 1, errors)
            min_y = require_finite_float(fields, "minY", errors)
            final_min_y = require_finite_float(
                fields, "finalMinY", errors
            )
            max_speed = require_finite_float(
                fields, "maxSpeed", errors
            )
            final_max_speed = require_finite_float(
                fields, "finalMaxSpeed", errors
            )
            if min_y is not None and min_y >= 0.1:
                errors.append("surface never reached the ground")
            if min_y is not None and min_y <= -0.05:
                errors.append("surface exceeded the penetration tolerance")
            if final_min_y is not None and abs(final_min_y) >= 0.05:
                errors.append("surface did not settle near the ground")
            if max_speed is not None and max_speed >= 20.0:
                errors.append("surface speed exceeded the stability bound")
            if (
                final_max_speed is not None
                and final_max_speed >= 1.0e-3
            ):
                errors.append("surface did not settle to rest")
        if case_name == "surface-sleep-wake":
            for key in (
                "initialSleepObserved",
                "velocityWakeIssued",
                "velocityWakeObserved",
                "movedAfterVelocityWake",
            ):
                require_int(fields, key, 1, errors)
            wake_rise = require_finite_float(
                fields, "maxWakeCentroidRise", errors
            )
            if wake_rise is not None and wake_rise <= 1.0e-3:
                errors.append("surface did not move after velocity wake")
        if case_name == "surface-buffer-mutation":
            for key in (
                "bufferMutationIssued",
                "bufferMutationApplied",
                "bufferPinHeld",
                "bufferInvMassRestored",
                "bufferRestoredMoved",
            ):
                require_int(fields, key, 1, errors)
            pinned_drift = require_finite_float(
                fields, "bufferPinnedDrift", errors
            )
            restored_displacement = require_finite_float(
                fields, "bufferRestoredDisplacement", errors
            )
            if pinned_drift is not None and pinned_drift > 1.0e-4:
                errors.append("mutated pinned vertex drifted")
            if (
                restored_displacement is not None
                and restored_displacement <= 1.0e-3
            ):
                errors.append("restored vertex did not resume movement")
        if case_name in (
            "surface-dynamic-box",
            "surface-dynamic-sphere",
            "surface-dynamic-capsule",
            "surface-dynamic-convex",
        ):
            dynamic_sphere = case_name in (
                "surface-dynamic-sphere",
                "surface-dynamic-capsule",
            )
            dynamic_convex = case_name == "surface-dynamic-convex"
            for key in (
                "dynamicBoxAdded",
                "dynamicBoxInitiallySleeping",
                "dynamicBoxWoke",
            ):
                require_int(fields, key, 1, errors)
            dynamic_drop = require_finite_float(
                fields, "dynamicBoxMaxDrop", errors
            )
            dynamic_final_y = require_finite_float(
                fields, "dynamicBoxFinalY", errors
            )
            dynamic_max_linear_speed = require_finite_float(
                fields, "dynamicBoxMaxLinearSpeed", errors
            )
            dynamic_final_linear_speed = require_finite_float(
                fields, "dynamicBoxFinalLinearSpeed", errors
            )
            dynamic_max_angular_speed = require_finite_float(
                fields, "dynamicBoxMaxAngularSpeed", errors
            )
            dynamic_final_angular_speed = require_finite_float(
                fields, "dynamicBoxFinalAngularSpeed", errors
            )
            surface_min_y = require_finite_float(fields, "minY", errors)
            surface_max_speed = require_finite_float(
                fields, "maxSpeed", errors
            )
            surface_final_speed = require_finite_float(
                fields, "finalMaxSpeed", errors
            )
            if dynamic_convex:
                if (
                    dynamic_drop is not None
                    and not 0.5 < dynamic_drop < 1.2
                ):
                    errors.append(
                        "dynamic convex response left the bounded drop range"
                    )
                if (
                    dynamic_final_y is not None
                    and not -0.8 < dynamic_final_y < 0.0
                ):
                    errors.append(
                        "dynamic convex did not remain on the bounded floor"
                    )
            elif dynamic_sphere:
                if (
                    dynamic_drop is not None
                    and not 0.05 < dynamic_drop < 1.5
                ):
                    errors.append(
                        "dynamic sphere response left the bounded drop range"
                    )
                if (
                    dynamic_final_y is not None
                    and not -0.4 < dynamic_final_y < 0.4
                ):
                    errors.append(
                        "dynamic sphere did not remain on the bounded floor"
                    )
            else:
                if (
                    dynamic_drop is not None
                    and not 0.5 < dynamic_drop < 1.2
                ):
                    errors.append(
                        "dynamic box response left the bounded drop range"
                    )
                if (
                    dynamic_final_y is not None
                    and not -0.8 < dynamic_final_y < -0.6
                ):
                    errors.append(
                        "dynamic box did not remain on the bounded floor"
                    )
            if (
                dynamic_max_linear_speed is not None
                and dynamic_max_linear_speed >= 5.0
            ):
                errors.append("dynamic box linear speed exceeded the bound")
            if (
                dynamic_max_angular_speed is not None
                and dynamic_max_angular_speed >= 5.0
            ):
                errors.append("dynamic box angular speed exceeded the bound")
            if (
                dynamic_final_linear_speed is not None
                and dynamic_final_linear_speed >= 0.05
            ):
                errors.append("dynamic box retained excessive linear speed")
            if (
                dynamic_final_angular_speed is not None
                and dynamic_final_angular_speed >= 0.05
            ):
                errors.append("dynamic box retained excessive angular speed")
            if surface_min_y is not None and surface_min_y <= -1.05:
                errors.append("surface crossed the dynamic-box floor")
            if surface_max_speed is not None and surface_max_speed >= 10.0:
                errors.append("surface speed exceeded the mixed-rigid bound")
            if (
                surface_final_speed is not None
                and surface_final_speed >= 0.5
            ):
                errors.append("surface retained excessive mixed-rigid speed")
            if dynamic_displacement is not None and dynamic_displacement >= 5.0:
                errors.append("surface escaped the mixed-rigid fixture")
        if case_name in (
            "surface-kinematic-box",
            "surface-kinematic-sphere",
            "surface-kinematic-capsule",
            "surface-kinematic-convex",
            "surface-kinematic-triangle-mesh",
            "surface-kinematic-heightfield",
        ):
            for key in (
                "kinematicBoxAdded",
                "initialSleepObserved",
                "kinematicTargetIssued",
                "kinematicTargetReached",
                "kinematicSurfaceWoke",
                "kinematicSurfaceMoved",
                "kinematicContactObserved",
            ):
                require_int(fields, key, 1, errors)
            pose_error = require_finite_float(
                fields, "kinematicMaxPoseError", errors
            )
            surface_displacement = require_finite_float(
                fields, "kinematicSurfaceDisplacement", errors
            )
            final_y = require_finite_float(
                fields, "kinematicFinalY", errors
            )
            surface_max_speed = require_finite_float(
                fields, "maxSpeed", errors
            )
            surface_final_speed = require_finite_float(
                fields, "finalMaxSpeed", errors
            )
            if pose_error is not None and pose_error > 1.0e-4:
                errors.append(
                    "kinematic rigid did not follow its prescribed target"
                )
            if (
                surface_displacement is not None
                and surface_displacement <= 0.02
            ):
                errors.append(
                    "kinematic rigid did not move the sleeping surface"
                )
            if final_y is not None and abs(final_y - 2.35) > 1.0e-4:
                errors.append(
                    "kinematic rigid ended away from its prescribed target"
                )
            if surface_max_speed is not None and surface_max_speed >= 2.0:
                errors.append(
                    "kinematic coupling amplified the surface speed"
                )
            if (
                surface_final_speed is not None
                and surface_final_speed >= 0.5
            ):
                errors.append(
                    "kinematic coupling left excessive residual speed"
                )
        if case_name in (
            "surface-soft-soft-wake",
            "surface-soft-soft-swept-ccd",
        ):
            for key in (
                "secondSurfaceCreated",
                "secondSurfaceAdded",
                "secondSurfaceInitiallySleeping",
                "secondSurfaceWoke",
                "secondSurfaceMoved",
            ):
                require_int(fields, key, 1, errors)
            second_displacement = require_finite_float(
                fields, "secondSurfaceMaxDisplacement", errors
            )
            second_final_centroid = require_finite_float(
                fields, "secondSurfaceFinalCentroidY", errors
            )
            second_min_y = require_finite_float(
                fields, "secondSurfaceMinY", errors
            )
            second_final_min_y = require_finite_float(
                fields, "secondSurfaceFinalMinY", errors
            )
            second_max_speed = require_finite_float(
                fields, "secondSurfaceMaxSpeed", errors
            )
            second_final_speed = require_finite_float(
                fields, "secondSurfaceFinalMaxSpeed", errors
            )
            surface_min_y = require_finite_float(fields, "minY", errors)
            surface_max_speed = require_finite_float(
                fields, "maxSpeed", errors
            )
            surface_final_speed = require_finite_float(
                fields, "finalMaxSpeed", errors
            )
            minimum_second_displacement = (
                0.01
                if case_name == "surface-soft-soft-swept-ccd"
                else 0.05
            )
            if (
                second_displacement is not None
                and not minimum_second_displacement
                < second_displacement
                < 2.0
            ):
                errors.append(
                    "target surface response left the bounded movement range"
                )
            if (
                second_final_centroid is not None
                and not 0.0 <= second_final_centroid <= 2.0
            ):
                errors.append("target surface centroid escaped the fixture")
            if second_min_y is not None and second_min_y <= -0.05:
                errors.append("target surface exceeded ground penetration")
            if second_final_min_y is not None and second_final_min_y <= -0.05:
                errors.append("target surface ended below the ground bound")
            maximum_speed = (
                80.0
                if case_name == "surface-soft-soft-swept-ccd"
                else 10.0
            )
            if (
                second_max_speed is not None
                and second_max_speed >= maximum_speed
            ):
                errors.append("target surface speed exceeded the bound")
            if (
                case_name != "surface-soft-soft-swept-ccd"
                and second_final_speed is not None
                and second_final_speed >= 0.1
            ):
                errors.append("target surface did not settle to a bounded tail")
            if surface_min_y is not None and surface_min_y <= -0.05:
                errors.append("driving surface exceeded ground penetration")
            if (
                surface_max_speed is not None
                and surface_max_speed >= maximum_speed
            ):
                errors.append("driving surface speed exceeded the bound")
            if case_name == "surface-soft-soft-swept-ccd":
                wake_frame = require_finite_float(
                    fields, "secondSurfaceWakeFrame", errors
                )
                if wake_frame is not None and wake_frame > 1:
                    errors.append(
                        "swept soft-soft contact did not wake the target "
                        "at first impact"
                    )
            if (
                case_name != "surface-soft-soft-swept-ccd"
                and surface_final_speed is not None
                and surface_final_speed >= 0.1
            ):
                errors.append("driving surface did not settle to a bounded tail")
            if dynamic_displacement is not None and dynamic_displacement >= 5.0:
                errors.append("driving surface escaped the soft-soft fixture")
        if case_name == "surface-volume-wake":
            for key in (
                "mixedVolumeCreated",
                "mixedVolumeAdded",
                "mixedVolumeInitiallySleeping",
                "mixedVolumeWoke",
                "mixedVolumeMoved",
            ):
                require_int(fields, key, 1, errors)
            mixed_displacement = require_finite_float(
                fields, "mixedVolumeMaxDisplacement", errors
            )
            mixed_final_centroid = require_finite_float(
                fields, "mixedVolumeFinalCentroidY", errors
            )
            mixed_min_y = require_finite_float(
                fields, "mixedVolumeMinY", errors
            )
            mixed_final_min_y = require_finite_float(
                fields, "mixedVolumeFinalMinY", errors
            )
            mixed_max_speed = require_finite_float(
                fields, "mixedVolumeMaxSpeed", errors
            )
            mixed_final_speed = require_finite_float(
                fields, "mixedVolumeFinalMaxSpeed", errors
            )
            surface_min_y = require_finite_float(fields, "minY", errors)
            surface_max_speed = require_finite_float(
                fields, "maxSpeed", errors
            )
            surface_final_speed = require_finite_float(
                fields, "finalMaxSpeed", errors
            )
            if (
                mixed_displacement is not None
                and not 0.05 < mixed_displacement < 2.0
            ):
                errors.append(
                    "volume response left the bounded movement range"
                )
            if (
                mixed_final_centroid is not None
                and not 0.0 <= mixed_final_centroid <= 2.0
            ):
                errors.append("volume centroid escaped the fixture")
            if mixed_min_y is not None and mixed_min_y <= -0.05:
                errors.append("volume exceeded ground penetration")
            if mixed_final_min_y is not None and mixed_final_min_y <= -0.05:
                errors.append("volume ended below the ground bound")
            if mixed_max_speed is not None and mixed_max_speed >= 10.0:
                errors.append("volume speed exceeded the mixed-soft bound")
            if mixed_final_speed is not None and mixed_final_speed >= 1.0:
                errors.append("volume retained unbounded tail speed")
            if surface_min_y is not None and surface_min_y <= -0.05:
                errors.append("driving surface exceeded ground penetration")
            if surface_max_speed is not None and surface_max_speed >= 10.0:
                errors.append("driving surface speed exceeded the bound")
            if surface_final_speed is not None and surface_final_speed >= 1.0:
                errors.append("driving surface retained unbounded tail speed")
            if dynamic_displacement is not None and dynamic_displacement >= 5.0:
                errors.append("driving surface escaped the surface-volume fixture")
        if case_name in (
            "surface-self-collision",
            "surface-self-collision-swept-ccd",
        ):
            for key in (
                "selfCollisionEnabled",
                "selfCollisionFilterApplied",
                "selfCollisionPreventedCrossing",
                "selfCollisionDisableIssued",
                "selfCollisionDisabledCrossed",
            ):
                require_int(fields, key, 1, errors)
            enabled_separation = require_finite_float(
                fields, "selfCollisionMinEnabledSeparation", errors
            )
            disabled_separation = require_finite_float(
                fields, "selfCollisionMinDisabledSeparation", errors
            )
            if (
                enabled_separation is not None
                and enabled_separation <= -0.02
            ):
                errors.append(
                    "enabled self collision did not prevent crossing"
                )
            if (
                disabled_separation is not None
                and disabled_separation >= -0.05
            ):
                errors.append(
                    "disabled self collision unexpectedly prevented crossing"
                )
        if case_name == "surface-self-collision-filter":
            for key in (
                "selfCollisionEnabled",
                "selfCollisionFilterApplied",
                "selfCollisionFilterExcludedPair",
            ):
                require_int(fields, key, 1, errors)
            filter_separation = require_finite_float(
                fields, "selfCollisionFilterMinSeparation", errors
            )
            if (
                filter_separation is not None
                and filter_separation >= -0.05
            ):
                errors.append(
                    "rest-near self-collision pair was not filtered"
                )
        if case_name == "surface-material-friction":
            for key in (
                "materialFrictionLowApplied",
                "materialFrictionHighApplied",
                "materialFrictionResponseObserved",
            ):
                require_int(fields, key, 1, errors)
            low_displacement = require_finite_float(
                fields, "materialFrictionLowDisplacement", errors
            )
            high_displacement = require_finite_float(
                fields, "materialFrictionHighDisplacement", errors
            )
            high_final_speed = require_finite_float(
                fields, "materialFrictionHighFinalSpeed", errors
            )
            final_min_y = require_finite_float(
                fields, "finalMinY", errors
            )
            if (
                low_displacement is not None
                and low_displacement <= 0.2
            ):
                errors.append(
                    "low-friction control did not slide far enough"
                )
            if (
                low_displacement is not None
                and high_displacement is not None
                and high_displacement >= 0.5 * low_displacement
            ):
                errors.append(
                    "high deformable friction did not reduce sliding"
                )
            if high_final_speed is not None and high_final_speed >= 0.2:
                errors.append(
                    "high deformable friction retained excessive speed"
                )
            if final_min_y is not None and final_min_y <= -0.05:
                errors.append(
                    "friction fixture exceeded ground penetration"
                )
        if case_name in (
            "surface-world-pin",
            "surface-world-element-attachment",
        ):
            for key in (
                "attachmentCreated",
                "attachmentPinned",
                "attachmentReleased",
                "attachmentMovedAfterRelease",
            ):
                require_int(fields, key, 1, errors)
            attachment_drift = require_finite_float(
                fields, "attachmentPinMaxDrift", errors
            )
            released_displacement = require_finite_float(
                fields, "attachmentReleasedMaxDisplacement", errors
            )
            if attachment_drift is not None and attachment_drift > 1.0e-4:
                errors.append("world-attached surface vertex drifted")
            if (
                released_displacement is not None
                and released_displacement <= 1.0e-3
            ):
                errors.append(
                    "released world-attached vertex did not resume movement"
                )
        if case_name in (
            "surface-surface-attachment",
            "surface-volume-attachment",
        ):
            for key in (
                "attachmentCreated",
                "attachmentPinned",
                "rigidAttachmentHeldAcrossReadd",
                "attachmentReleased",
                "attachmentMovedAfterRelease",
            ):
                require_int(fields, key, 1, errors)
            attachment_drift = require_finite_float(
                fields, "attachmentPinMaxDrift", errors
            )
            released_separation = require_finite_float(
                fields, "attachmentReleasedMaxDisplacement", errors
            )
            source_speed = require_finite_float(
                fields, "maxSpeed", errors
            )
            if attachment_drift is not None and attachment_drift >= 0.05:
                errors.append(
                    "deformable-pair attachment drift exceeded 5cm"
                )
            if (
                released_separation is not None
                and released_separation <= 0.1
            ):
                errors.append(
                    "deformable pair did not separate after release"
                )
            if source_speed is not None and source_speed >= 5.0:
                errors.append(
                    "deformable-pair source speed exceeded the bound"
                )
            if case_name == "surface-surface-attachment":
                for key in (
                    "secondSurfaceCreated",
                    "secondSurfaceAdded",
                    "secondSurfaceInitiallySleeping",
                    "secondSurfaceWoke",
                    "secondSurfaceMoved",
                ):
                    require_int(fields, key, 1, errors)
                target_displacement = require_finite_float(
                    fields, "secondSurfaceMaxDisplacement", errors
                )
                target_speed = require_finite_float(
                    fields, "secondSurfaceMaxSpeed", errors
                )
            else:
                for key in (
                    "mixedVolumeCreated",
                    "mixedVolumeAdded",
                    "mixedVolumeInitiallySleeping",
                    "mixedVolumeWoke",
                    "mixedVolumeMoved",
                ):
                    require_int(fields, key, 1, errors)
                target_displacement = require_finite_float(
                    fields, "mixedVolumeMaxDisplacement", errors
                )
                target_speed = require_finite_float(
                    fields, "mixedVolumeMaxSpeed", errors
                )
            if (
                target_displacement is not None
                and target_displacement <= 1.0e-3
            ):
                errors.append(
                    "deformable-pair target did not move"
                )
            if target_speed is not None and target_speed >= 5.0:
                errors.append(
                    "deformable-pair target speed exceeded the bound"
                )
        if case_name in (
            "surface-rigid-attachment",
            "surface-rigid-element-attachment",
            "surface-static-attachment",
            "surface-static-element-attachment",
            "surface-kinematic-attachment",
            "surface-kinematic-element-attachment",
            "surface-articulation-attachment",
            "surface-articulation-element-attachment",
        ):
            common_attachment_keys = (
                "rigidAttachmentActorAdded",
                "rigidAttachmentInitiallySleeping",
                "rigidAttachmentCreated",
                "rigidAttachmentRigidMoved",
                "rigidAttachmentHeldAcrossReadd",
                "rigidAttachmentReleased",
                "rigidAttachmentSeparatedAfterRelease",
            )
            for key in common_attachment_keys:
                require_int(fields, key, 1, errors)
            if case_name in (
                "surface-rigid-attachment",
                "surface-rigid-element-attachment",
            ):
                require_int(
                    fields, "rigidAttachmentRigidWoke", 1, errors
                )
                require_int(
                    fields, "rigidAttachmentRigidRotated", 1, errors
                )
            elif kinematic_attachment_case or static_attachment_case:
                for key in (
                    "kinematicBoxAdded",
                    "kinematicTargetIssued",
                    "kinematicTargetReached",
                    "kinematicSurfaceWoke",
                    "kinematicSurfaceMoved",
                ):
                    require_int(fields, key, 1, errors)
                require_int(
                    fields, "rigidAttachmentRigidRotated", 1, errors
                )
            else:
                for key in (
                    "articulationCreated",
                    "articulationAdded",
                    "articulationInitiallySleeping",
                    "articulationWoke",
                    "articulationJointSubspaceHeld",
                    "articulationRootStable",
                ):
                    require_int(fields, key, 1, errors)
            attachment_drift = require_finite_float(
                fields, "rigidAttachmentMaxDrift", errors
            )
            rigid_displacement = require_finite_float(
                fields, "rigidAttachmentMaxRigidDisplacement", errors
            )
            rigid_speed = require_finite_float(
                fields, "rigidAttachmentMaxRigidSpeed", errors
            )
            angular_displacement = require_finite_float(
                fields,
                "rigidAttachmentMaxAngularDisplacement",
                errors,
            )
            angular_speed = require_finite_float(
                fields, "rigidAttachmentMaxAngularSpeed", errors
            )
            released_separation = require_finite_float(
                fields, "rigidAttachmentReleasedSeparation", errors
            )
            kinematic_pose_error = None
            if kinematic_attachment_case or static_attachment_case:
                kinematic_pose_error = require_finite_float(
                    fields, "kinematicMaxPoseError", errors
                )
                soft_max_speed = require_finite_float(
                    fields, "maxSpeed", errors
                )
                soft_final_speed = require_finite_float(
                    fields, "finalMaxSpeed", errors
                )
            elif articulation_attachment_case:
                articulation_root_displacement = require_finite_float(
                    fields, "articulationRootMaxDisplacement", errors
                )
                articulation_forbidden_displacement = require_finite_float(
                    fields,
                    "articulationChildMaxForbiddenDisplacement",
                    errors,
                )
                articulation_angular_displacement = require_finite_float(
                    fields,
                    "articulationChildMaxAngularDisplacement",
                    errors,
                )
                soft_max_speed = require_finite_float(
                    fields, "maxSpeed", errors
                )
                soft_final_speed = require_finite_float(
                    fields, "finalMaxSpeed", errors
                )
            else:
                articulation_root_displacement = None
                articulation_forbidden_displacement = None
                articulation_angular_displacement = None
                soft_max_speed = None
                soft_final_speed = None
            if not articulation_attachment_case:
                articulation_root_displacement = None
                articulation_forbidden_displacement = None
                articulation_angular_displacement = None
            if attachment_drift is not None and attachment_drift >= 0.05:
                errors.append(
                    "surface vertex-to-rigid attachment drift exceeded 5cm"
                )
            if rigid_displacement is not None and rigid_displacement <= 0.02:
                errors.append(
                    "surface attachment target did not move"
                )
            if rigid_speed is not None and rigid_speed >= 5.0:
                errors.append(
                    "surface vertex-to-rigid response exceeded speed bound"
                )
            if (
                not articulation_attachment_case
                and
                angular_displacement is not None
                and angular_displacement <= 0.02
            ):
                errors.append(
                    "off-center surface attachment did not rotate rigid"
                )
            if (
                not articulation_attachment_case
                and not static_attachment_case
                and angular_speed is not None
            ):
                if angular_speed <= 0.02:
                    errors.append(
                        "off-center surface attachment did not publish "
                        "angular velocity"
                    )
                elif angular_speed >= 5.0:
                    errors.append(
                        "off-center surface attachment exceeded angular "
                        "speed bound"
                    )
            if (
                released_separation is not None
                and released_separation <= 0.2
            ):
                errors.append(
                    "surface and rigid did not separate after release"
                )
            if (
                kinematic_pose_error is not None
                and kinematic_pose_error > 1.0e-4
            ):
                errors.append(
                    "prescribed attachment actor missed its target"
                )
            if soft_max_speed is not None and soft_max_speed >= 5.0:
                errors.append(
                    "prescribed attachment amplified surface speed"
                )
            if soft_final_speed is not None and soft_final_speed >= 2.0:
                errors.append(
                    "attachment left excessive surface speed"
                )
            if (
                articulation_root_displacement is not None
                and articulation_root_displacement > 1.0e-4
            ):
                errors.append("articulation fixed root moved")
            if (
                articulation_forbidden_displacement is not None
                and articulation_forbidden_displacement > 1.0e-3
            ):
                errors.append(
                    "articulation child left its prismatic subspace"
                )
            if (
                articulation_angular_displacement is not None
                and articulation_angular_displacement > 1.0e-3
            ):
                errors.append(
                    "prismatic articulation child rotated"
                )
        if case_name in (
            "surface-element-filter",
            "surface-partial-element-filter",
            "surface-soft-soft-element-filter",
            "surface-volume-element-filter",
            "volume-volume-element-filter",
        ):
            for key in (
                "elementFilterCreated",
                "elementFilterHeldAcrossReadd",
                "elementFilterSuppressedContact",
                "elementFilterReleased",
                "elementFilterContactRestored",
            ):
                require_int(fields, key, 1, errors)
            filtered_min_y = require_finite_float(
                fields, "elementFilterMinY", errors
            )
            restored_min_y = require_finite_float(
                fields, "elementFilterFinalMinY", errors
            )
            final_speed = require_finite_float(
                fields, "finalMaxSpeed", errors
            )
            if filtered_min_y is not None and filtered_min_y >= -0.2:
                errors.append(
                    "filtered deformable elements did not suppress contact"
                )
            if restored_min_y is not None:
                if case_name in (
                    "surface-volume-element-filter",
                    "volume-volume-element-filter",
                ):
                    if not -0.05 < restored_min_y < 0.1:
                        errors.append(
                            "deformable-pair contact did not recover on the "
                            "target shell after element-filter release"
                        )
                elif abs(restored_min_y) >= 0.05:
                    errors.append(
                        "contact did not recover after element-filter release"
                    )
            if final_speed is not None and final_speed >= 0.1:
                errors.append(
                    "deformable actor did not settle after "
                    "element-filter release"
                )
        if case_name in (
            "surface-partial-element-filter",
            "surface-soft-soft-element-filter",
            "surface-volume-element-filter",
            "volume-volume-element-filter",
        ):
            for key in (
                "partialFilterExactOwnership",
                "partialFilterUnfilteredContactHeld",
            ):
                require_int(fields, key, 1, errors)
            filtered_min_y = require_finite_float(
                fields, "partialFilterFilteredMinY", errors
            )
            unfiltered_min_y = require_finite_float(
                fields, "partialFilterUnfilteredMinY", errors
            )
            if filtered_min_y is not None and filtered_min_y >= -0.2:
                errors.append(
                    "filtered element did not pass through its contact target"
                )
            if unfiltered_min_y is not None and unfiltered_min_y <= -0.05:
                errors.append(
                    "unfiltered element lost its contact target"
                )
        if case_name == "surface-bending":
            for key in (
                "bendingMaterialPairCreated",
                "bendingZeroControlHeld",
                "bendingResponseObserved",
                "bendingMembraneIsolated",
            ):
                require_int(fields, key, 1, errors)
            initial_error = require_finite_float(
                fields, "bendingInitialPlaneError", errors
            )
            final_error = require_finite_float(
                fields, "bendingFinalPlaneError", errors
            )
            zero_displacement = require_finite_float(
                fields, "bendingZeroControlDisplacement", errors
            )
            stiff_displacement = require_finite_float(
                fields, "bendingStiffDisplacement", errors
            )
            max_edge_strain = require_finite_float(
                fields, "bendingMaxEdgeStrain", errors
            )
            if (
                initial_error is not None
                and final_error is not None
                and final_error >= initial_error - 0.05
            ):
                errors.append(
                    "positive bending stiffness did not reduce dihedral error"
                )
            if (
                zero_displacement is not None
                and zero_displacement > 1.0e-4
            ):
                errors.append(
                    "zero-bending negative control unexpectedly moved"
                )
            if (
                stiff_displacement is not None
                and stiff_displacement <= 0.05
            ):
                errors.append(
                    "positive-bending surface lacked a restoring response"
                )
            if max_edge_strain is not None and max_edge_strain >= 0.05:
                errors.append(
                    "bending fixture introduced excessive membrane strain"
                )
        if case_name == "surface-flattening":
            for key in (
                "flatteningFlagApplied",
                "flatteningControlHeld",
                "flatteningResponseObserved",
                "flatteningRetargetObserved",
                "flatteningMembraneIsolated",
            ):
                require_int(fields, key, 1, errors)
            initial_error = require_finite_float(
                fields, "flatteningInitialPlaneError", errors
            )
            minimum_error = require_finite_float(
                fields, "flatteningMinimumPlaneError", errors
            )
            final_error = require_finite_float(
                fields, "flatteningFinalPlaneError", errors
            )
            control_displacement = require_finite_float(
                fields, "flatteningControlDisplacement", errors
            )
            target_displacement = require_finite_float(
                fields, "flatteningTargetDisplacement", errors
            )
            max_edge_strain = require_finite_float(
                fields, "flatteningMaxEdgeStrain", errors
            )
            if (
                initial_error is not None
                and minimum_error is not None
                and minimum_error >= initial_error - 0.05
            ):
                errors.append(
                    "flattening did not reduce the curved rest angle"
                )
            if (
                minimum_error is not None
                and final_error is not None
                and final_error <= minimum_error + 0.05
            ):
                errors.append(
                    "clearing flattening did not restore the curved rest angle"
                )
            if (
                control_displacement is not None
                and control_displacement > 1.0e-4
            ):
                errors.append(
                    "flattening-disabled negative control unexpectedly moved"
                )
            if (
                target_displacement is not None
                and target_displacement <= 0.05
            ):
                errors.append("flattening-enabled surface did not move")
            if max_edge_strain is not None and max_edge_strain >= 0.05:
                errors.append(
                    "flattening fixture introduced excessive membrane strain"
                )
        if case_name == "surface-motion-controls":
            for key in (
                "motionMaxVelocityBounded",
                "motionSettlingApplied",
                "motionSettlingSlept",
                "motionControlStayedAwake",
            ):
                require_int(fields, key, 1, errors)
            first_displacement = require_finite_float(
                fields, "motionMaxVelocityFirstStepDisplacement", errors
            )
            first_speed = require_finite_float(
                fields, "motionMaxVelocityFirstStepSpeed", errors
            )
            settling_speed = require_finite_float(
                fields, "motionSettlingFinalSpeed", errors
            )
            control_speed = require_finite_float(
                fields, "motionControlFinalSpeed", errors
            )
            if (
                first_displacement is not None
                and first_displacement > 0.0166666675 * 1.01
            ):
                errors.append(
                    "maxLinearVelocity did not bound first-frame displacement"
                )
            if first_speed is not None and first_speed > 1.01:
                errors.append(
                    "maxLinearVelocity did not bound first-frame speed"
                )
            if settling_speed is not None and settling_speed > 1.0e-6:
                errors.append("settling actor did not reach zero-speed sleep")
            if control_speed is not None and not 0.07 <= control_speed <= 0.09:
                errors.append(
                    "no-settling control did not retain its commanded speed"
                )
            if (
                settling_speed is not None
                and control_speed is not None
                and settling_speed >= control_speed
            ):
                errors.append(
                    "settling policy did not separate target from control"
                )

    label = f"{case_name}-r{repeat_index}"
    signature = (
        tuple(fields.get(key, "<missing>") for key in DETERMINISM_KEYS)
        if fields
        else None
    )
    if errors:
        print(f"[FAIL] {label}: " + "; ".join(errors))
        if combined:
            print(combined.rstrip())
        return False, signature

    print(
        f"[PASS] {label}: "
        f"maxPinnedDrift={fields['maxPinnedDrift']} "
        f"maxDynamicDisplacement={fields['maxDynamicDisplacement']} "
        f"centroidY={fields['initialDynamicCentroidY']}"
        f"->{fields['finalDynamicCentroidY']} "
        f"minY={fields['minY']}->{fields['finalMinY']} "
        f"maxSpeed={fields['maxSpeed']}->{fields['finalMaxSpeed']} "
        f"surfaceSlept={fields['surfaceSlept']} "
        f"maxWakeCentroidRise={fields['maxWakeCentroidRise']} "
        f"bufferPinnedDrift={fields['bufferPinnedDrift']} "
        f"bufferRestoredDisplacement="
        f"{fields['bufferRestoredDisplacement']} "
        f"dynamicBoxWakeFrame={fields['dynamicBoxWakeFrame']} "
        f"dynamicBoxMaxDrop={fields['dynamicBoxMaxDrop']} "
        f"dynamicBoxY={fields['dynamicBoxInitialY']}"
        f"->{fields['dynamicBoxFinalY']} "
        f"dynamicBoxSpeed={fields['dynamicBoxMaxLinearSpeed']}"
        f"->{fields['dynamicBoxFinalLinearSpeed']} "
        f"dynamicBoxAngularSpeed={fields['dynamicBoxMaxAngularSpeed']}"
        f"->{fields['dynamicBoxFinalAngularSpeed']} "
        f"dynamicBoxFinalSleeping={fields['dynamicBoxFinalSleeping']} "
        f"kinematicPoseError={fields['kinematicMaxPoseError']} "
        f"kinematicSurfaceDisplacement="
        f"{fields['kinematicSurfaceDisplacement']} "
        f"kinematicFinalY={fields['kinematicFinalY']} "
        f"secondSurfaceWakeFrame={fields['secondSurfaceWakeFrame']} "
        f"secondSurfaceCentroidY="
        f"{fields['secondSurfaceInitialCentroidY']}"
        f"->{fields['secondSurfaceFinalCentroidY']} "
        f"secondSurfaceSpeed={fields['secondSurfaceMaxSpeed']}"
        f"->{fields['secondSurfaceFinalMaxSpeed']} "
        f"secondSurfaceMinY={fields['secondSurfaceMinY']}"
        f"->{fields['secondSurfaceFinalMinY']} "
        f"secondSurfaceFinalSleeping="
        f"{fields['secondSurfaceFinalSleeping']} "
        f"mixedVolumeWakeFrame={fields['mixedVolumeWakeFrame']} "
        f"mixedVolumeCentroidY="
        f"{fields['mixedVolumeInitialCentroidY']}"
        f"->{fields['mixedVolumeFinalCentroidY']} "
        f"mixedVolumeSpeed={fields['mixedVolumeMaxSpeed']}"
        f"->{fields['mixedVolumeFinalMaxSpeed']} "
        f"mixedVolumeMinY={fields['mixedVolumeMinY']}"
        f"->{fields['mixedVolumeFinalMinY']} "
        f"mixedVolumeFinalSleeping={fields['mixedVolumeFinalSleeping']} "
        f"selfCollisionSeparation="
        f"{fields['selfCollisionMinEnabledSeparation']}"
        f"->{fields['selfCollisionMinDisabledSeparation']} "
        f"selfCollisionFilterSeparation="
        f"{fields['selfCollisionFilterMinSeparation']} "
        f"materialFrictionDisplacement="
        f"{fields['materialFrictionLowDisplacement']}"
        f"->{fields['materialFrictionHighDisplacement']} "
        f"materialFrictionHighFinalSpeed="
        f"{fields['materialFrictionHighFinalSpeed']} "
        f"attachmentDrift={fields['attachmentPinMaxDrift']} "
        f"attachmentReleasedDisplacement="
        f"{fields['attachmentReleasedMaxDisplacement']} "
        f"rigidAttachmentDrift={fields['rigidAttachmentMaxDrift']} "
        f"rigidAttachmentRigidDisplacement="
        f"{fields['rigidAttachmentMaxRigidDisplacement']} "
        f"rigidAttachmentRigidSpeed="
        f"{fields['rigidAttachmentMaxRigidSpeed']} "
        f"rigidAttachmentAngularDisplacement="
        f"{fields['rigidAttachmentMaxAngularDisplacement']} "
        f"rigidAttachmentAngularSpeed="
        f"{fields['rigidAttachmentMaxAngularSpeed']} "
        f"rigidAttachmentReleasedSeparation="
        f"{fields['rigidAttachmentReleasedSeparation']} "
        f"articulationRootDisplacement="
        f"{fields['articulationRootMaxDisplacement']} "
        f"articulationForbiddenDisplacement="
        f"{fields['articulationChildMaxForbiddenDisplacement']} "
        f"articulationAngularDisplacement="
        f"{fields['articulationChildMaxAngularDisplacement']} "
        f"partialFilterMinY={fields['partialFilterFilteredMinY']}"
        f"/{fields['partialFilterUnfilteredMinY']} "
        f"bendingPlaneError={fields['bendingInitialPlaneError']}"
        f"->{fields['bendingFinalPlaneError']} "
        f"bendingZeroDisplacement="
        f"{fields['bendingZeroControlDisplacement']} "
        f"bendingStiffDisplacement="
        f"{fields['bendingStiffDisplacement']} "
        f"bendingMaxEdgeStrain={fields['bendingMaxEdgeStrain']} "
        f"flatteningPlaneError="
        f"{fields['flatteningInitialPlaneError']}"
        f"->{fields['flatteningMinimumPlaneError']}"
        f"->{fields['flatteningFinalPlaneError']} "
        f"flatteningControlDisplacement="
        f"{fields['flatteningControlDisplacement']} "
        f"flatteningTargetDisplacement="
        f"{fields['flatteningTargetDisplacement']} "
        f"flatteningMaxEdgeStrain={fields['flatteningMaxEdgeStrain']} "
        f"motionMaxVelocityBounded="
        f"{fields['motionMaxVelocityBounded']} "
        f"motionSettlingApplied={fields['motionSettlingApplied']} "
        f"motionSettlingSlept={fields['motionSettlingSlept']} "
        f"motionControlStayedAwake="
        f"{fields['motionControlStayedAwake']} "
        f"motionMaxVelocityFirstStepDisplacement="
        f"{fields['motionMaxVelocityFirstStepDisplacement']} "
        f"motionMaxVelocityFirstStepSpeed="
        f"{fields['motionMaxVelocityFirstStepSpeed']} "
        f"motionSettlingFinalSpeed="
        f"{fields['motionSettlingFinalSpeed']} "
        f"motionControlFinalSpeed={fields['motionControlFinalSpeed']}"
    )
    return True, signature


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--case", choices=CASES, default="surface-lifecycle"
    )
    parser.add_argument(
        "--execution",
        choices=("parallel", "sequential"),
        default="parallel",
    )
    args = parser.parse_args()

    if args.frames < 3:
        parser.error("--frames must be at least 3")
    if (
        args.case == "surface-max-depenetration-velocity"
        and args.frames < 8
    ):
        parser.error(
            "surface-max-depenetration-velocity requires at least 8 frames"
        )
    if args.case == "surface-ogc-box-edge" and args.frames < 180:
        parser.error("surface-ogc-box-edge requires at least 180 frames")
    if args.case == "surface-motion-controls" and args.frames < 30:
        parser.error("surface-motion-controls requires at least 30 frames")
    if (
        args.case
        in (
            "surface-dynamic-box",
            "surface-dynamic-sphere",
            "surface-dynamic-capsule",
            "surface-dynamic-convex",
            "surface-kinematic-box",
            "surface-kinematic-sphere",
            "surface-kinematic-capsule",
            "surface-kinematic-convex",
            "surface-soft-soft-wake",
            "surface-soft-soft-swept-ccd",
            "surface-volume-wake",
            "surface-rigid-attachment",
            "surface-rigid-element-attachment",
            "surface-static-attachment",
            "surface-static-element-attachment",
            "surface-kinematic-attachment",
            "surface-kinematic-element-attachment",
            "surface-articulation-attachment",
            "surface-articulation-element-attachment",
            "surface-partial-element-filter",
            "surface-soft-soft-element-filter",
            "surface-volume-element-filter",
            "volume-volume-element-filter",
            "surface-moving-kinematic-sphere-speculative-ccd",
            "surface-moving-kinematic-capsule-speculative-ccd",
            "surface-rotating-kinematic-capsule-speculative-ccd",
            "surface-moving-kinematic-convex-speculative-ccd",
            "surface-dynamic-sphere-relative-swept-ccd",
            "surface-dynamic-capsule-relative-swept-ccd",
            "surface-dynamic-rotating-capsule-relative-swept-ccd",
            "surface-dynamic-convex-relative-swept-ccd",
            "surface-dynamic-rotating-convex-relative-swept-ccd",
            "surface-sphere-reverse-feature",
            "surface-capsule-reverse-feature",
            "surface-convex-reverse-feature",
            "surface-triangle-mesh-reverse-feature",
            "surface-heightfield-reverse-feature",
        )
        and args.frames < 600
    ):
        parser.error("mixed-contact cases require at least 600 frames")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    executable = args.bin_dir / EXECUTABLE
    if not executable.is_file():
        print(f"[FAIL] executable not found: {executable}")
        return 2

    passed = True
    # Parallel AVBD is a throughput mode.  It is required to satisfy every
    # per-run finite/constraint/contact gate above, but it is not required to
    # replay the scalar GS floating-point trajectory across dispatcher runs.
    # Keep the old exact repeat oracle for the explicitly sequential reference
    # lane only.
    baseline_signature: tuple[str, ...] | None = None
    for repeat_index in range(1, args.repeat + 1):
        repeat_passed, signature = run_once(
            args.bin_dir,
            args.frames,
            args.timeout,
            args.execution,
            args.case,
            repeat_index,
        )
        passed = repeat_passed and passed
        if signature is None or args.execution != "sequential":
            continue
        if baseline_signature is None:
            baseline_signature = signature
        elif signature != baseline_signature:
            print(
                f"[FAIL] {args.case}-r{repeat_index}: "
                "deterministic metrics differ from repeat 1"
            )
            passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
