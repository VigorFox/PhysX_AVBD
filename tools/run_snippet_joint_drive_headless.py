#!/usr/bin/env python3
"""Run SnippetJointDrive headless matrices without shell argument expansion.

Every child is launched with an explicit argv list, ``--headless``, the
``PHYSX_SNIPPET_HEADLESS`` environment sentinel, and (on Windows)
``CREATE_NO_WINDOW``, ``SW_HIDE``, and a kill-on-close Job Object.  The runner
polls for visible child windows, terminates the process tree, and aborts the
matrix if one appears.
A run is accepted only when it emits exactly one machine-readable
``[AVBD_GATE]`` authority line with the expected status, reason, exit code,
frame count, and clean runtime diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

from snippet_headless_process import (
    run_headless_process,
    windows_creation_flags,
    windows_startup_info,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = (
    REPO_ROOT
    / "physx"
    / "bin"
    / "win.x86_64.vc143.md"
    / "checked"
    / "SnippetJointDrive_64.exe"
)
JOINT_OBJECTIVE_IR_PREFIX = "[avbd:joint-objective-ir] "
JOINT_OBJECTIVE_PARTITION_FIELDS = (
    "jointObjectivePositionRows",
    "jointObjectiveFinalizeRows",
    "jointObjectiveUnsupportedRows",
    "jointObjectiveLegacyRows",
    "jointObjectiveInvalidRows",
)
JOINT_OBJECTIVE_EXPECTED_OWNER_BY_CASE = {
    "smoke-angular-position-avbd-parallel-60hz": "PositionAL",
    "position-x-static-identity-avbd-parallel": "PositionAL",
    "position-x-static-identity-avbd-sequential": "PositionAL",
    "angular-slerp-position-avbd-parallel-forward-identity-60hz": "PositionAL",
    "angular-slerp-position-avbd-sequential-forward-identity-60hz":
        "PositionAL",
    "angular-output-twist-forward-on-none-60hz-avbd-parallel":
        "JointFinalize",
    "angular-output-twist-forward-on-none-60hz-avbd-sequential":
        "JointFinalize",
    "angular-output-slerp-forward-on-none-60hz-avbd-parallel":
        "JointFinalize",
    "angular-output-slerp-forward-on-none-60hz-avbd-sequential":
        "JointFinalize",
    "dynamic-mass-scaling-mass-1-forward-avbd-parallel": "JointFinalize",
    "dynamic-mass-scaling-mass-1-forward-avbd-sequential":
        "JointFinalize",
    "contact-position-forward-60hz-avbd-parallel": "PositionAL",
    "contact-position-forward-60hz-avbd-sequential": "PositionAL",
    "dynamic-angular-position-twist-forward-mass1-low-60hz-avbd-parallel":
        "PositionAL",
    "dynamic-angular-position-twist-forward-mass1-low-60hz-avbd-sequential":
        "PositionAL",
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    args: tuple[str, ...]
    expected_status: str = "PASS"
    expected_reason: str = "none"


@dataclass
class RunResult:
    name: str
    command: list[str]
    expected_status: str
    expected_reason: str
    actual_status: str
    actual_reason: str
    exit_code: int | None
    authority_count: int
    timed_out: bool
    visible_window_detected: bool
    visible_window_titles: list[str]
    executable_sha256_before: str
    executable_sha256_after: str
    residual_process: bool
    joint_objective_signature: str
    passed: bool
    errors: list[str]
    log: str


def common_args(solver: str, execution: str) -> list[str]:
    return [f"--solver={solver}", f"--execution={execution}"]


def angular_position_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "90"),
        ("60hz", "0.0166666675", "180"),
        ("120hz", "0.00833333377", "360"),
    )
    lanes = (
        ("pgs", "parallel"),
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for solver, execution in lanes:
        for endpoint in ("forward", "reverse"):
            for initial in ("identity", "driven-pos20"):
                for rate_name, dt, frames in rates:
                    name = (
                        f"angular-position-{solver}-{execution}-{endpoint}-"
                        f"{initial}-{rate_name}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=angular-position",
                        "--drive=twist",
                        f"--endpoint={endpoint}",
                        f"--initial-relative={initial}",
                        "--frame-a=rotz-neg45",
                        "--frame-b=identity",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def angular_swing1_position_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "90"),
        ("60hz", "0.0166666675", "180"),
        ("120hz", "0.00833333377", "360"),
    )
    lanes = (
        ("pgs", "parallel"),
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for solver, execution in lanes:
        for endpoint in ("forward", "reverse"):
            for initial in ("identity", "driven-pos20"):
                for rate_name, dt, frames in rates:
                    name = (
                        f"angular-swing1-position-{solver}-{execution}-{endpoint}-"
                        f"{initial}-{rate_name}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=angular-position",
                        "--drive=swing1",
                        f"--endpoint={endpoint}",
                        f"--initial-relative={initial}",
                        "--frame-a=rotz-neg45",
                        "--frame-b=identity",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def angular_swing2_position_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "90"),
        ("60hz", "0.0166666675", "180"),
        ("120hz", "0.00833333377", "360"),
    )
    lanes = (
        ("pgs", "parallel"),
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for solver, execution in lanes:
        for endpoint in ("forward", "reverse"):
            for initial in ("identity", "driven-pos20"):
                for rate_name, dt, frames in rates:
                    name = (
                        f"angular-swing2-position-{solver}-{execution}-{endpoint}-"
                        f"{initial}-{rate_name}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=angular-position",
                        "--drive=swing2",
                        f"--endpoint={endpoint}",
                        f"--initial-relative={initial}",
                        "--frame-a=rotx-neg45",
                        "--frame-b=identity",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def angular_slerp_position_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "90"),
        ("60hz", "0.0166666675", "180"),
        ("120hz", "0.00833333377", "360"),
    )
    lanes = (
        ("pgs", "parallel"),
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for solver, execution in lanes:
        for endpoint in ("forward", "reverse"):
            for initial in ("identity", "driven-pos20"):
                for rate_name, dt, frames in rates:
                    name = (
                        f"angular-slerp-position-{solver}-{execution}-{endpoint}-"
                        f"{initial}-{rate_name}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=angular-position",
                        "--drive=slerp",
                        f"--endpoint={endpoint}",
                        f"--initial-relative={initial}",
                        "--frame-a=rotx-neg45",
                        "--frame-b=identity",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def dynamic_angular_position_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "90"),
        ("60hz", "0.0166666675", "180"),
        ("120hz", "0.00833333377", "360"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    drives = (
        ("twist", "rotz-neg45"),
        ("swing1", "rotz-neg45"),
        ("swing2", "rotx-neg45"),
        ("slerp", "rotx-neg45"),
    )
    specs: list[RunSpec] = []
    for drive, frame_a in drives:
        for endpoint in ("forward", "reverse"):
            for mass in ("1", "10"):
                for limit in ("low", "high"):
                    for rate_name, dt, frames in rates:
                        for solver, execution in lanes:
                            name = (
                                f"dynamic-angular-position-{drive}-{endpoint}-"
                                f"mass{mass}-{limit}-{rate_name}-{solver}-"
                                f"{execution}"
                            )
                            args = common_args(solver, execution) + [
                                "--case=angular-position",
                                f"--drive={drive}",
                                f"--endpoint={endpoint}",
                                "--topology=dynamic-dynamic",
                                f"--mass={mass}",
                                f"--limit={limit}",
                                "--initial-relative=identity",
                                f"--frame-a={frame_a}",
                                "--frame-b=identity",
                                f"--dt={dt}",
                                f"--frames={frames}",
                            ]
                            specs.append(RunSpec(name, tuple(args)))
    return specs


def dynamic_angular_position12_specs() -> list[RunSpec]:
    return [
        spec
        for spec in dynamic_angular_position_specs()
        if "-twist-" in spec.name
        and "-low-" in spec.name
        and "-60hz-" in spec.name
    ]


def dynamic_angular_slerp72_specs() -> list[RunSpec]:
    return [
        spec
        for spec in dynamic_angular_position_specs()
        if "-slerp-" in spec.name
    ]


def position42_specs() -> list[RunSpec]:
    configs = (
        ("x", "static", "identity", "identity"),
        ("twist", "static", "identity", "identity"),
        ("swing1", "static", "identity", "identity"),
        ("swing2", "static", "identity", "identity"),
        ("slerp", "static", "identity", "identity"),
        ("x", "kinematic", "rotz-neg45", "rotz-neg45"),
        ("slerp", "kinematic", "rotz-neg45", "rotz-neg45"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for drive, actor_a, frame_a, frame_b in configs:
        for initial in ("identity", "driven-pos20"):
            for solver, execution in lanes:
                supported = (
                    solver == "tgs"
                    or drive == "x"
                    or drive == "twist"
                    or drive == "swing1"
                    or drive == "swing2"
                    or drive == "slerp"
                )
                status = "PASS" if supported else "FAIL"
                reason = "none" if supported else "position_target_not_tracked"
                name = (
                    f"position-{drive}-{actor_a}-{initial}-{solver}-{execution}"
                )
                args = common_args(solver, execution) + [
                    "--case=position",
                    f"--drive={drive}",
                    "--drive-mode=force",
                    f"--actor-a={actor_a}",
                    f"--frame-a={frame_a}",
                    f"--frame-b={frame_b}",
                    f"--initial-relative={initial}",
                    "--frames=180",
                ]
                specs.append(RunSpec(name, tuple(args), status, reason))
    return specs


def repeated_lanes() -> tuple[tuple[str, str, int], ...]:
    return (
        ("avbd", "parallel", 1),
        ("avbd", "parallel", 2),
        ("avbd", "parallel", 3),
        ("avbd", "parallel", 4),
        ("avbd", "parallel", 5),
        ("avbd", "sequential", 1),
        ("tgs", "parallel", 1),
    )


def velocity154_specs() -> list[RunSpec]:
    configs = (
        ("x", "static", "identity", "identity", "identity"),
        ("y", "static", "identity", "identity", "identity"),
        ("z", "static", "identity", "identity", "identity"),
        ("twist", "static", "identity", "identity", "identity"),
        ("swing1", "static", "identity", "identity", "identity"),
        ("swing2", "static", "identity", "identity", "identity"),
        ("slerp", "static", "identity", "identity", "identity"),
        ("x", "kinematic", "rotz-neg45", "identity", "identity"),
        ("twist", "static", "rotz-neg45", "rotz-neg45", "identity"),
        ("swing1", "static", "rotz-neg45", "identity", "rotz-neg45"),
        ("swing2", "static", "rotz-neg45", "rotz-neg45", "rotz-neg45"),
        ("slerp", "kinematic", "rotz-neg45", "identity", "rotz-neg45"),
    )
    specs: list[RunSpec] = []
    for index, config in enumerate(configs, start=1):
        drive, actor_a, frame_a, frame_b, body_b = config
        for solver, execution, seed in repeated_lanes():
            name = f"velocity-{index:02d}-{solver}-{execution}-seed{seed}"
            args = common_args(solver, execution) + [
                "--case=velocity",
                f"--drive={drive}",
                f"--actor-a={actor_a}",
                f"--frame-a={frame_a}",
                f"--frame-b={frame_b}",
                f"--body-b-rotation={body_b}",
                "--frames=180",
                f"--seed={seed}",
            ]
            specs.append(RunSpec(name, tuple(args)))

    for endpoint in ("forward", "reverse"):
        for solver, execution, seed in repeated_lanes():
            name = f"velocity-ordering-{endpoint}-{solver}-{execution}-seed{seed}"
            args = common_args(solver, execution) + [
                "--case=velocity-ordering",
                "--drive=x",
                f"--endpoint={endpoint}",
                "--frames=180",
                f"--seed={seed}",
            ]
            specs.append(RunSpec(name, tuple(args)))

    for drive in ("twist", "swing1", "swing2", "slerp"):
        for endpoint in ("forward", "reverse"):
            for solver, execution, seed in repeated_lanes():
                name = (
                    f"angular-ordering-{drive}-{endpoint}-{solver}-"
                    f"{execution}-seed{seed}"
                )
                args = common_args(solver, execution) + [
                    "--case=angular-ordering",
                    f"--drive={drive}",
                    f"--endpoint={endpoint}",
                    "--frames=180",
                    f"--seed={seed}",
                ]
                specs.append(RunSpec(name, tuple(args)))
    return specs


def comparison30_specs() -> list[RunSpec]:
    selectors = (
        ("mass-scaling", "--drive-mode=force", "--mass=1", False),
        ("acceleration-mode", "--drive-mode=acceleration", "--mass=1", False),
        ("force-limit", "--drive-mode=force", "--limit=high", False),
        ("mass-scaling", "--drive-mode=force", "--mass=10", True),
        ("acceleration-mode", "--drive-mode=acceleration", "--mass=10", True),
        ("force-limit", "--drive-mode=force", "--limit=low", True),
    )
    specs: list[RunSpec] = []
    for case, mode, selector, capability in selectors:
        lanes = (
            repeated_lanes()
            if capability
            else (
                ("tgs", "parallel", 1),
                ("avbd", "parallel", 1),
                ("avbd", "sequential", 1),
            )
        )
        for solver, execution, seed in lanes:
            name = (
                f"comparison-{case}-{selector.removeprefix('--').replace('=', '-')}-"
                f"{solver}-{execution}-seed{seed}"
            )
            args = common_args(solver, execution) + [
                f"--case={case}",
                "--drive=x",
                mode,
                selector,
                "--frames=180",
                f"--seed={seed}",
            ]
            specs.append(RunSpec(name, tuple(args)))
    return specs


def soak2_specs() -> list[RunSpec]:
    return [
        RunSpec(
            "soak-velocity-x-static-identity",
            tuple(
                common_args("avbd", "parallel")
                + [
                    "--case=velocity",
                    "--drive=x",
                    "--actor-a=static",
                    "--frame-a=identity",
                    "--frame-b=identity",
                    "--body-b-rotation=identity",
                    "--frames=10000",
                ]
            ),
        ),
        RunSpec(
            "soak-velocity-slerp-kinematic-rotated",
            tuple(
                common_args("avbd", "parallel")
                + [
                    "--case=velocity",
                    "--drive=slerp",
                    "--actor-a=kinematic",
                    "--frame-a=rotz-neg45",
                    "--frame-b=identity",
                    "--body-b-rotation=rotz-neg45",
                    "--frames=10000",
                ]
            ),
        ),
    ]


def legacy228_specs() -> list[RunSpec]:
    return velocity154_specs() + position42_specs() + comparison30_specs() + soak2_specs()


def position72_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "90"),
        ("60hz", "0.0166666675", "180"),
        ("120hz", "0.00833333377", "360"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for endpoint in ("forward", "reverse"):
        for mass in ("1", "10"):
            for limit in ("high", "low"):
                for rate_name, dt, frames in rates:
                    for solver, execution in lanes:
                        name = (
                            f"position-direct-{endpoint}-mass{mass}-{limit}-"
                            f"{rate_name}-{solver}-{execution}"
                        )
                        args = common_args(solver, execution) + [
                            "--case=position",
                            "--drive=x",
                            "--drive-mode=force",
                            f"--endpoint={endpoint}",
                            f"--mass={mass}",
                            f"--limit={limit}",
                            f"--dt={dt}",
                            f"--frames={frames}",
                        ]
                        specs.append(RunSpec(name, tuple(args)))
    return specs


def dynamic36_specs() -> list[RunSpec]:
    selectors = (
        ("mass-scaling", "--drive-mode=force", "--mass=1", False),
        ("mass-scaling", "--drive-mode=force", "--mass=10", False),
        ("acceleration-mode", "--drive-mode=acceleration", "--mass=1", False),
        ("acceleration-mode", "--drive-mode=acceleration", "--mass=10", False),
        ("force-limit", "--drive-mode=force", "--limit=high", False),
        ("force-limit", "--drive-mode=force", "--limit=low", False),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for case, mode, selector, avbd_expected_failure in selectors:
        for endpoint in ("forward", "reverse"):
            for solver, execution in lanes:
                expected_failure = avbd_expected_failure and solver == "avbd"
                status = "FAIL" if expected_failure else "PASS"
                reason = (
                    "acceleration_drive_not_distinct"
                    if expected_failure
                    else "none"
                )
                name = (
                    f"dynamic-{case}-{selector.removeprefix('--').replace('=', '-')}-"
                    f"{endpoint}-{solver}-{execution}"
                )
                args = common_args(solver, execution) + [
                    f"--case={case}",
                    "--drive=x",
                    mode,
                    selector,
                    "--topology=dynamic-dynamic",
                    f"--endpoint={endpoint}",
                    "--frames=180",
                ]
                specs.append(RunSpec(name, tuple(args), status, reason))
    return specs


def contact72_specs() -> list[RunSpec]:
    selectors = (
        ("mass-scaling", "--mass=1"),
        ("mass-scaling", "--mass=10"),
        ("force-limit", "--limit=high"),
        ("force-limit", "--limit=low"),
    )
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for case, selector in selectors:
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        f"contact-{case}-{selector.removeprefix('--').replace('=', '-')}-"
                        f"{endpoint}-{rate_name}-{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        f"--case={case}",
                        "--drive=x",
                        "--drive-mode=force",
                        selector,
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_acceleration36_specs() -> list[RunSpec]:
    selectors = (
        ("mass1", "--mass=1"),
        ("mass10", "--mass=10"),
    )
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for selector_name, selector in selectors:
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        f"contact-acceleration-{selector_name}-{endpoint}-"
                        f"{rate_name}-{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=acceleration-mode",
                        "--drive=x",
                        "--drive-mode=acceleration",
                        selector,
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_acceleration_limit18_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for endpoint in ("forward", "reverse"):
        for rate_name, dt, frames in rates:
            for solver, execution in lanes:
                name = (
                    f"contact-acceleration-limit-low-{endpoint}-"
                    f"{rate_name}-{solver}-{execution}"
                )
                args = common_args(solver, execution) + [
                    "--case=acceleration-mode",
                    "--drive=x",
                    "--drive-mode=acceleration",
                    "--mass=1",
                    "--limit=low",
                    "--topology=contact-dynamic-dynamic",
                    f"--endpoint={endpoint}",
                    f"--dt={dt}",
                    f"--frames={frames}",
                ]
                specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position18_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for endpoint in ("forward", "reverse"):
        for rate_name, dt, frames in rates:
            for solver, execution in lanes:
                name = (
                    f"contact-position-{endpoint}-{rate_name}-"
                    f"{solver}-{execution}"
                )
                args = common_args(solver, execution) + [
                    "--case=position",
                    "--drive=x",
                    "--drive-mode=force",
                    "--mass=1",
                    "--limit=high",
                    "--topology=contact-dynamic-dynamic",
                    f"--endpoint={endpoint}",
                    f"--dt={dt}",
                    f"--frames={frames}",
                ]
                specs.append(
                    RunSpec(
                        name,
                        tuple(args),
                        "PASS",
                        "none",
                    )
                )
    return specs


def contact_position_limit18_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for endpoint in ("forward", "reverse"):
        for rate_name, dt, frames in rates:
            for solver, execution in lanes:
                name = (
                    f"contact-position-limit-low-{endpoint}-{rate_name}-"
                    f"{solver}-{execution}"
                )
                args = common_args(solver, execution) + [
                    "--case=position",
                    "--drive=x",
                    "--drive-mode=force",
                    "--mass=1",
                    "--limit=low",
                    "--topology=contact-dynamic-dynamic",
                    f"--endpoint={endpoint}",
                    f"--dt={dt}",
                    f"--frames={frames}",
                ]
                specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position_unequal36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for limit in ("high", "low"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        f"contact-position-unequal-{limit}-{endpoint}-"
                        f"{rate_name}-{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=10",
                        f"--limit={limit}",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position_friction36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for friction in ("zero", "standard"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        f"contact-position-friction-{friction}-{endpoint}-"
                        f"{rate_name}-{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        "--limit=high",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--friction={friction}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position_support_slope36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for friction in ("zero", "standard"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        "contact-position-support-slope-"
                        f"{friction}-{endpoint}-{rate_name}-"
                        f"{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        "--limit=high",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--friction={friction}",
                        "--support=slope-x10",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position_output_force36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for output in ("off", "on"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        "contact-position-output-force-"
                        f"{output}-{endpoint}-{rate_name}-"
                        f"{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=output-force",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        "--limit=low",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--output-force={output}",
                        "--break=none",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position_friction_limit36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for limit in ("high", "low"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        f"contact-position-friction-limit-{limit}-"
                        f"{endpoint}-{rate_name}-{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        f"--limit={limit}",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        "--friction=standard",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position_friction_unequal36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for limit in ("high", "low"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        f"contact-position-friction-unequal-{limit}-"
                        f"{endpoint}-{rate_name}-{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=10",
                        f"--limit={limit}",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        "--friction=standard",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position_target_velocity36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for target_velocity in ("zero", "positive"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        "contact-position-target-velocity-"
                        f"{target_velocity}-{endpoint}-{rate_name}-"
                        f"{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        "--limit=high",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--position-target-velocity={target_velocity}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position_anchor36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for anchor in ("centered", "symmetric-y25"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        f"contact-position-anchor-{anchor}-{endpoint}-"
                        f"{rate_name}-{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        "--limit=high",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--anchor={anchor}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position_anchor_z36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for anchor in ("centered", "symmetric-z25"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        f"contact-position-anchor-z-{anchor}-{endpoint}-"
                        f"{rate_name}-{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        "--limit=high",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--anchor={anchor}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position_anchor_x36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for anchor in ("centered", "symmetric-x25"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        f"contact-position-anchor-x-{anchor}-{endpoint}-"
                        f"{rate_name}-{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        "--limit=high",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--anchor={anchor}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def contact_position_anchor_asymmetric_z36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for anchor in ("centered", "asymmetric-z25"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        "contact-position-anchor-asymmetric-z-"
                        f"{anchor}-{endpoint}-{rate_name}-"
                        f"{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        "--limit=high",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--anchor={anchor}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(
                        RunSpec(
                            name,
                            tuple(args),
                            "PASS",
                            "none",
                        )
                    )
    return specs


def contact_position_anchor_asymmetric_x36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for anchor in ("centered", "asymmetric-x25"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        "contact-position-anchor-asymmetric-x-"
                        f"{anchor}-{endpoint}-{rate_name}-"
                        f"{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        "--limit=high",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--anchor={anchor}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(
                        RunSpec(
                            name,
                            tuple(args),
                            "PASS",
                            "none",
                        )
                    )
    return specs


def contact_position_anchor_two_sided_z36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for anchor in ("centered", "asymmetric-zpair25"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        "contact-position-anchor-two-sided-z-"
                        f"{anchor}-{endpoint}-{rate_name}-"
                        f"{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        "--limit=high",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--anchor={anchor}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(
                        RunSpec(
                            name,
                            tuple(args),
                            "PASS",
                            "none",
                        )
                    )
    return specs


def contact_position_frame36_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for frame in ("identity", "roty-neg45"):
        for endpoint in ("forward", "reverse"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        f"contact-position-frame-{frame}-{endpoint}-"
                        f"{rate_name}-{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=position",
                        "--drive=x",
                        "--drive-mode=force",
                        "--mass=1",
                        "--limit=high",
                        "--topology=contact-dynamic-dynamic",
                        f"--endpoint={endpoint}",
                        f"--frame-a={frame}",
                        f"--frame-b={frame}",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def dynamic_acceleration_limit18_specs() -> list[RunSpec]:
    rates = (
        ("30hz", "0.0333333351", "300"),
        ("60hz", "0.0166666675", "600"),
        ("120hz", "0.00833333377", "1200"),
    )
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    specs: list[RunSpec] = []
    for endpoint in ("forward", "reverse"):
        for rate_name, dt, frames in rates:
            for solver, execution in lanes:
                name = (
                    f"dynamic-acceleration-limit-low-{endpoint}-"
                    f"{rate_name}-{solver}-{execution}"
                )
                args = common_args(solver, execution) + [
                    "--case=acceleration-mode",
                    "--drive=x",
                    "--drive-mode=acceleration",
                    "--mass=1",
                    "--limit=low",
                    "--topology=dynamic-dynamic",
                    f"--endpoint={endpoint}",
                    f"--dt={dt}",
                    f"--frames={frames}",
                ]
                specs.append(RunSpec(name, tuple(args)))
    return specs


def linear_output72_specs() -> list[RunSpec]:
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    rates = (
        ("30hz", "0.0333333351", "90"),
        ("60hz", "0.0166666675", "180"),
        ("120hz", "0.00833333377", "360"),
    )
    specs: list[RunSpec] = []
    for endpoint in ("forward", "reverse"):
        for output in ("on", "off"):
            for rate_name, dt, frames in rates:
                for solver, execution in lanes:
                    name = (
                        f"linear-output-nobreak-{endpoint}-{output}-{rate_name}-"
                        f"{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=output-force",
                        "--drive=x",
                        f"--endpoint={endpoint}",
                        f"--output-force={output}",
                        "--break=none",
                        f"--dt={dt}",
                        f"--frames={frames}",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    for endpoint in ("forward", "reverse"):
        for output in ("on", "off"):
            for break_mode in ("none", "below", "above"):
                for solver, execution in lanes:
                    name = (
                        f"linear-output-breakmatrix-{endpoint}-{output}-"
                        f"{break_mode}-60hz-"
                        f"{solver}-{execution}"
                    )
                    args = common_args(solver, execution) + [
                        "--case=output-force",
                        "--drive=x",
                        f"--endpoint={endpoint}",
                        f"--output-force={output}",
                        f"--break={break_mode}",
                        "--dt=0.0166666675",
                        "--frames=180",
                    ]
                    specs.append(RunSpec(name, tuple(args)))
    return specs


def angular_output432_specs() -> list[RunSpec]:
    lanes = (
        ("tgs", "parallel"),
        ("avbd", "parallel"),
        ("avbd", "sequential"),
    )
    rates = (
        ("30hz", "0.0333333351", "90"),
        ("60hz", "0.0166666675", "180"),
        ("120hz", "0.00833333377", "360"),
    )
    specs: list[RunSpec] = []
    for drive in ("twist", "swing1", "swing2", "slerp"):
        for endpoint in ("forward", "reverse"):
            for output in ("on", "off"):
                for break_mode in ("none", "below", "above"):
                    for rate_name, dt, frames in rates:
                        for solver, execution in lanes:
                            name = (
                                f"angular-output-{drive}-{endpoint}-{output}-"
                                f"{break_mode}-{rate_name}-{solver}-{execution}"
                            )
                            args = common_args(solver, execution) + [
                                "--case=angular-output-force",
                                f"--drive={drive}",
                                f"--endpoint={endpoint}",
                                f"--output-force={output}",
                                f"--break={break_mode}",
                                f"--dt={dt}",
                                f"--frames={frames}",
                            ]
                            specs.append(RunSpec(name, tuple(args)))
    return specs


SUITES = {
    "smoke": lambda: [
        RunSpec(
            "smoke-angular-position-avbd-parallel-60hz",
            tuple(
                common_args("avbd", "parallel")
                + [
                    "--case=angular-position",
                    "--drive=twist",
                    "--endpoint=forward",
                    "--initial-relative=identity",
                    "--frame-a=rotz-neg45",
                    "--frame-b=identity",
                    "--dt=0.0166666675",
                    "--frames=180",
                ]
            ),
        )
    ],
    "angular-position48": angular_position_specs,
    "angular-swing1-position48": angular_swing1_position_specs,
    "angular-swing2-position48": angular_swing2_position_specs,
    "angular-slerp-position48": angular_slerp_position_specs,
    "dynamic-angular-position12": dynamic_angular_position12_specs,
    "dynamic-angular-position288": dynamic_angular_position_specs,
    "dynamic-angular-slerp72": dynamic_angular_slerp72_specs,
    "angular-output432": angular_output432_specs,
    "contact72": contact72_specs,
    "contact-acceleration36": contact_acceleration36_specs,
    "contact-acceleration-limit18": contact_acceleration_limit18_specs,
    "contact-position18": contact_position18_specs,
    "contact-position-limit18": contact_position_limit18_specs,
    "contact-position-unequal36": contact_position_unequal36_specs,
    "contact-position-friction36": contact_position_friction36_specs,
    "contact-position-support-slope36":
        contact_position_support_slope36_specs,
    "contact-position-support-slope-smoke": lambda: [
        spec
        for spec in contact_position_support_slope36_specs()
        if spec.name
        in {
            (
                "contact-position-support-slope-zero-forward-"
                "60hz-avbd-parallel"
            ),
            (
                "contact-position-support-slope-standard-forward-"
                "60hz-tgs-parallel"
            ),
            (
                "contact-position-support-slope-standard-forward-"
                "60hz-avbd-parallel"
            ),
            (
                "contact-position-support-slope-standard-forward-"
                "60hz-avbd-sequential"
            ),
        }
    ],
    "contact-position-output-force36":
        contact_position_output_force36_specs,
    "contact-position-output-force-smoke": lambda: [
        spec
        for spec in contact_position_output_force36_specs()
        if spec.name
        in {
            (
                "contact-position-output-force-"
                "off-forward-60hz-avbd-parallel"
            ),
            (
                "contact-position-output-force-"
                "on-forward-60hz-tgs-parallel"
            ),
            (
                "contact-position-output-force-"
                "on-forward-60hz-avbd-parallel"
            ),
            (
                "contact-position-output-force-"
                "on-forward-60hz-avbd-sequential"
            ),
        }
    ],
    "contact-position-friction-limit36":
        contact_position_friction_limit36_specs,
    "contact-position-friction-limit-smoke": lambda: [
        spec
        for spec in contact_position_friction_limit36_specs()
        if spec.name
        in {
            (
                "contact-position-friction-limit-low-forward-"
                "60hz-tgs-parallel"
            ),
            (
                "contact-position-friction-limit-low-forward-"
                "60hz-avbd-parallel"
            ),
        }
    ],
    "contact-position-friction-unequal36":
        contact_position_friction_unequal36_specs,
    "contact-position-friction-unequal-smoke": lambda: [
        spec
        for spec in contact_position_friction_unequal36_specs()
        if spec.name
        in {
            (
                "contact-position-friction-unequal-high-forward-"
                "60hz-tgs-parallel"
            ),
            (
                "contact-position-friction-unequal-high-forward-"
                "60hz-avbd-parallel"
            ),
            (
                "contact-position-friction-unequal-low-forward-"
                "60hz-tgs-parallel"
            ),
            (
                "contact-position-friction-unequal-low-forward-"
                "60hz-avbd-parallel"
            ),
        }
    ],
    "contact-position-friction-smoke": lambda: [
        spec
        for spec in contact_position_friction36_specs()
        if spec.name
        == "contact-position-friction-standard-forward-60hz-avbd-parallel"
    ],
    "contact-position-target-velocity36":
        contact_position_target_velocity36_specs,
    "contact-position-target-velocity-smoke": lambda: [
        spec
        for spec in contact_position_target_velocity36_specs()
        if spec.name
        == (
            "contact-position-target-velocity-positive-forward-"
            "60hz-avbd-parallel"
        )
    ],
    "contact-position-anchor36": contact_position_anchor36_specs,
    "contact-position-anchor-smoke": lambda: [
        spec
        for spec in contact_position_anchor36_specs()
        if spec.name
        == (
            "contact-position-anchor-symmetric-y25-forward-"
            "60hz-avbd-parallel"
        )
    ],
    "contact-position-anchor-z36": contact_position_anchor_z36_specs,
    "contact-position-anchor-z-smoke": lambda: [
        spec
        for spec in contact_position_anchor_z36_specs()
        if spec.name
        == (
            "contact-position-anchor-z-symmetric-z25-forward-"
            "60hz-avbd-parallel"
        )
    ],
    "contact-position-anchor-x36": contact_position_anchor_x36_specs,
    "contact-position-anchor-x-smoke": lambda: [
        spec
        for spec in contact_position_anchor_x36_specs()
        if spec.name
        == (
            "contact-position-anchor-x-symmetric-x25-forward-"
            "60hz-avbd-parallel"
        )
    ],
    "contact-position-anchor-asymmetric-z36":
        contact_position_anchor_asymmetric_z36_specs,
    "contact-position-anchor-asymmetric-z-smoke": lambda: [
        spec
        for spec in contact_position_anchor_asymmetric_z36_specs()
        if spec.name
        in {
            (
                "contact-position-anchor-asymmetric-z-"
                "asymmetric-z25-forward-60hz-tgs-parallel"
            ),
            (
                "contact-position-anchor-asymmetric-z-"
                "asymmetric-z25-forward-60hz-avbd-parallel"
            ),
        }
    ],
    "contact-position-anchor-asymmetric-x36":
        contact_position_anchor_asymmetric_x36_specs,
    "contact-position-anchor-asymmetric-x-smoke": lambda: [
        spec
        for spec in contact_position_anchor_asymmetric_x36_specs()
        if spec.name
        in {
            (
                "contact-position-anchor-asymmetric-x-"
                "asymmetric-x25-forward-60hz-tgs-parallel"
            ),
            (
                "contact-position-anchor-asymmetric-x-"
                "asymmetric-x25-forward-60hz-avbd-parallel"
            ),
        }
    ],
    "contact-position-anchor-two-sided-z36":
        contact_position_anchor_two_sided_z36_specs,
    "contact-position-anchor-two-sided-z-smoke": lambda: [
        spec
        for spec in contact_position_anchor_two_sided_z36_specs()
        if spec.name
        in {
            (
                "contact-position-anchor-two-sided-z-"
                "asymmetric-zpair25-forward-60hz-tgs-parallel"
            ),
            (
                "contact-position-anchor-two-sided-z-"
                "asymmetric-zpair25-forward-60hz-avbd-parallel"
            ),
        }
    ],
    "contact-position-frame36": contact_position_frame36_specs,
    "contact-position-frame-smoke": lambda: [
        spec
        for spec in contact_position_frame36_specs()
        if spec.name
        == (
            "contact-position-frame-roty-neg45-forward-"
            "60hz-avbd-parallel"
        )
    ],
    "dynamic36": dynamic36_specs,
    "dynamic-acceleration-limit18": dynamic_acceleration_limit18_specs,
    "legacy228": legacy228_specs,
    "linear-output72": linear_output72_specs,
    "position42": position42_specs,
    "position72": position72_specs,
    "velocity154": velocity154_specs,
}


def parse_authority(line: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    errors: list[str] = []
    for token in line.split()[1:]:
        if "=" not in token:
            errors.append(f"malformed authority token: {token}")
            continue
        key, value = token.split("=", 1)
        if key in fields:
            errors.append(f"duplicate authority key: {key}")
        fields[key] = value
    return fields, errors


def requested_frames(args: Iterable[str]) -> str | None:
    values = [arg.split("=", 1)[1] for arg in args if arg.startswith("--frames=")]
    return values[0] if len(values) == 1 else None


def validate_joint_objective_ir(
    stdout: str,
    expected_owner: str | None,
) -> tuple[list[str], str]:
    errors: list[str] = []
    lines = [
        line.strip()
        for line in stdout.splitlines()
        if line.startswith(JOINT_OBJECTIVE_IR_PREFIX)
    ]
    if not lines:
        return ["no [avbd:joint-objective-ir] diagnostic samples"], ""

    totals = {field: 0 for field in JOINT_OBJECTIVE_PARTITION_FIELDS}
    signatures: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        fields, parse_errors = parse_authority(line)
        errors.extend(
            f"joint objective diagnostic {line_number}: {error}"
            for error in parse_errors
        )
        required = (
            "jointObjectiveRows",
            *JOINT_OBJECTIVE_PARTITION_FIELDS,
            "jointObjectiveFingerprint",
        )
        missing = [field for field in required if field not in fields]
        if missing:
            errors.append(
                f"joint objective diagnostic {line_number} is missing "
                + ", ".join(missing)
            )
            continue
        try:
            values = {field: int(fields[field]) for field in required}
        except ValueError:
            errors.append(
                f"joint objective diagnostic {line_number} has "
                "a non-integer field"
            )
            continue
        if any(value < 0 for value in values.values()):
            errors.append(
                f"joint objective diagnostic {line_number} has "
                "a negative field"
            )
            continue
        partition_rows = sum(
            values[field] for field in JOINT_OBJECTIVE_PARTITION_FIELDS
        )
        if values["jointObjectiveRows"] != partition_rows:
            errors.append(
                f"joint objective diagnostic {line_number} has "
                f"jointObjectiveRows={values['jointObjectiveRows']} "
                f"but partition={partition_rows}"
            )
        if values["jointObjectiveInvalidRows"] != 0:
            errors.append(
                f"joint objective diagnostic {line_number} has "
                "jointObjectiveInvalidRows="
                f"{values['jointObjectiveInvalidRows']}"
            )
        for field in JOINT_OBJECTIVE_PARTITION_FIELDS:
            totals[field] += values[field]
        signatures.append(
            ":".join(str(values[field]) for field in required)
        )

    if expected_owner is not None:
        owner_field = {
            "PositionAL": "jointObjectivePositionRows",
            "JointFinalize": "jointObjectiveFinalizeRows",
        }[expected_owner]
        if totals[owner_field] == 0:
            errors.append(
                f"focused lane compiled no {expected_owner} owner"
            )
        if totals["jointObjectiveUnsupportedRows"] != 0:
            errors.append(
                "focused owner lane fell back to Unsupported"
            )
    return errors, "|".join(signatures)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def process_is_running(image_name: str, creationflags: int) -> bool:
    if os.name != "nt":
        return False
    check = subprocess.run(
        ["tasklist.exe", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=creationflags,
        startupinfo=windows_startup_info(),
        shell=False,
    )
    return image_name.casefold() in check.stdout.casefold()


def run_one(
    executable: Path,
    output_root: Path,
    spec: RunSpec,
    timeout_seconds: float,
) -> RunResult:
    argv = [str(executable), "--headless", *spec.args]
    log_path = output_root / f"{spec.name}.log"
    environment = os.environ.copy()
    environment["PHYSX_SNIPPET_HEADLESS"] = "1"
    environment["PHYSX_AVBD_ITER_DIAG"] = "1"
    environment["PHYSX_AVBD_ITER_DIAG_EVERY"] = "60"
    creationflags = windows_creation_flags()
    before_hash = sha256(executable)
    timed_out = False
    exit_code: int | None = None
    stdout = ""
    stderr = ""
    completed = run_headless_process(
        argv,
        cwd=executable.parent,
        env=environment,
        timeout_seconds=timeout_seconds,
    )
    exit_code = completed.returncode
    stdout = completed.stdout
    stderr = completed.stderr
    timed_out = completed.timed_out

    authority_lines = [
        line.strip()
        for line in stdout.splitlines()
        if line.startswith("[AVBD_GATE] ")
    ]
    fields: dict[str, str] = {}
    errors: list[str] = []
    if timed_out:
        errors.append(f"timeout after {timeout_seconds:g} seconds")
    if completed.visible_window_detected:
        errors.append(
            "visible child window detected; process tree terminated: "
            + ", ".join(completed.visible_window_titles)
        )
    if len(authority_lines) != 1:
        errors.append(f"authority count is {len(authority_lines)}, expected 1")
    else:
        fields, parse_errors = parse_authority(authority_lines[0])
        errors.extend(parse_errors)

    actual_status = fields.get("status", "MISSING")
    actual_reason = fields.get("reason", "missing")
    expected_exit = 0 if spec.expected_status == "PASS" else 1
    if exit_code != expected_exit:
        errors.append(f"exit code {exit_code}, expected {expected_exit}")
    if actual_status != spec.expected_status:
        errors.append(
            f"status {actual_status}, expected {spec.expected_status}"
        )
    if actual_reason != spec.expected_reason:
        errors.append(
            f"reason {actual_reason}, expected {spec.expected_reason}"
        )
    if fields.get("snippet") != "SnippetJointDrive":
        errors.append(f"snippet {fields.get('snippet', 'MISSING')}")
    if fields.get("case") == "config-error":
        errors.append("headless invocation was rejected as config-error")
    frames = requested_frames(spec.args)
    if frames is None:
        errors.append("run spec must contain exactly one --frames selector")
    else:
        if fields.get("requestedFrames") != frames:
            errors.append(
                f"requestedFrames {fields.get('requestedFrames', 'MISSING')}, "
                f"expected {frames}"
            )
        if fields.get("completedFrames") != frames:
            errors.append(
                f"completedFrames {fields.get('completedFrames', 'MISSING')}, "
                f"expected {frames}"
            )
    for key in (
        "nonFinite",
        "physicsErrors",
        "physicsWarnings",
        "fetchFailures",
        "fetchErrorState",
    ):
        if fields.get(key) != "0":
            errors.append(f"{key}={fields.get(key, 'MISSING')}, expected 0")
    if stderr:
        errors.append(f"stderr is not empty ({len(stderr.encode('utf-8'))} bytes)")
    is_avbd = "--solver=avbd" in spec.args
    joint_objective_signature = ""
    if is_avbd and spec.expected_status == "PASS":
        joint_objective_errors, joint_objective_signature = (
            validate_joint_objective_ir(
                stdout,
                expected_owner=JOINT_OBJECTIVE_EXPECTED_OWNER_BY_CASE.get(
                    spec.name
                ),
            )
        )
        errors.extend(joint_objective_errors)

    after_hash = sha256(executable)
    if after_hash != before_hash:
        errors.append("executable SHA-256 changed during the run")
    residual_process = process_is_running(executable.name, creationflags)
    if residual_process:
        errors.append(f"residual process detected: {executable.name}")

    command_text = subprocess.list2cmdline(argv)
    log_text = (
        f"COMMAND: {command_text}\n"
        f"HEADLESS_ENV: PHYSX_SNIPPET_HEADLESS=1\n"
        "JOINT_OBJECTIVE_IR_ENV: PHYSX_AVBD_ITER_DIAG=1 "
        "PHYSX_AVBD_ITER_DIAG_EVERY=60\n"
        f"CREATE_NO_WINDOW: {int(os.name == 'nt')}\n"
        f"STARTUPINFO_SW_HIDE: {int(os.name == 'nt')}\n"
        f"KILL_ON_JOB_CLOSE: {int(os.name == 'nt')}\n"
        f"VISIBLE_WINDOW_DETECTED: {int(completed.visible_window_detected)}\n"
        "VISIBLE_WINDOW_TITLES: "
        + (", ".join(completed.visible_window_titles) or "none")
        + "\n"
        f"EXECUTABLE_SHA256_BEFORE: {before_hash}\n"
        f"EXECUTABLE_SHA256_AFTER: {after_hash}\n"
        f"EXPECTED: status={spec.expected_status} reason={spec.expected_reason}\n"
        f"EXIT_CODE: {exit_code}\n"
        f"TIMED_OUT: {int(timed_out)}\n"
        f"RESIDUAL_PROCESS: {int(residual_process)}\n"
        "--- STDOUT ---\n"
        f"{stdout}"
        "\n--- STDERR ---\n"
        f"{stderr}"
        "\n--- RUNNER ERRORS ---\n"
        + ("\n".join(errors) if errors else "none")
        + "\n"
    )
    log_path.write_text(log_text, encoding="utf-8")
    return RunResult(
        name=spec.name,
        command=argv,
        expected_status=spec.expected_status,
        expected_reason=spec.expected_reason,
        actual_status=actual_status,
        actual_reason=actual_reason,
        exit_code=exit_code,
        authority_count=len(authority_lines),
        timed_out=timed_out,
        visible_window_detected=completed.visible_window_detected,
        visible_window_titles=list(completed.visible_window_titles),
        executable_sha256_before=before_hash,
        executable_sha256_after=after_hash,
        residual_process=residual_process,
        joint_objective_signature=joint_objective_signature,
        passed=not errors,
        errors=errors,
        log=str(log_path),
    )


def make_output_root(requested: Path | None) -> Path:
    if requested is not None:
        root = requested.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        root = Path(tempfile.gettempdir()) / f"PhysX_AVBD_jointdrive_headless_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        choices=sorted(SUITES),
        help="Named matrix; may be repeated (default: smoke).",
    )
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--list-suites", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        dest="selected_cases",
        help="Run only the named case; searches selected suites or all suites.",
    )
    parser.add_argument("--list-cases", action="store_true")
    options = parser.parse_args()

    if options.list_suites:
        for name in sorted(SUITES):
            print(name)
        return 0
    if options.list_cases:
        cases = {
            spec.name
            for factory in SUITES.values()
            for spec in factory()
        }
        for name in sorted(cases):
            print(name)
        return 0
    if options.timeout <= 0:
        parser.error("--timeout must be positive")

    executable = options.exe.resolve()
    if not executable.is_file():
        parser.error(f"executable does not exist: {executable}")
    if options.selected_cases:
        suites = options.suite or list(SUITES)
        candidates = {
            spec.name: spec
            for suite in suites
            for spec in SUITES[suite]()
        }
        missing = [
            name for name in options.selected_cases
            if name not in candidates
        ]
        if missing:
            parser.error("unknown selected cases: " + ", ".join(missing))
        specs = [candidates[name] for name in options.selected_cases]
    else:
        suites = options.suite or ["smoke"]
        specs = []
        for suite in suites:
            specs.extend(SUITES[suite]())
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        parser.error("selected suites contain duplicate run names")

    output_root = make_output_root(options.output_root)
    print(f"ARTIFACT_ROOT={output_root}", flush=True)
    print(f"RUN_COUNT={len(specs)}", flush=True)
    results: list[RunResult] = []
    for index, spec in enumerate(specs, start=1):
        result = run_one(executable, output_root, spec, options.timeout)
        results.append(result)
        outcome = "OK" if result.passed else "BAD"
        print(
            f"[{index:03d}/{len(specs):03d}] {outcome} {spec.name} "
            f"status={result.actual_status} reason={result.actual_reason} "
            f"exit={result.exit_code}",
            flush=True,
        )
        if not result.passed:
            for error in result.errors:
                print(f"  {error}", flush=True)
        if result.visible_window_detected:
            print("ABORTED: visible snippet window detected", flush=True)
            break

    parity_groups: dict[str, list[RunResult]] = {}
    for result in results:
        if not result.joint_objective_signature:
            continue
        parity_key = result.name.replace(
            "-avbd-parallel", "-avbd"
        ).replace("-avbd-sequential", "-avbd")
        parity_groups.setdefault(parity_key, []).append(result)
    for parity_key, group in parity_groups.items():
        executions = {
            "parallel" if "-avbd-parallel" in result.name
            else "sequential" if "-avbd-sequential" in result.name
            else "other"
            for result in group
        }
        if not {"parallel", "sequential"}.issubset(executions):
            continue
        signatures = {
            result.joint_objective_signature for result in group
        }
        if len(signatures) != 1:
            for result in group:
                result.passed = False
                result.errors.append(
                    f"joint objective parallel/sequential signature "
                    f"mismatch in {parity_key}"
                )
            print(
                f"PARITY BAD {parity_key} joint objective signature "
                "mismatch",
                flush=True,
            )

    summary_json = output_root / "summary.json"
    summary_json.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    with (output_root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "name",
                "expected_status",
                "expected_reason",
                "actual_status",
                "actual_reason",
                "exit_code",
                "authority_count",
                "timed_out",
                "visible_window_detected",
                "executable_sha256_before",
                "executable_sha256_after",
                "residual_process",
                "joint_objective_signature",
                "passed",
                "log",
            ),
        )
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({key: row[key] for key in writer.fieldnames})

    failures = sum(not result.passed for result in results)
    expected_physical_failures = sum(
        result.expected_status == "FAIL" for result in results
    )
    print(
        f"SUMMARY runs={len(results)} accepted={len(results) - failures} "
        f"expectedPhysicalFailures={expected_physical_failures} "
        f"runnerFailures={failures}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
