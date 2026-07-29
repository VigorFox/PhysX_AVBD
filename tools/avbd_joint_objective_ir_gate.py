#!/usr/bin/env python3
"""Shared parser for the AVBD compiled joint-objective IR diagnostic."""

from __future__ import annotations


PREFIX = "[avbd:joint-objective-ir] "
PARTITION_FIELDS = (
    "jointObjectivePositionRows",
    "jointObjectiveFinalizeRows",
    "jointObjectiveUnsupportedRows",
    "jointObjectiveLegacyRows",
    "jointObjectiveInvalidRows",
)
OWNER_FIELDS = {
    "PositionAL": "jointObjectivePositionRows",
    "JointFinalize": "jointObjectiveFinalizeRows",
}


def validate_joint_objective_ir(
    output: str,
    *,
    expected_owner: str,
    allow_unsupported: bool = False,
    require_unsupported: bool = False,
) -> tuple[list[str], str]:
    """Validate exact partitions and require at least one expected owner."""

    if expected_owner not in OWNER_FIELDS:
        raise ValueError(f"unsupported expected owner: {expected_owner}")

    errors: list[str] = []
    lines = [
        line.strip()
        for line in output.splitlines()
        if line.startswith(PREFIX)
    ]
    if not lines:
        return ["no [avbd:joint-objective-ir] diagnostic samples"], ""

    totals = {field: 0 for field in PARTITION_FIELDS}
    signatures: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        fields: dict[str, str] = {}
        for token in line[len(PREFIX) :].split():
            if "=" not in token:
                errors.append(
                    f"joint objective diagnostic {line_number} "
                    f"has malformed token {token!r}"
                )
                continue
            key, value = token.split("=", 1)
            if key in fields:
                errors.append(
                    f"joint objective diagnostic {line_number} "
                    f"duplicates {key}"
                )
            fields[key] = value

        required = (
            "jointObjectiveRows",
            *PARTITION_FIELDS,
            "jointObjectiveFingerprint",
        )
        values: dict[str, int] = {}
        for key in required:
            try:
                values[key] = int(fields[key])
            except (KeyError, ValueError):
                errors.append(
                    f"joint objective diagnostic {line_number} "
                    f"has {key}={fields.get(key)!r}, expected integer"
                )
        if len(values) != len(required):
            continue

        partition = sum(values[key] for key in PARTITION_FIELDS)
        if values["jointObjectiveRows"] != partition:
            errors.append(
                f"joint objective diagnostic {line_number} has "
                f"jointObjectiveRows={values['jointObjectiveRows']} "
                f"but partition={partition}"
            )
        if values["jointObjectiveInvalidRows"] != 0:
            errors.append(
                f"joint objective diagnostic {line_number} has "
                f"jointObjectiveInvalidRows="
                f"{values['jointObjectiveInvalidRows']}"
            )
        for key in PARTITION_FIELDS:
            totals[key] += values[key]
        signatures.append(
            ":".join(str(values[key]) for key in required)
        )

    owner_field = OWNER_FIELDS[expected_owner]
    if totals[owner_field] == 0:
        errors.append(
            f"focused lane compiled no {expected_owner} objective"
        )
    if (
        not allow_unsupported
        and totals["jointObjectiveUnsupportedRows"] != 0
    ):
        errors.append("focused lane fell back to Unsupported")
    if (
        require_unsupported
        and totals["jointObjectiveUnsupportedRows"] == 0
    ):
        errors.append(
            "focused lane did not expose the expected Unsupported objective"
        )
    return errors, "|".join(signatures)
