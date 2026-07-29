#!/usr/bin/env python3
"""Shared parser for the AVBD compiled contact-objective IR diagnostic."""

from __future__ import annotations


PREFIX = "[avbd:contact-objective-ir] "
PARTITION_FIELDS = (
    "contactObjectivePositionSlots",
    "contactObjectivePointSlots",
    "contactObjectiveManifoldSlots",
    "contactObjectiveComponentSlots",
    "contactObjectiveJointSlots",
    "contactObjectiveUnsupportedSlots",
    "contactObjectiveLegacySlots",
    "contactObjectiveInvalidSlots",
)
LEGACY_SOURCE_FIELDS = (
    "contactObjectiveLegacyNormalSlots",
    "contactObjectiveLegacyTangentSlots",
)
LEGACY_TANGENT_TOPOLOGY_FIELDS = (
    "contactObjectiveLegacyRigidStaticTangentSlots",
    "contactObjectiveLegacyDynamicTangentSlots",
    "contactObjectiveLegacyDeformableTangentSlots",
    "contactObjectiveLegacyJointMixedTangentSlots",
    "contactObjectiveLegacyOtherTangentSlots",
)
OWNER_FIELDS = {
    "PositionAL": "contactObjectivePositionSlots",
    "PointFinalize": "contactObjectivePointSlots",
    "ManifoldFinalize": "contactObjectiveManifoldSlots",
    "ComponentFinalize": "contactObjectiveComponentSlots",
    "JointFinalize": "contactObjectiveJointSlots",
    "Unsupported": "contactObjectiveUnsupportedSlots",
}


def validate_contact_objective_ir(
    output: str,
    *,
    required_owners: tuple[str, ...] = ("PositionAL",),
    allow_unsupported: bool = False,
) -> tuple[list[str], str]:
    """Validate exact source-slot partitions and required unique owners."""

    unknown = [
        owner for owner in required_owners if owner not in OWNER_FIELDS
    ]
    if unknown:
        raise ValueError(
            "unsupported required owner(s): " + ", ".join(unknown)
        )

    errors: list[str] = []
    lines = [
        line.strip()
        for line in output.splitlines()
        if line.startswith(PREFIX)
    ]
    if not lines:
        return ["no [avbd:contact-objective-ir] diagnostic samples"], ""

    totals = {field: 0 for field in PARTITION_FIELDS}
    signatures: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        fields: dict[str, str] = {}
        for token in line[len(PREFIX) :].split():
            if "=" not in token:
                errors.append(
                    f"contact objective diagnostic {line_number} "
                    f"has malformed token {token!r}"
                )
                continue
            key, value = token.split("=", 1)
            if key in fields:
                errors.append(
                    f"contact objective diagnostic {line_number} "
                    f"duplicates {key}"
                )
            fields[key] = value

        required = (
            "contactObjectiveSlots",
            *PARTITION_FIELDS,
            *LEGACY_SOURCE_FIELDS,
            *LEGACY_TANGENT_TOPOLOGY_FIELDS,
            "contactObjectiveFingerprint",
        )
        values: dict[str, int] = {}
        for key in required:
            try:
                values[key] = int(fields[key])
            except (KeyError, ValueError):
                errors.append(
                    f"contact objective diagnostic {line_number} "
                    f"has {key}={fields.get(key)!r}, expected integer"
                )
        if len(values) != len(required):
            continue

        partition = sum(values[key] for key in PARTITION_FIELDS)
        if values["contactObjectiveSlots"] != partition:
            errors.append(
                f"contact objective diagnostic {line_number} has "
                "contactObjectiveSlots="
                f"{values['contactObjectiveSlots']} but "
                f"partition={partition}"
            )
        if values["contactObjectiveInvalidSlots"] != 0:
            errors.append(
                f"contact objective diagnostic {line_number} has "
                "contactObjectiveInvalidSlots="
                f"{values['contactObjectiveInvalidSlots']}"
            )
        legacy_sources = sum(
            values[key] for key in LEGACY_SOURCE_FIELDS
        )
        if values["contactObjectiveLegacySlots"] != legacy_sources:
            errors.append(
                f"contact objective diagnostic {line_number} has "
                "contactObjectiveLegacySlots="
                f"{values['contactObjectiveLegacySlots']} but "
                f"legacy source partition={legacy_sources}"
            )
        legacy_tangent_topologies = sum(
            values[key] for key in LEGACY_TANGENT_TOPOLOGY_FIELDS
        )
        if (
            values["contactObjectiveLegacyTangentSlots"]
            != legacy_tangent_topologies
        ):
            errors.append(
                f"contact objective diagnostic {line_number} has "
                "contactObjectiveLegacyTangentSlots="
                f"{values['contactObjectiveLegacyTangentSlots']} but "
                f"topology partition={legacy_tangent_topologies}"
            )
        if (
            values["contactObjectiveSlots"] > 0
            and values["contactObjectivePositionSlots"] == 0
        ):
            errors.append(
                f"contact objective diagnostic {line_number} has contact "
                "source slots but no PositionAL geometry owner"
            )
        for key in PARTITION_FIELDS:
            totals[key] += values[key]
        signatures.append(
            ":".join(str(values[key]) for key in required)
        )

    for owner in required_owners:
        if totals[OWNER_FIELDS[owner]] == 0:
            errors.append(
                f"focused lane compiled no {owner} contact objective"
            )
    if (
        not allow_unsupported
        and totals["contactObjectiveUnsupportedSlots"] != 0
    ):
        errors.append("focused lane fell back to Unsupported contact slots")
    return errors, "|".join(signatures)
