#!/usr/bin/env python3
"""Read Windows core records and derive explicit AVBD benchmark masks.

The selectors deliberately describe the OS topology facts they require rather
than guessing that a logical-CPU number denotes a performance core or an SMT
sibling.  A machine-specific handoff may map the discovered records to vendor
core names only after recording the raw topology output.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os


RELATION_PROCESSOR_CORE = 0


class _GroupAffinity(ctypes.Structure):
    _fields_ = (
        ("mask", ctypes.c_size_t),
        ("group", wintypes.WORD),
        ("reserved", wintypes.WORD * 3),
    )


class _ProcessorRelationshipHeader(ctypes.Structure):
    _fields_ = (
        ("flags", wintypes.BYTE),
        ("efficiency_class", wintypes.BYTE),
        ("reserved", wintypes.BYTE * 20),
        ("group_count", wintypes.WORD),
    )


@dataclass(frozen=True)
class WindowsCpuCore:
    """One Windows ``RelationProcessorCore`` record."""

    ordinal: int
    efficiency_class: int
    flags: int
    group_masks: tuple[tuple[int, int], ...]

    @property
    def logical_processor_count(self) -> int:
        # Keep the helper compatible with the repository's older Python
        # runners, which predate ``int.bit_count``.
        return sum(bin(mask).count("1") for _, mask in self.group_masks)

    @property
    def smt_capable(self) -> bool:
        # LTP_PC_SMT is bit 0 of PROCESSOR_RELATIONSHIP.Flags.
        return bool(self.flags & 1)

    @property
    def is_group_zero(self) -> bool:
        return all(group == 0 for group, _ in self.group_masks)


@dataclass(frozen=True)
class TopologyAffinityConfiguration:
    """A named mask with the exact topology-selection evidence behind it."""

    name: str
    cpu_affinity_mask: int
    selected_core_ordinals: tuple[int, ...]
    description: str


TOPOLOGY_CONFIGURATION_NAMES = (
    "smt-four-physical",
    "smt-two-physical-pairs",
    "single-logical-four-physical",
)


def discover_windows_cpu_cores() -> tuple[WindowsCpuCore, ...]:
    """Return all Windows processor-core records without vendor assumptions."""

    if os.name != "nt":
        raise RuntimeError("Windows CPU topology discovery requires Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetLogicalProcessorInformationEx.argtypes = (
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetLogicalProcessorInformationEx.restype = wintypes.BOOL
    byte_count = wintypes.DWORD()
    kernel32.GetLogicalProcessorInformationEx(
        RELATION_PROCESSOR_CORE, None, ctypes.byref(byte_count)
    )
    if byte_count.value == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    storage = (ctypes.c_ubyte * byte_count.value)()
    if not kernel32.GetLogicalProcessorInformationEx(
        RELATION_PROCESSOR_CORE, ctypes.byref(storage), ctypes.byref(byte_count)
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    records: list[WindowsCpuCore] = []
    offset = 0
    core_ordinal = 0
    information_header_size = ctypes.sizeof(wintypes.DWORD) * 2
    processor_header_size = ctypes.sizeof(_ProcessorRelationshipHeader)
    group_affinity_size = ctypes.sizeof(_GroupAffinity)
    while offset < byte_count.value:
        if offset + information_header_size > byte_count.value:
            raise RuntimeError("truncated Windows processor topology header")
        relationship = ctypes.c_int.from_buffer_copy(storage, offset).value
        record_size = wintypes.DWORD.from_buffer_copy(storage, offset + 4).value
        if record_size < information_header_size or offset + record_size > byte_count.value:
            raise RuntimeError("invalid Windows processor topology record size")
        if relationship == RELATION_PROCESSOR_CORE:
            processor_offset = offset + information_header_size
            if processor_offset + processor_header_size > offset + record_size:
                raise RuntimeError("truncated Windows processor-core record")
            processor = _ProcessorRelationshipHeader.from_buffer_copy(
                storage, processor_offset
            )
            affinity_offset = processor_offset + processor_header_size
            affinity_end = affinity_offset + processor.group_count * group_affinity_size
            if affinity_end > offset + record_size:
                raise RuntimeError("truncated Windows processor group-affinity record")
            group_masks = tuple(
                (
                    affinity.group,
                    int(affinity.mask),
                )
                for affinity in (
                    _GroupAffinity.from_buffer_copy(
                        storage, affinity_offset + index * group_affinity_size
                    )
                    for index in range(processor.group_count)
                )
            )
            if not group_masks or any(mask == 0 for _, mask in group_masks):
                raise RuntimeError("Windows processor-core record has no logical mask")
            records.append(
                WindowsCpuCore(
                    ordinal=core_ordinal,
                    efficiency_class=processor.efficiency_class,
                    flags=processor.flags,
                    group_masks=group_masks,
                )
            )
            core_ordinal += 1
        offset += record_size
    if not records:
        raise RuntimeError("Windows returned no processor-core topology records")
    return tuple(records)


def _require_group_zero(cores: tuple[WindowsCpuCore, ...]) -> None:
    if not all(core.is_group_zero for core in cores):
        raise RuntimeError(
            "the P5 affinity benchmark currently supports only group-0 core records"
        )


def _lowest_set_bit(mask: int) -> int:
    return mask & -mask


def resolve_topology_affinity_configuration(
    name: str,
) -> TopologyAffinityConfiguration:
    """Resolve one documented, topology-derived P5.21 affinity configuration."""

    if name not in TOPOLOGY_CONFIGURATION_NAMES:
        raise ValueError(f"unknown topology configuration: {name}")
    cores = discover_windows_cpu_cores()
    _require_group_zero(cores)
    smt_cores = tuple(core for core in cores if core.smt_capable)
    single_logical_cores = tuple(
        core for core in cores if not core.smt_capable and core.logical_processor_count == 1
    )
    if name == "smt-four-physical":
        selected = smt_cores[:4]
        if len(selected) != 4:
            raise RuntimeError("need four SMT-capable physical-core records")
        mask = sum(_lowest_set_bit(core.group_masks[0][1]) for core in selected)
        description = "four SMT-capable core records; one logical processor from each"
    elif name == "smt-two-physical-pairs":
        selected = smt_cores[:2]
        if len(selected) != 2 or any(core.logical_processor_count != 2 for core in selected):
            raise RuntimeError("need two dual-logical SMT-capable physical-core records")
        mask = sum(core.group_masks[0][1] for core in selected)
        description = "two dual-logical SMT-capable core records; both logical processors"
    else:
        selected = single_logical_cores[:4]
        if len(selected) != 4:
            raise RuntimeError("need four single-logical physical-core records")
        mask = sum(core.group_masks[0][1] for core in selected)
        description = "four single-logical core records; one logical processor from each"
    if mask <= 0:
        raise RuntimeError("topology configuration resolved an empty affinity mask")
    return TopologyAffinityConfiguration(
        name=name,
        cpu_affinity_mask=mask,
        selected_core_ordinals=tuple(core.ordinal for core in selected),
        description=description,
    )


def _format_core(core: WindowsCpuCore) -> str:
    masks = ",".join(f"g{group}:0x{mask:X}" for group, mask in core.group_masks)
    return (
        "[WINDOWS_CPU_TOPOLOGY_CORE] "
        f"ordinal={core.ordinal} efficiency-class={core.efficiency_class} "
        f"smt-capable={int(core.smt_capable)} "
        f"logical-processors={core.logical_processor_count} masks={masks}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolve", choices=TOPOLOGY_CONFIGURATION_NAMES)
    arguments = parser.parse_args()
    for core in discover_windows_cpu_cores():
        print(_format_core(core))
    if arguments.resolve:
        configuration = resolve_topology_affinity_configuration(arguments.resolve)
        print(
            "[WINDOWS_CPU_TOPOLOGY_CONFIGURATION] "
            f"name={configuration.name} "
            f"affinity-mask=0x{configuration.cpu_affinity_mask:X} "
            f"selected-core-ordinals={','.join(map(str, configuration.selected_core_ordinals))} "
            f"description={configuration.description.replace(' ', '-')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
