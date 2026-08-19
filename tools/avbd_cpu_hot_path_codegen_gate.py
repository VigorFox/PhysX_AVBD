#!/usr/bin/env python3
"""Guard the MSVC code shape of the AVBD corotational polar hot loop.

The scalar regression recovered in R0 was caused by MSVC turning ordinary
inline PxMat33 add/scale and PxVec3 finite checks into calls inside the polar
iteration.  Source review alone cannot prove that the compiler kept those
operations inline, so this gate inspects COFF relocations in the dedicated
Release scalar-step object.  The polar loop consumes its already-validated
determinant and must not call the general PxMat33 inverse.  The gate also
verifies that the Scene consumer does not emit its own scalar-step copy.
Standalone component snippets deliberately retain a private copy because the
internal low-level implementation is not part of the PhysX DLL ABI.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECT = (
    ROOT
    / "physx/compiler/vc17win64-cpu-only/sdk_source_bin"
    / "LowLevelDynamics.dir/release/DyAvbdSoftBodyScalar.obj"
)
DEFAULT_CONSUMER_OBJECT = (
    ROOT
    / "physx/compiler/vc17win64-cpu-only/sdk_source_bin"
    / "SimulationController.dir/release/ScScene.obj"
)
DEFAULT_STEP_STATE_OBJECT = (
    ROOT
    / "physx/compiler/vc17win64-cpu-only/sdk_source_bin"
    / "LowLevelDynamics.dir/release/DyAvbdSoftBodyStepState.obj"
)

SYMBOL_ANCHOR = "?avbdStepSoftBodies@Dy@physx@@"
INVERSE_SYMBOL = "?getInverse@?$PxMat33T@M@physx@@"


def find_dumpbin(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise RuntimeError(f"dumpbin does not exist: {path}")

    on_path = shutil.which("dumpbin.exe") or shutil.which("dumpbin")
    if on_path:
        return Path(on_path)

    candidates: list[Path] = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(variable)
        if not root:
            continue
        vs_root = Path(root) / "Microsoft Visual Studio"
        if vs_root.is_dir():
            candidates.extend(vs_root.glob(
                "*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"))
    if not candidates:
        raise RuntimeError("dumpbin.exe was not found")
    return sorted(candidates)[-1]


def run_dumpbin(dumpbin: Path, option: str, obj: Path) -> list[str]:
    completed = subprocess.run(
        [str(dumpbin), option, str(obj)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        raise RuntimeError(
            f"dumpbin {option} failed ({completed.returncode}):\n"
            f"{completed.stdout}{completed.stderr}")
    return completed.stdout.splitlines()


def find_step_section(symbol_lines: list[str]) -> str:
    for line in symbol_lines:
        if (SYMBOL_ANCHOR in line and "External" in line and
                "notype ()" in line):
            match = re.search(r"\bSECT([0-9A-F]+)\b", line)
            if match:
                return match.group(1)
    raise RuntimeError("external avbdStepSoftBodies COFF section was not found")


def parse_section_relocations(
        relocation_lines: list[str], section: str) -> list[tuple[int, str]]:
    header = f"RELOCATIONS #{section}"
    try:
        start = relocation_lines.index(header) + 1
    except ValueError as exc:
        raise RuntimeError(f"missing relocation section {section}") from exc

    entries: list[tuple[int, str]] = []
    current_offset: int | None = None
    current_text: list[str] = []
    entry_pattern = re.compile(r"^\s*([0-9A-F]{8})\s+\S+")
    for line in relocation_lines[start:]:
        if line.startswith("RELOCATIONS #"):
            break
        match = entry_pattern.match(line)
        if match:
            if current_offset is not None:
                entries.append((current_offset, "".join(current_text)))
            current_offset = int(match.group(1), 16)
            current_text = [line]
        elif current_offset is not None:
            current_text.append(line.strip())
    if current_offset is not None:
        entries.append((current_offset, "".join(current_text)))
    return entries


def parse_section_size(header_lines: list[str], section: str) -> int:
    marker = f"SECTION HEADER #{section}"
    try:
        start = next(
            index for index, line in enumerate(header_lines)
            if marker in line)
    except StopIteration as exc:
        raise RuntimeError(f"missing section header {section}") from exc
    for line in header_lines[start + 1:start + 16]:
        match = re.match(r"^\s*([0-9A-F]+) size of raw data\s*$", line)
        if match:
            return int(match.group(1), 16)
    raise RuntimeError(f"missing raw size for section {section}")


def parse_stack_size(disassembly_lines: list[str]) -> int:
    try:
        start = next(
            index for index, line in enumerate(disassembly_lines)
            if line.lstrip().startswith(SYMBOL_ANCHOR))
    except StopIteration as exc:
        raise RuntimeError("avbdStepSoftBodies disassembly was not found") from exc
    for line in disassembly_lines[start + 1:start + 80]:
        match = re.search(r"\bmov\s+eax,([0-9A-F]+)h\b", line)
        if match:
            return int(match.group(1), 16)
    raise RuntimeError("avbdStepSoftBodies stack allocation was not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=Path, default=DEFAULT_OBJECT)
    parser.add_argument(
        "--consumer-object", type=Path, default=DEFAULT_CONSUMER_OBJECT)
    parser.add_argument(
        "--step-state-object", type=Path, default=DEFAULT_STEP_STATE_OBJECT)
    parser.add_argument("--dumpbin")
    parser.add_argument(
        "--max-section-bytes", type=lambda value: int(value, 0),
        default=0x8C00)
    parser.add_argument(
        "--max-stack-bytes", type=lambda value: int(value, 0),
        default=0x1A00)
    args = parser.parse_args()

    obj = args.object.resolve()
    if not obj.is_file():
        print(
            "[AVBD_CPU_HOT_PATH_CODEGEN_GATE] status=FAIL "
            f"error=missing-release-object object={obj}")
        return 1
    step_state_obj = args.step_state_object.resolve()
    if not step_state_obj.is_file():
        print(
            "[AVBD_CPU_HOT_PATH_CODEGEN_GATE] status=FAIL "
            f"error=missing-step-state-object object={step_state_obj}")
        return 1
    consumer_obj = args.consumer_object.resolve()
    if not consumer_obj.is_file():
        print(
            "[AVBD_CPU_HOT_PATH_CODEGEN_GATE] status=FAIL "
            f"error=missing-consumer-object object={consumer_obj}")
        return 1

    try:
        dumpbin = find_dumpbin(args.dumpbin)
        symbols = run_dumpbin(dumpbin, "/symbols", obj)
        consumer_symbols = run_dumpbin(dumpbin, "/symbols", consumer_obj)
        step_state_symbols = run_dumpbin(
            dumpbin, "/symbols", step_state_obj)
        section = find_step_section(symbols)
        relocations = parse_section_relocations(
            run_dumpbin(dumpbin, "/relocations", obj), section)
        section_size = parse_section_size(
            run_dumpbin(dumpbin, "/headers", obj), section)
        stack_size = parse_stack_size(
            run_dumpbin(dumpbin, "/disasm", obj))
    except RuntimeError as exc:
        print(f"[AVBD_CPU_HOT_PATH_CODEGEN_GATE] status=FAIL error={exc}")
        return 1

    inverse_offsets = sorted(
        offset for offset, symbol in relocations
        if INVERSE_SYMBOL in symbol)
    errors: list[str] = []
    consumer_step_symbols = sum(
        SYMBOL_ANCHOR in line and "External" in line and
        "notype ()" in line and "SECT" in line
        for line in consumer_symbols)
    scene_step_state_symbols = sum(
        "AvbdSoftBodyStepState@Dy@physx" in line and "SECT" in line
        for line in consumer_symbols)
    implementation_step_state_symbols = sum(
        "AvbdSoftBodyStepState@Dy@physx" in line
        for line in step_state_symbols)
    if scene_step_state_symbols:
        errors.append(
            "scene-still-emits-step-state-symbols=" +
            str(scene_step_state_symbols))
    if consumer_step_symbols:
        errors.append(
            "consumer-still-emits-scalar-step-symbols=" +
            str(consumer_step_symbols))
    if not implementation_step_state_symbols:
        errors.append("dedicated-step-state-object-has-no-owned-symbols")
    # Component velocity finalization owns a separate inverse-inertia solve and
    # is deliberately out of line.  The scalar step's polar iteration already
    # owns the validated determinant and must not recompute it through the
    # general PxMat33 inverse.
    if inverse_offsets:
        errors.append(
            "unexpected-polar-getInverse-sites=" +
            str(len(inverse_offsets)))
    if section_size > args.max_section_bytes:
        errors.append(
            f"step-section-bytes=0x{section_size:x} "
            f"limit=0x{args.max_section_bytes:x}")
    if stack_size > args.max_stack_bytes:
        errors.append(
            f"step-stack-bytes=0x{stack_size:x} "
            f"limit=0x{args.max_stack_bytes:x}")

    if errors:
        for error in errors:
            print(
                "[AVBD_CPU_HOT_PATH_CODEGEN_GATE] "
                f"status=FAIL error={error}")
        return 1

    offsets = ",".join(f"0x{offset:x}" for offset in inverse_offsets) or "none"
    print(
        "[AVBD_CPU_HOT_PATH_CODEGEN_GATE] "
        f"object={obj.name} section={section} "
        f"sectionBytes=0x{section_size:x} stackBytes=0x{stack_size:x} "
        f"sceneStepStateSymbols={scene_step_state_symbols} "
        f"sceneStepSymbols={consumer_step_symbols} "
        f"implementationStepStateSymbols={implementation_step_state_symbols} "
        f"polarInverseSites={offsets} "
        "outOfLineGeneralInverse=0 status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
