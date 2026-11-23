#!/usr/bin/env python3
"""Replay an immutable diagnostic capture, never count it as a physics PASS."""
import argparse
import json
from pathlib import Path
import tempfile

from run_avbd_ogc_certificate_probe import ROOT, run_probe


def fields(line):
    return dict(word.split("=", 1) for word in line.split() if "=" in word)


def capture_data(text):
    lines = text.splitlines()
    block = [fields(line) for line in lines if line.startswith("[RECOVERY_BLOCK]")
             and fields(line).get("cache") == "3"]
    captured = [fields(line) for line in lines if line.startswith("[RECOVERY_CAPTURE]")]
    if len(block) != 1 or len(captured) != 1:
        raise ValueError("expected exactly one captured cache-3 block and metric")
    planes = [fields(line) for line in lines if line.startswith("[RECOVERY_OGC]")
              and fields(line).get("cache") == "3"]
    domain = [fields(line) for line in lines if line.startswith("[RECOVERY_DOMAIN]")]
    pairs = [line.split()[4:] for line in lines if line.startswith("[RECOVERY_PAIR]")]
    if len(planes) != int(block[0]["ogcPlanes"]) or len(domain) != int(block[0]["commitPlanes"]):
        raise ValueError("incomplete captured domain")
    if any(p["body"] != "4294967295" or p["peer"] != "4294967295" for p in planes):
        raise ValueError("this replay requires the captured static partner mapping")
    if len(pairs) != 5760 or any(len(pair) != 36 for pair in pairs):
        raise ValueError("incomplete captured triangle pairs")
    values = [block[0]["p"], captured[0]["anchor"], block[0]["delta"], captured[0]["H"], str(len(planes))]
    values.extend(p["n"] + " " + p["bound"] for p in planes)
    values.append(str(len(domain)))
    values.extend(p["n"] + " " + p["b"] + " " + p["roundoff"] for p in domain)
    values.append(str(len(pairs)))
    values.extend(" ".join(pair) for pair in pairs)
    return "\n".join(values).replace(",", " ") + "\n"


def fixture_data(fixture):
    values = [fixture["point"], fixture["anchor"], fixture["delta"], fixture["metric"],
              str(len(fixture["planes"]))]
    values.extend(fixture["planes"])
    values.append(str(len(fixture["domain"])))
    values.extend(fixture["domain"])
    values.append(str(len(fixture["pairs"])))
    for pair in fixture["pairs"]:
        values.append(" ".join(fixture["triangles"][index] for index in pair))
    return "\n".join(values) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, nargs="?",
                        default=ROOT / "tools/fixtures/c01_showcase_recovery.json")
    args = parser.parse_args()
    text = args.capture.read_text(encoding="utf8")
    data = fixture_data(json.loads(text)) if args.capture.suffix == ".json" else capture_data(text)
    with tempfile.TemporaryDirectory(prefix="avbd_recovery_") as directory:
        path = Path(directory) / "capture.txt"
        path.write_text(data, encoding="ascii")
        return run_probe(ROOT / "tools/avbd_recovery_residual_probe.cpp", [str(path)])


if __name__ == "__main__":
    raise SystemExit(main())
