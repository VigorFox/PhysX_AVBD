"""Inspect SDK 5.9 little-endian PVD debug recordings without opening a window.

Checks actual CreateInstance/render events, not merely class-name strings.
Wire formats: PxPvdCommStreamEvents.h and PxPvdUserRenderImpl.h in this SDK.
This is a stream/presentation check, not a physics or GUI rendering oracle.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, count):
        end = self.pos + count
        if count < 0 or end > len(self.data):
            raise ValueError(f"Truncated PVD stream at {self.pos}, requested {count}")
        result = self.data[self.pos:end]
        self.pos = end
        return result

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))

    def u32(self):
        return self.unpack("I")[0]

    def blob(self):
        return self.take(self.u32())

    def string(self):
        return self.blob().rstrip(b"\0").decode("utf-8")


def inspect(path):
    data = path.read_bytes()
    stream = Reader(data)
    magic, version, _, _, flags = stream.unpack("IIQQI")
    if (magic, version, flags) != (837150850, 1, 1):
        raise ValueError("Expected a little-endian, 64-bit SDK PVD v1 stream")
    names, instances, sections, meshes = {}, Counter(), Counter(), {}
    renderers = set()
    render_packets = 0
    wires = {}
    groups = events = 0

    def primitives(reader, vertices):
        kind = "lines" if vertices == 2 else "triangles"
        target = wires if vertices == 2 else meshes
        count = reader.u32()
        raw = reader.take(count * vertices * 16)
        if not count:
            return
        counts, first_positions = Counter(), {}
        for index, vertex in enumerate(struct.iter_unpack("<fffI", raw)):
            if not all(math.isfinite(value) for value in vertex[:3]):
                raise ValueError("Non-finite PVD skin geometry")
            if index % vertices == 0:
                counts[vertex[3]] += 1
                first_positions.setdefault(vertex[3], vertex[:3])
        for color, total in counts.items():
            first = first_positions[color]
            item = target.setdefault(f"0x{color:08x}", {"packets": 0, kind: total,
                                     "firstVertex": first, "lastVertex": first, "moved": False})
            item["packets"] += 1
            item["lastVertex"] = first
            item["moved"] |= first != item["firstVertex"]
            if item[kind] != total:
                raise ValueError("Unexpected changing display topology")

    def render(payload):
        reader = Reader(payload)
        while reader.pos < len(payload):
            event = reader.unpack("B")[0]
            if event == 1:
                reader.take(8)  # instance ID
            elif event == 2:
                reader.take(reader.u32() * 16)
            elif event == 3:
                primitives(reader, 2)
            elif event == 4:
                primitives(reader, 3)
            elif event == 11:  # SDK's batched debug render buffer
                reader.take(reader.u32() * 16)
                primitives(reader, 2)
                primitives(reader, 3)
            else:
                raise ValueError(f"Unsupported render event {event}")

    while stream.pos < len(data):
        size, count, stream_id, _ = stream.unpack("IIQQ")
        r = Reader(stream.take(size))
        groups += 1
        for _ in range(count):
            event = r.unpack("B")[0]
            events += 1
            if event == 1:
                value = r.string()
                names[stream_id, r.u32()] = value
            elif event == 2:
                r.take(8)
            elif event == 3:
                r.take(16)
            elif event == 4:
                r.take(25)
                r.take(r.u32() * 8)
            elif event == 5:
                r.take(16)
                r.take(r.u32() * 20)
                r.take(4)
            elif event == 6:
                _, name, instance = r.unpack("IIQ")
                class_name = names[stream_id, name]
                instances[class_name] += 1
                if class_name == "PvdUserRenderer":
                    renderers.add(instance)
            elif event == 7:
                instance, name, _, _, _ = r.unpack("QIIII")
                payload = r.blob()
                prop = names[stream_id, name]
                if prop == "events" and instance in renderers:
                    render_packets += 1
                    render(payload)
            elif event == 8:
                r.take(20)
            elif event == 9:
                r.blob()
                r.take(4)
            elif event in (10, 14):
                pass
            elif event == 11:
                r.take(16)
                r.blob()
            elif event == 12:
                r.take(8)
            elif event == 15:
                renderers.discard(r.unpack("Q")[0])
            elif event == 13:
                r.take(8)
                r.blob()
            elif event in (16, 17, 28):
                r.take(20)
            elif event in (18, 19):
                _, name, _ = r.unpack("QIQ")
                sections[("begin:" if event == 18 else "end:") + names[stream_id, name]] += 1
            elif event in (20, 22):
                r.take(9)
            elif event == 21:
                r.take(12)
            elif event == 23:
                r.string()
                r.take(36)
            elif event in (24, 25):
                r.take(8)
                r.string()
                if event == 25:
                    r.take(3)
            elif event == 26:
                r.string()
            elif event == 27:
                code = r.u32()
                message, file = r.string(), r.string()
                line = r.u32()
                raise ValueError(f"PVD error {code}: {message} at {file}:{line}")
            else:
                raise ValueError(f"Unknown PVD event {event}")
        if r.pos != size:
            raise ValueError(f"PVD event group size mismatch: {r.pos} != {size}")
    return {"file": str(path), "bytes": len(data), "groups": groups, "events": events,
            "instances": dict(instances), "sections": dict(sections),
            "renderPropertyPackets": render_packets, "skins": meshes, "wireMeshes": wires}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.capture), indent=2))


if __name__ == "__main__":
    main()
