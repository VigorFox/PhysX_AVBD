"""Build the reserved-full-leaf screen from the measured static policy."""
from pathlib import Path
import shutil
import subprocess
import sys

work = Path(__file__).resolve().parent

def edit(path, old, new):
    source = path.read_text()
    assert source.count(old) == 1, old
    path.write_bytes(source.replace(old, new).encode())

def add_reservation(variant):
    source = work / (variant + "-src") / "avbd"
    edit(source / "DyAvbdColorTasks.h",
         "        PxArray<AvbdMixedColorLane*> tasks;",
         "        bool singleFullLeaf=false;\n        PxArray<AvbdMixedColorLane*> tasks;")
    edit(source / "DyAvbdDynamics.cpp",
         "    mixed.lanes=lanes;\n    mLocalWork->cursor+=leaves;",
         """    mixed.lanes=lanes;
    // The first mixed ticket is the first ready assembly leaf.
    const auto& first=mLocalWork->nodes[mLocalWork->leaves[mixed.firstLeaf]];
    mixed.singleFullLeaf=first.end-first.begin==AvbdLocalWork::RowLimit &&
        tiles+leaves-1<=(lanes-1)*LocalQueueDepth;
    mLocalWork->cursor+=leaves;""")
    edit(source / "DyAvbdDynamics.cpp",
         """    const PxU32 perLane=count/mixed.lanes, extra=count%mixed.lanes;
    const PxU32 begin=rank*perLane+PxMin(rank,extra);
    const PxU32 claimed=perLane+(rank<extra ? 1u : 0u);""",
         """    const PxU32 reserved=mixed.singleFullLeaf ? 1u : 0u;
    const PxU32 lanes=mixed.lanes-reserved, remaining=count-reserved;
    const PxU32 perLane=remaining/lanes, extra=remaining%lanes;
    const bool singleton=reserved && !slot;
    const PxU32 begin=singleton ? 0u : reserved+rank*perLane+PxMin(rank,extra);
    const PxU32 claimed=singleton ? 1u : perLane+(rank<extra ? 1u : 0u);""")

if __name__ == "__main__":
    # The wall-clock study runs alone. Delay compilation until it is over.
    import json, time
    while True:
        try:
            if json.loads((work / "study.json").read_text()).get("complete"):
                break
        except (OSError, ValueError):
            pass
        time.sleep(3)
    shutil.copytree(work / "batch-src", work / "reserved-src")
    add_reservation("reserved")
    for base, variant in [("batch-tail", "reserved-tail")]:
        shutil.copytree(work / (base + "-src"), work / (variant + "-src"))
        (work / variant).mkdir()
        for item in (work / base).iterdir():
            if item.suffix.lower() in [".dll", ".exe", ".lib"]:
                shutil.copyfile(item, work / variant / item.name)
        add_reservation(variant)
        source = work / (variant + "-src") / "avbd"
        edit(source / "DyAvbdColorTasks.h",
             "longest, claims, hotLeaf, hotKind, hotRows, hotIndex;",
             "longest, claims, hotLeaf, hotKind, hotRows, hotIndex, reserved;")
        edit(source / "DyAvbdDynamics.cpp",
             "observed.hotRows,observed.hotIndex});",
             "observed.hotRows,observed.hotIndex,mColorTasks->mixed.singleFullLeaf ? 1u : 0u});")
        edit(source / "DyAvbdDynamics.cpp", "hotIndex=%u offer=%.4f", "hotIndex=%u reserved=%u offer=%.4f")
        edit(source / "DyAvbdDynamics.cpp", "v.hotRows,v.hotIndex,", "v.hotRows,v.hotIndex,v.reserved,")
        subprocess.run([sys.executable, work / "build.py", variant], check=True)

