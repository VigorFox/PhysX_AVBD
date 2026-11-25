"""Compare support, release, impact reporting and free joint spin at three step sizes."""
import os
import subprocess
import tempfile
from pathlib import Path
from run_avbd_ogc_certificate_probe import ROOT, build_probe


def main():
    default_sdk = ROOT / "physx/bin/win.x86_64.vc143.md/checked"
    sdk = Path(os.environ.get("AVBD_PROBE_BINARY_DIR", default_sdk)).resolve()
    libs = ["PhysXExtensions_static_64.lib", "PhysXPvdSDK_static_64.lib", "PhysX_64.lib",
            "PhysXCommon_64.lib", "PhysXCooking_64.lib"]
    with tempfile.TemporaryDirectory(prefix="physx_contact_reaction_") as tmp:
        work = Path(tmp)
        build_status = build_probe(work, sdk, ROOT / "tools/avbd_contact_reaction_probe.cpp", libraries=libs)
        if build_status:
            return build_status
        env = dict(os.environ, PATH=str(sdk) + os.pathsep + os.environ["PATH"])
        for solver in ("avbd", "tgs"):
            for hz in (30, 60, 120):
                print(f"[CONTACT_REACTION] solver={solver} hz={hz}", flush=True)
                process = subprocess.run([work / "probe.exe", solver, str(hz)], cwd=work, env=env, timeout=30,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
                print(process.stdout, end="")
                if process.returncode or "invalid operation" in process.stdout.lower():
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
