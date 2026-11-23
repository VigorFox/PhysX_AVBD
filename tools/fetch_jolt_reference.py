#!/usr/bin/env python3
"""Fetch the pinned, unmodified Jolt reference without changing other clones."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
repository = "https://github.com/jrouwe/JoltPhysics.git"
revision = "a45f543e24f630b33fd3366c7c2ae241b63816b3"
target = root / "reference" / "JoltPhysics"
if not target.exists():
    target.parent.mkdir(exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", repository, str(target)], check=True)
elif not (target / ".git").is_dir():
    raise SystemExit(f"Refusing to overwrite non-clone directory: {target}")
elif subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"]):
    raise SystemExit(f"Reference has local changes: {target}")
if subprocess.run(["git", "-C", str(target), "cat-file", "-e", revision + "^{commit}"],
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
    subprocess.run(["git", "-C", str(target), "fetch", "origin", revision], check=True)
subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)
print(target)
