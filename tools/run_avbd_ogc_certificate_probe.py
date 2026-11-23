#!/usr/bin/env python3
"""Build/run production OGC counterexamples. Known violations return exit 1."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
VCVARS = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")


def build_probe(work, binary_dir, source, extra_sources=(), include_dirs=(), defines=(), libraries=(),
                compiler_options=(), config="checked"):
    if not VCVARS.is_file():
        print("Missing Visual Studio environment: %s" % VCVARS)
        return 2
    command = work / "build.cmd"
    command.write_text(
        '@echo off\ncall "%s"\nif errorlevel 1 exit /b 2\n' % VCVARS +
        'cl /nologo /EHsc /std:c++14 ' +
        ('/Od /MDd /D_DEBUG ' if config == "debug" else '/O2 /MD /DNDEBUG ') +
        ''.join('%s ' % option for option in compiler_options) +
        ''.join('/I"%s" ' % path for path in include_dirs) +
        ''.join('/D%s ' % define for define in defines) +
        '/I"%s" /I"%s" %s /Fe:probe.exe /link "%s" %s\n' % (
            ROOT / "physx/include", ROOT / "physx/source/lowleveldynamics/src",
            " ".join('"%s"' % path for path in [source] + list(extra_sources)),
            binary_dir / "PhysXFoundation_64.lib",
            " ".join('"%s"' % (binary_dir / name) for name in libraries)), encoding="ascii")
    build = subprocess.run(["cmd", "/c", str(command)], cwd=work, capture_output=True)
    if build.returncode:
        sys.stdout.buffer.write(build.stdout + build.stderr)
        return 2
    return 0


def retire_probe(executable):
    # Windows can retain a mapped image briefly after the process has exited.
    for attempt in range(6):
        try:
            executable.unlink(missing_ok=True)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(0.05 * 2**attempt)


def run_probe(source, args=(), extra_sources=(), include_dirs=(), defines=(), libraries=(), compiler_options=(), config="checked"):
    with tempfile.TemporaryDirectory(prefix="avbd_certificate_") as directory:
        work = Path(directory)
        binary_dir = Path(os.environ.get("AVBD_PROBE_BINARY_DIR",
                          str(ROOT / "physx/bin/win.x86_64.vc143.md" / config))).resolve()
        code = build_probe(work, binary_dir, source, extra_sources, include_dirs, defines, libraries, compiler_options, config)
        if code:
            return code
        env = os.environ.copy()
        env["PATH"] = str(binary_dir) + os.pathsep + env.get("PATH", "")
        executable = work / "probe.exe"
        try:
            return subprocess.run([str(executable)] + list(args), cwd=work, env=env).returncode
        finally:
            retire_probe(executable)


def main():
    return run_probe(ROOT / "tools/avbd_ogc_certificate_probe.cpp")


if __name__ == "__main__":
    raise SystemExit(main())
