"""Platform helpers for the AVBD headless benchmarks.

Windows keeps the historical SDK layout (``win.x86_64.vc143.md``, ``*_64.exe``,
``*.dll``, process-affinity mask, 100 ns process times). Linux uses the SDK
layout produced by ``generate_projects.sh linux-gcc-cpu-only``
(``bin/linux.x86_64/<config>/``), accepts either ``SnippetName_64`` or
``SnippetName``, and reads process CPU from ``wait4`` so the sample still
exists after the child is reaped. Times are 100 ns units on both platforms,
matching ``GetProcessTimes`` / ``GetSystemTimes``.
"""
import os
import subprocess
import time
from pathlib import Path


def is_windows():
    return os.name == "nt"


def default_sdk_dir(root, config="checked"):
    root = Path(root)
    if is_windows():
        return root / "physx/bin/win.x86_64.vc143.md" / config
    return root / "physx/bin/linux.x86_64" / config


def find_snippet(sdk, base):
    """Return the first existing snippet binary for ``base`` (no suffix)."""
    sdk = Path(sdk)
    names = (f"{base}_64.exe", f"{base}.exe") if is_windows() else (f"{base}_64", base)
    for name in names:
        path = sdk / name
        if path.is_file():
            return path
    return None


def runtime_binaries(sdk, bases=()):
    """Shared libraries plus any snippet binaries that are actually present."""
    sdk = Path(sdk)
    found = []
    seen = set()
    for pattern in (("*.dll",) if is_windows() else ("*.so", "*.so.*")):
        for path in sorted(sdk.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                found.append(path)
    for base in bases:
        path = find_snippet(sdk, base)
        if path is not None and path not in seen:
            seen.add(path)
            found.append(path)
    return found


def cpu_ticks(process=None):
    """System or process CPU in 100 ns units.

    With ``process is None``, return ``(total, busy)`` where ``total`` includes
    idle, matching ``GetSystemTimes`` (kernel time includes idle). With a
    process, return that process's user+kernel time. On Linux a reaped child
    has no ``/proc`` entry; use :func:`wait_child` to capture it at exit.
    """
    if is_windows():
        return _windows_cpu_ticks(process)
    if process is not None:
        return _linux_proc_cpu_100ns(process.pid)
    return _linux_system_cpu_100ns()


def wait_child(child, timeout):
    """Wait for ``child`` and return ``(returncode, cpu_100ns)``.

    ``cpu_100ns`` is ``None`` when the platform cannot report it. A timeout
    kills the child and raises ``subprocess.TimeoutExpired``, as ``Popen.wait``
    does.
    """
    if is_windows():
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            raise
        return code, _windows_cpu_ticks(child)
    deadline = time.perf_counter() + timeout
    while True:
        pid, status, usage = os.wait4(child.pid, os.WNOHANG)
        if pid:
            child.returncode = os.waitstatus_to_exitcode(status)
            cpu = int((usage.ru_utime + usage.ru_stime) * 1e7)
            return child.returncode, cpu
        if time.perf_counter() >= deadline:
            child.kill()
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
            raise subprocess.TimeoutExpired(child.args, timeout)
        time.sleep(0.02)


def process_affinity(mask):
    """Read or set this process's CPU affinity. ``mask`` is a bit mask or None.

    The child inherits the mask. Returns the resulting mask as a hex string.
    """
    if is_windows():
        return _windows_affinity(mask)
    allowed = set(os.sched_getaffinity(0))
    if mask is not None:
        wanted = {bit for bit in range(256) if mask & (1 << bit)}
        if not wanted or not wanted <= allowed:
            raise ValueError("CPU mask exceeds process affinity")
        os.sched_setaffinity(0, wanted)
        allowed = set(os.sched_getaffinity(0))
        if allowed != wanted:
            raise ValueError("Requested affinity unavailable")
    bits = 0
    for cpu in allowed:
        bits |= 1 << cpu
    return hex(bits)


def inherited_cpus(pid):
    if is_windows():
        import psutil
        return psutil.Process(pid).cpu_affinity()
    return sorted(os.sched_getaffinity(pid))


def _windows_cpu_ticks(process):
    import ctypes
    times = [ctypes.c_ulonglong() for _ in range(3 if process is None else 4)]
    pointers = [ctypes.byref(t) for t in times]
    if process is None:
        if not ctypes.windll.kernel32.GetSystemTimes(*pointers):
            raise ctypes.WinError()
        idle, kernel, user = (t.value for t in times)
        return kernel + user, kernel + user - idle
    if not ctypes.windll.kernel32.GetProcessTimes(ctypes.c_void_p(int(process._handle)), *pointers):
        raise ctypes.WinError()
    return times[2].value + times[3].value


def _windows_affinity(mask):
    import ctypes
    dll = ctypes.windll.kernel32
    handle = ctypes.c_void_p(-1)
    current, system = ctypes.c_size_t(), ctypes.c_size_t()

    def read():
        if not dll.GetProcessAffinityMask(handle, ctypes.byref(current), ctypes.byref(system)):
            raise ctypes.WinError()

    read()
    if mask is not None:
        if not (mask > 0 and mask & current.value == mask):
            raise ValueError("CPU mask exceeds process affinity")
        if not dll.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
            raise ctypes.WinError()
        read()
        if current.value != mask:
            raise ValueError("Requested affinity unavailable")
    return hex(current.value)


def _linux_system_cpu_100ns():
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    ticks = [int(v) for v in fields]
    # user, nice, system, idle, iowait, irq, softirq, steal, ...
    idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
    total = sum(ticks)
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    scale = 1e7 / hz
    return int(total * scale), int((total - idle) * scale)


def _linux_proc_cpu_100ns(pid):
    # utime and stime are fields 14 and 15 (1-based) of /proc/pid/stat.
    text = Path(f"/proc/{pid}/stat").read_text()
    end = text.rfind(")")
    fields = text[end + 2:].split()
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    return int((int(fields[11]) + int(fields[12])) * 1e7 / hz)
