#!/usr/bin/env python3
"""Fail-closed process launcher for PhysX headless snippet runners."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence


WINDOW_POLL_SECONDS = 0.025
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    )


class _IoCounters(ctypes.Structure):
    _fields_ = (
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    )


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


class _KillOnCloseJob:
    """Own a Windows Job Object that kills assigned children on close."""

    def __init__(self) -> None:
        self._handle: int | None = None
        self._kernel32: ctypes.WinDLL | None = None
        if os.name != "nt":
            return

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = (
            wintypes.HANDLE,
            wintypes.HANDLE,
        )
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        information = _JobObjectExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not kernel32.SetInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise ctypes.WinError(error)
        self._kernel32 = kernel32
        self._handle = int(handle)

    def assign(self, process: subprocess.Popen[str]) -> None:
        if self._handle is None or self._kernel32 is None:
            return
        if self._kernel32.AssignProcessToJobObject(
            self._handle, int(process._handle)  # type: ignore[attr-defined]
        ):
            return
        error = ctypes.get_last_error()
        if process.poll() is None:
            process.kill()
            process.wait()
            raise ctypes.WinError(error)

    def terminate(self) -> bool:
        return bool(
            self._handle is not None
            and self._kernel32 is not None
            and self._kernel32.TerminateJobObject(self._handle, 1)
        )

    def close(self) -> None:
        if self._handle is not None and self._kernel32 is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


@dataclass(frozen=True)
class HeadlessProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    visible_window_detected: bool
    visible_window_titles: tuple[str, ...]


def windows_startup_info() -> subprocess.STARTUPINFO | None:
    if os.name != "nt":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup_info.wShowWindow = subprocess.SW_HIDE
    return startup_info


def windows_creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _visible_windows(process_id: int) -> tuple[str, ...]:
    if os.name != "nt":
        return ()

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )
    titles: list[str] = []

    @enum_callback_type
    def collect_window(window: int, unused: int) -> bool:
        del unused
        owner_process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(owner_process_id))
        if owner_process_id.value != process_id or not user32.IsWindowVisible(window):
            return True
        title_length = user32.GetWindowTextLengthW(window)
        if title_length:
            title_buffer = ctypes.create_unicode_buffer(title_length + 1)
            user32.GetWindowTextW(window, title_buffer, title_length + 1)
            title = title_buffer.value
        else:
            title = f"<untitled HWND=0x{int(window):X}>"
        titles.append(title)
        return True

    if not user32.EnumWindows(collect_window, 0):
        error = ctypes.get_last_error()
        if error:
            raise ctypes.WinError(error)
    return tuple(titles)


def _terminate_process_tree(
    process: subprocess.Popen[str], job: _KillOnCloseJob
) -> None:
    if process.poll() is not None:
        return
    job.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=windows_creation_flags(),
            startupinfo=windows_startup_info(),
            shell=False,
        )
    if process.poll() is None:
        process.kill()


def run_headless_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> HeadlessProcessResult:
    """Run one snippet and terminate fail-closed on timeout or visible UI."""

    job = _KillOnCloseJob()
    process: subprocess.Popen[str] | None = None
    stdout = ""
    stderr = ""
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=windows_creation_flags(),
            startupinfo=windows_startup_info(),
            shell=False,
        )
        job.assign(process)
        deadline = time.monotonic() + timeout_seconds
        timed_out = False
        visible_titles: tuple[str, ...] = ()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _terminate_process_tree(process, job)
                break
            try:
                stdout, stderr = process.communicate(
                    timeout=min(WINDOW_POLL_SECONDS, remaining)
                )
                break
            except subprocess.TimeoutExpired:
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    break
                visible_titles = _visible_windows(process.pid)
                if visible_titles:
                    _terminate_process_tree(process, job)
                    break

        if process.poll() is None:
            try:
                stdout, stderr = process.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process, job)
                stdout, stderr = process.communicate()
        elif not stdout and not stderr:
            stdout, stderr = process.communicate()

        return HeadlessProcessResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            visible_window_detected=bool(visible_titles),
            visible_window_titles=visible_titles,
        )
    except BaseException:
        if process is not None:
            _terminate_process_tree(process, job)
            try:
                process.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        raise
    finally:
        job.close()
