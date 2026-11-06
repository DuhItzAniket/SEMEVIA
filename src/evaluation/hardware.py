"""Describe the machine a timing was measured on, for report labels (never hard-coded)."""
from __future__ import annotations

import os
import platform


def cpu_name() -> str:
    """Marketing name of the CPU, e.g. 'Intel(R) Core(TM) i7-14650HX'; falls back to platform info."""
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def hardware_label(threads: int | None = None) -> str:
    """'<cpu name> CPU, <n> torch threads (<logical> logical cores)' for latency reports."""
    parts = [f"{cpu_name()} CPU"]
    if threads is not None:
        parts.append(f"{threads} torch threads")
    parts.append(f"{os.cpu_count()} logical cores")
    return ", ".join(parts)
