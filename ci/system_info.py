#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os
import shutil
import platform
import subprocess
from typing import List, Optional


def get_native_cuda_sm_architecture() -> str:
    """Get the cuda architecture from nvidia-smi"""
    try:
        command_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv'])
        arch = command_output.decode('utf-8').split()[1].replace('.', '')
        return arch
    except FileNotFoundError:
        print('::error :: nvidia-smi not found. Cannot detect native CUDA SM architecture.')
        raise


def print_system_info() -> None:
    """Print system information"""

    def _border(char: str = '=', width: int = 80) -> str:
        return char * width

    def _print_section(title: str, lines: List[str]) -> None:
        print(_border('='))
        print(f'[ {title} ]')
        print(_border('-'))
        for line in lines:
            print(line)

    def _read_os_release() -> Optional[dict]:
        os_release_path = '/etc/os-release'
        if not os.path.exists(os_release_path):
            return None
        info = {}
        try:
            with open(os_release_path, 'r', encoding='utf-8') as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    info[key] = value
            return info
        except (OSError, IOError):
            return None

    def _human_bytes(num_bytes: int) -> str:
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        size = float(num_bytes)
        unit_idx = 0
        while size >= 1024.0 and unit_idx < len(units) - 1:
            size /= 1024.0
            unit_idx += 1
        # Use at most 2 decimals for readability
        if size >= 100 or unit_idx == 0:
            return f'{int(size)} {units[unit_idx]}'
        return f'{size:.2f} {units[unit_idx]}'

    def _read_meminfo() -> Optional[dict]:
        meminfo_path = '/proc/meminfo'
        if not os.path.exists(meminfo_path):
            return None
        info = {}
        try:
            with open(meminfo_path, 'r', encoding='utf-8') as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if ':' not in line:
                        continue
                    key, value = line.split(':', 1)
                    value = value.strip()
                    # Values are typically like "123456 kB"
                    parts = value.split()
                    if not parts:
                        continue
                    try:
                        amount_kb = int(parts[0])
                        info[key] = amount_kb * 1024    # store in bytes
                    except ValueError:
                        # Non-integer field; ignore
                        pass
            return info
        except (OSError, IOError):
            return None

    def _memory_lines() -> List[str]:
        meminfo = _read_meminfo() or {}
        total = meminfo.get('MemTotal')
        available = meminfo.get('MemAvailable')
        if available is None:
            # Fallback heuristic for older kernels without MemAvailable
            free = meminfo.get('MemFree', 0)
            buffers = meminfo.get('Buffers', 0)
            cached = meminfo.get('Cached', 0)
            available = free + buffers + cached
        used = None
        if total is not None and available is not None:
            used = max(total - available, 0)
        lines: List[str] = []
        if total is not None:
            lines.append(f'Total:      {_human_bytes(total)}')
        if used is not None:
            lines.append(f'Used:       {_human_bytes(used)}')
        if available is not None:
            lines.append(f'Available:  {_human_bytes(available)}')
        if not lines:
            lines.append('Memory information not available.')
        return lines

    def _disk_lines() -> List[str]:
        try:
            usage = shutil.disk_usage('/')
            return [
                'Mount:      /',
                f'Total:      {_human_bytes(usage.total)}',
                f'Used:       {_human_bytes(usage.used)}',
                f'Free:       {_human_bytes(usage.free)}',
            ]
        except OSError as e:
            return [f'Disk information not available: {e}']

    def _nvidia_smi_lines() -> List[str]:
        try:
            result = subprocess.run(['nvidia-smi'],
                                    check=False,
                                    capture_output=True,
                                    text=True,
                                    timeout=10)
            if result.returncode == 0 and result.stdout:
                # Limit very long outputs to keep CI logs readable
                stdout = result.stdout.strip()
                max_chars = 8000
                if len(stdout) > max_chars:
                    truncated = stdout[:max_chars] + '\n... (truncated) ...'
                    return truncated.splitlines()
                return stdout.splitlines()
            # If stderr contains something useful, show it; otherwise, generic message
            err = (result.stderr or '').strip()
            if err:
                return ['nvidia-smi returned non-zero exit code:', err]
            return ['nvidia-smi returned non-zero exit code with no output.']
        except FileNotFoundError:
            return ['nvidia-smi not found on this system.']
        except subprocess.TimeoutExpired:
            return ['nvidia-smi timed out.']
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            return [f'Failed to run nvidia-smi: {e}']

    # System section
    os_info = _read_os_release() or {}
    pretty_name = os_info.get('PRETTY_NAME')
    if not pretty_name:
        name = os_info.get('NAME')
        version = os_info.get('VERSION')
        if name and version:
            pretty_name = f'{name} {version}'
        elif name:
            pretty_name = name
        else:
            pretty_name = 'Unknown'
    system_lines = [
        f'Platform:   {platform.platform()}',
        f'OS:         {pretty_name}',
        f'Kernel:     {platform.release()}',
        f'Machine:    {platform.machine()}',
        f'Num CPUs:   {os.cpu_count()}',
    ]
    _print_section('System Information', system_lines)

    # Memory section
    _print_section('Memory', _memory_lines())

    # Disk section
    _print_section('Disk (root filesystem)', _disk_lines())

    # NVIDIA SMI section
    _print_section('NVIDIA SMI', _nvidia_smi_lines())

    # Ensure it is flushed to the console
    print('', flush=True)
