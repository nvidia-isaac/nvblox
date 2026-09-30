#
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""CPU-only regression tests for CUDA-aware package installation."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import shlex
from typing import Callable
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'docker'))
# pylint: disable=wrong-import-position
import torch_install_helper as helper


class CudaVersionTest(unittest.TestCase):
    """Parse nvcc output without invoking a CUDA installation."""

    def test_cuda_version_preserves_minor(self) -> None:
        for version in ('12.8', '13.0', '13.2', '13.20'):
            with self.subTest(version=version):
                output = f'Cuda compilation tools, release {version}, V{version}.88\n'
                result = subprocess.CompletedProcess(['nvcc', '--version'], 0, stdout=output)
                with patch.object(helper.subprocess, 'run', return_value=result) as run:
                    self.assertEqual(helper.get_cuda_version(), version)
                run.assert_called_once_with(['nvcc', '--version'],
                                            capture_output=True,
                                            text=True,
                                            check=True)

    def test_missing_release_is_rejected(self) -> None:
        result = subprocess.CompletedProcess(['nvcc', '--version'], 0, stdout='not nvcc')
        with patch.object(helper.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(ValueError, 'Failed to find cuda version'):
                helper.get_cuda_version()


class TorchSelectionTest(unittest.TestCase):
    """Select the exact CUDA minor before falling back to legacy major mappings."""

    def select(self, machine: str, cuda: str) -> helper.TorchVersion | None:
        result = subprocess.CompletedProcess(['nvcc', '--version'],
                                             0,
                                             stdout=f'release {cuda}, V{cuda}.88')
        with patch.object(helper.platform, 'machine', return_value=machine), \
                patch.object(helper.subprocess, 'run', return_value=result), \
                redirect_stdout(io.StringIO()):
            return helper.get_pytorch_version_for_this_machine()

    def test_cuda132_prefers_exact_mapping_over_major(self) -> None:
        selected = self.select('x86_64', '13.2')
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.cuda_version, '13.2')
        self.assertEqual(selected.pytorch_version, '2.14.1')
        self.assertEqual(selected.torchvision_version, '0.29.1')
        self.assertEqual(selected.index_url, 'https://download.pytorch.org/whl/cu132')

    def test_legacy_x86_major_mappings_are_preserved(self) -> None:
        for cuda, index in (('12.0', 'cu128'), ('12.8', 'cu128'), ('12.9', 'cu128'),
                            ('13.0', 'cu130'), ('13.1', 'cu130'), ('13.20', 'cu130')):
            with self.subTest(cuda=cuda):
                selected = self.select('x86_64', cuda)
                self.assertIsNotNone(selected)
                assert selected is not None
                self.assertEqual(selected.pytorch_version, '2.9.1')
                self.assertEqual(selected.torchvision_version, '0.24.1')
                self.assertEqual(selected.index_url, f'https://download.pytorch.org/whl/{index}')

    def test_legacy_aarch64_cuda12_mapping_is_preserved(self) -> None:
        selected = self.select('aarch64', '12.6')
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.pytorch_version, '2.3.0')
        self.assertEqual(selected.torchvision_version, '0.18.0')
        self.assertIsNone(selected.index_url)
        self.assertEqual(selected.pytorch_filename, 'torch-2.3.0-cp310-cp310-linux_aarch64.whl')
        self.assertEqual(selected.torchvision_filename,
                         'torchvision-0.18.0-cp310-cp310-linux_aarch64.whl')

    def test_unsupported_platforms_return_none(self) -> None:
        for machine, cuda in (('x86_64', '11.8'), ('x86_64', '14.0'), ('aarch64', '13.2'),
                              ('ppc64le', '13.2')):
            with self.subTest(machine=machine, cuda=cuda):
                self.assertIsNone(self.select(machine, cuda))


class TorchInstallationTest(unittest.TestCase):
    """Inspect real install commands while preventing subprocess side effects."""

    def install_commands(self, installer: Callable[[], None], machine: str,
                         cuda: str) -> list[list[str]]:
        result = subprocess.CompletedProcess(['nvcc', '--version'],
                                             0,
                                             stdout=f'release {cuda}, V{cuda}.88')
        with patch.object(helper.platform, 'machine', return_value=machine), \
                patch.object(helper.subprocess, 'run', return_value=result) as run, \
                redirect_stdout(io.StringIO()):
            installer()
        commands: list[list[str]] = []
        for call in run.call_args_list:
            command = call.args[0]
            if isinstance(command, str):
                self.assertTrue(call.kwargs['shell'])
                self.assertTrue(call.kwargs['check'])
                commands.extend(
                    shlex.split(line, comments=True) for line in command.splitlines()
                    if line.strip())
        return commands

    def test_x86_torch_install_pins_versions_and_index(self) -> None:
        for cuda, torch, vision, index in (('13.2', '2.14.1', '0.29.1',
                                            'cu132'), ('13.0', '2.9.1', '0.24.1', 'cu130'),
                                           ('12.8', '2.9.1', '0.24.1', 'cu128')):
            with self.subTest(cuda=cuda):
                commands = self.install_commands(
                    helper.install_pytorch_if_supported_for_this_machine, 'x86_64', cuda)
                installs = [
                    cmd for cmd in commands if cmd[:2] == ['pip', 'install'] and any(
                        arg.startswith('torch==') for arg in cmd)
                ]
                self.assertEqual(len(installs), 1)
                command = installs[0]
                self.assertIn(f'torch=={torch}', command)
                self.assertIn(f'torchvision=={vision}', command)
                self.assertEqual(command[command.index('--index-url') + 1],
                                 f'https://download.pytorch.org/whl/{index}')

    def test_disabled_wrapper_is_not_packaged(self) -> None:
        """Sanitizer builds intentionally have no native wrapper or version record."""
        for value in ('0', 'OFF', 'false', 'NO', 'N', 'IGNORE', 'NOTFOUND', 'WRAPPER-NOTFOUND', ''):
            with self.subTest(value=value):
                cache = f'BUILD_PYTORCH_WRAPPER:BOOL={value}\n'
                with patch.object(Path, 'read_text', return_value=cache):
                    commands = self.install_commands(
                        helper.install_nvblox_torch_if_supported_for_this_machine, 'x86_64', '12.8')
                self.assertEqual(commands, [])

    def test_enabled_or_unspecified_wrapper_keeps_packaging(self) -> None:
        """Missing build records must remain errors for ordinary wrapper builds."""
        for cache in ('BUILD_PYTORCH_WRAPPER:BOOL=ON\n', 'BUILD_PYTORCH_WRAPPER:BOOL=1\n',
                      '# No wrapper entry\n'):
            with self.subTest(cache=cache):
                with patch.object(Path, 'read_text', return_value=cache):
                    commands = self.install_commands(
                        helper.install_nvblox_torch_if_supported_for_this_machine, 'x86_64', '13.2')
                self.assertTrue(any('/nvblox/nvblox_torch/' in cmd for cmd in commands))

    def test_missing_cache_does_not_skip_packaging(self) -> None:
        """An absent CMake cache is not an explicit wrapper-disabled configuration."""
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
            commands = self.install_commands(
                helper.install_nvblox_torch_if_supported_for_this_machine, 'x86_64', '13.2')
        self.assertTrue(any('/nvblox/nvblox_torch/' in cmd for cmd in commands))

    def test_unreadable_cache_is_not_silently_ignored(self) -> None:
        """A permission error must not be interpreted as a disabled wrapper."""
        with patch.object(Path, 'read_text', side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                self.install_commands(helper.install_nvblox_torch_if_supported_for_this_machine,
                                      'x86_64', '13.2')

    def test_nvblox_install_retains_selected_torch_pin(self) -> None:
        for machine, cuda, torch in (('x86_64', '13.2', '2.14.1'), ('x86_64', '13.0', '2.9.1'),
                                     ('x86_64', '12.8', '2.9.1'), ('aarch64', '12.6', '2.3.0')):
            with self.subTest(machine=machine, cuda=cuda):
                commands = self.install_commands(
                    helper.install_nvblox_torch_if_supported_for_this_machine, machine, cuda)
                installs = [cmd for cmd in commands if '/nvblox/nvblox_torch/' in cmd]
                self.assertEqual(len(installs), 1)
                self.assertEqual(installs[0][:2], ['pip', 'install'])
                self.assertIn(f'torch=={torch}', installs[0])

    def test_aarch64_downloads_renamed_wheels(self) -> None:
        commands = self.install_commands(helper.install_pytorch_if_supported_for_this_machine,
                                         'aarch64', '12.6')
        wheels = (
            ('https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl',
             'torch-2.3.0-cp310-cp310-linux_aarch64.whl'),
            ('https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl',
             'torchvision-0.18.0-cp310-cp310-linux_aarch64.whl'),
        )
        for url, filename in wheels:
            self.assertIn(['wget', url, '-O', filename], commands)
            self.assertIn(['pip', 'install', '--no-cache-dir', filename], commands)

    def test_unsupported_platforms_do_not_install(self) -> None:
        for installer in (helper.install_pytorch_if_supported_for_this_machine,
                          helper.install_nvblox_torch_if_supported_for_this_machine):
            for machine, cuda in (('aarch64', '13.2'), ('x86_64', '11.8')):
                with self.subTest(installer=installer.__name__, machine=machine, cuda=cuda):
                    self.assertEqual(self.install_commands(installer, machine, cuda), [])


if __name__ == '__main__':
    unittest.main()
