"""Regression tests for the CUDA-dependent CMake minimum."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('cmake'), 'CMake is required for policy tests')
class CmakeVersionTest(unittest.TestCase):
    """Exercise the real policy for both legacy and modern toolchains."""

    def configure_policy(self, cmake_version: str,
                         cuda_version: str) -> 'subprocess.CompletedProcess[str]':
        """Simulate version inputs only; compiler coverage lives in test_cxx_standard."""
        module = ROOT / 'cmake/nvblox_targets.cmake'
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
            script = Path(directory) / 'check.cmake'
            script.write_text(
                f'set(CMAKE_VERSION "{cmake_version}")\n'
                f'set(CMAKE_CUDA_COMPILER_VERSION "{cuda_version}")\n'
                f'include("{module.as_posix()}")\n',
                encoding='utf-8')
            return subprocess.run(['cmake', '-P', str(script)],
                                  capture_output=True,
                                  text=True,
                                  check=False,
                                  timeout=30)

    def test_modern_cuda_rejects_cmake_without_nvcc_cxx20_support(self) -> None:
        for version in ('3.22.1', '3.24.4', '3.25.0', '3.25.1'):
            with self.subTest(version=version):
                result = self.configure_policy(version, '13.2.51')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('CMake 3.25.2 or newer', result.stderr)
                self.assertIn('C++20', result.stderr)

    def test_modern_cuda_accepts_first_supported_cmake(self) -> None:
        for cuda_version in ('12.0', '12.8', '13.0', '13.2.51'):
            with self.subTest(cuda_version=cuda_version):
                result = self.configure_policy('3.25.2', cuda_version)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cuda11_keeps_cmake_3_22_1_compatibility(self) -> None:
        result = self.configure_policy('3.22.1', '11.4')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
