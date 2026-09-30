"""Check the project's language policy through the actual CMake module."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('cmake'), 'CMake is required for script tests')
class CxxStandardTest(unittest.TestCase):
    """Modern CUDA builds must not leave the standalone core on C++17."""

    def check_standard(self, cuda_version: str, expected: str) -> None:
        """Exercise the real policy, not a copy of its condition in Python."""
        module = ROOT / 'cmake/nvblox_targets.cmake'
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
            script = Path(directory) / 'check.cmake'
            script.write_text(
                f'set(CMAKE_CUDA_COMPILER_VERSION "{cuda_version}")\n'
                f'include("{module.as_posix()}")\n'
                f'if(NOT NVBLOX_CXX_STANDARD STREQUAL "{expected}")\n'
                '  message(FATAL_ERROR "Wrong C++ dialect: ${NVBLOX_CXX_STANDARD}")\n'
                'endif()\n',
                encoding='utf-8')
            result = subprocess.run(['cmake', '-P', str(script)],
                                    capture_output=True,
                                    text=True,
                                    check=False,
                                    timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_modern_cuda_uses_cxx20(self) -> None:
        """Cover the first supported toolkit and the current CUDA profiles."""
        for version in ('12.0', '12.8', '13.0', '13.2.51'):
            with self.subTest(version=version):
                self.check_standard(version, '20')

    def test_cuda11_retains_explicit_legacy_policy(self) -> None:
        """nvcc11 cannot compile C++20; its existing CI path must remain usable."""
        self.check_standard('11.8', '17')


@unittest.skipUnless(
    shutil.which('cmake') and shutil.which('nvcc'),
    'CMake and nvcc are required for compiler tests')
class CxxStandardCompilerTest(unittest.TestCase):
    """Compile both languages with the real compiler; no GPU is needed."""

    def test_host_and_cuda_use_the_required_standard(self) -> None:
        module = ROOT / 'cmake/nvblox_targets.cmake'
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
            source = Path(directory)
            (source / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.22.1)\n'
                'project(standard_probe LANGUAGES CXX CUDA)\n'
                'set(CMAKE_BUILD_TYPE Release)\n'
                f'include("{module.as_posix()}")\n'
                'add_library(probe STATIC host.cpp device.cu)\n'
                'set_nvblox_compiler_options(probe)\n'
                'target_compile_definitions(probe PRIVATE '
                'EXPECTED_STANDARD=${NVBLOX_CXX_STANDARD})\n',
                encoding='utf-8')
            check = ('#if EXPECTED_STANDARD == 20\n'
                     '#include <concepts>\n'
                     'static_assert(__cplusplus >= 202002L);\n'
                     'static_assert(std::integral<int>);\n'
                     '#else\n'
                     'static_assert(__cplusplus >= 201703L);\n'
                     '#endif\n')
            (source / 'host.cpp').write_text(check, encoding='utf-8')
            (source / 'device.cu').write_text(check + '__global__ void probe_kernel() {}\n',
                                              encoding='utf-8')
            build = source / 'build'
            for command in (
                ['cmake', '-S',
                 str(source), '-B',
                 str(build), '-DCMAKE_CUDA_ARCHITECTURES=75'],
                ['cmake', '--build', str(build), '--parallel', '2', '--verbose'],
            ):
                result = subprocess.run(command,
                                        capture_output=True,
                                        text=True,
                                        check=False,
                                        timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
