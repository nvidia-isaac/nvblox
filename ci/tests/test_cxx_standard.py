"""Check language selection and downstream compatibility using the CMake module."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'cmake/nvblox_targets.cmake'


@unittest.skipUnless(shutil.which('cmake'), 'CMake is required for script tests')
class CxxStandardTest(unittest.TestCase):
    """C++17 remains the default; C++20 requires an explicit choice."""

    def check_standard(self, cuda_version: str, expected: str, selected: str = '') -> None:
        """Exercise the real policy, not a copy of its condition in Python."""
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
            script = Path(directory) / 'check.cmake'
            script.write_text(
                f'set(CMAKE_CUDA_COMPILER_VERSION "{cuda_version}")\n'
                f'include("{MODULE.as_posix()}")\n'
                f'if(NOT NVBLOX_CXX_STANDARD STREQUAL "{expected}")\n'
                '  message(FATAL_ERROR "Wrong C++ dialect: ${NVBLOX_CXX_STANDARD}")\n'
                'endif()\n',
                encoding='utf-8')
            command = ['cmake']
            if selected:
                command.append(f'-DNVBLOX_CXX_STANDARD={selected}')
            result = subprocess.run(command + ['-P', str(script)],
                                    capture_output=True,
                                    text=True,
                                    check=False,
                                    timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_cuda_profiles_default_to_cxx17(self) -> None:
        for version in ('11.4', '11.8', '12.0', '12.8', '13.0', '13.2.51'):
            with self.subTest(version=version):
                self.check_standard(version, '17')

    def test_modern_cuda_supports_opt_in_cxx20(self) -> None:
        for version in ('12.0', '12.8', '13.0', '13.2.51'):
            with self.subTest(version=version):
                self.check_standard(version, '20', selected='20')


@unittest.skipUnless(
    shutil.which('cmake') and shutil.which('c++'),
    'CMake and a C++ compiler are required for consumer tests')
class CxxStandardConsumerTest(unittest.TestCase):
    """Build a parent-project consumer in the same layout as Isaac ROS."""

    def test_parent_helpers_and_linked_consumers_inherit_the_selected_standard(self) -> None:
        for standard in (17, 20):
            with self.subTest(standard=standard), tempfile.TemporaryDirectory(
                    dir=os.environ.get('TMPDIR')) as directory:
                source = Path(directory)
                core = source / 'core'
                core.mkdir()
                (core / 'CMakeLists.txt').write_text(
                    f'include("{MODULE.as_posix()}")\n'
                    'add_library(core STATIC source.cpp)\n'
                    'set_nvblox_compiler_options(core)\n',
                    encoding='utf-8')
                (source / 'CMakeLists.txt').write_text(
                    'cmake_minimum_required(VERSION 3.22.1)\n'
                    'project(consumer_probe LANGUAGES CXX)\n'
                    'set(CMAKE_BUILD_TYPE Release)\n'
                    'add_subdirectory(core)\n'
                    f'if(NOT NVBLOX_CXX_STANDARD EQUAL {standard})\n'
                    '  message(FATAL_ERROR "Standard unavailable in parent scope")\n'
                    'endif()\n'
                    'add_library(parent STATIC source.cpp)\n'
                    'set_nvblox_compiler_options(parent)\n'
                    'add_library(consumer STATIC source.cpp)\n'
                    'set_property(TARGET consumer PROPERTY CXX_STANDARD 11)\n'
                    'target_link_libraries(consumer PRIVATE core)\n',
                    encoding='utf-8')
                check = ('#include <type_traits>\n'
                         'static_assert(std::is_same_v<int, int>);\n')
                if standard == 20:
                    check += '#include <concepts>\nstatic_assert(std::integral<int>);\n'
                (source / 'source.cpp').write_text(check, encoding='utf-8')
                (core / 'source.cpp').write_text(check, encoding='utf-8')
                configure = ['cmake', '-S', str(source), '-B', str(source / 'build')]
                if standard == 20:
                    configure.append('-DNVBLOX_CXX_STANDARD=20')
                for command in (configure, ['cmake', '--build', str(source / 'build')]):
                    result = subprocess.run(command,
                                            capture_output=True,
                                            text=True,
                                            check=False,
                                            timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_torch_requirement_does_not_silently_change_the_project_standard(self) -> None:
        """Configure the wrapper with a small imported Torch target, without CUDA."""
        for standard, torch_standard, feature_standard in ((17, 17, 0), (17, 20, 0), (20, 17, 0),
                                                           (20, 20, 0), (17, 0, 20), (20, 0, 20),
                                                           (17, 17, 20), (20, 17, 20)):
            with self.subTest(standard=standard, torch_standard=torch_standard,
                              feature_standard=feature_standard), \
                    tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
                source = Path(directory)
                config = ('set(Torch_VERSION 2.14.1)\n'
                          'add_library(torch INTERFACE IMPORTED)\n')
                if torch_standard:
                    config += ('set_property(TARGET torch PROPERTY CXX_STANDARD '
                               f'{torch_standard})\n')
                if feature_standard:
                    config += ('set_property(TARGET torch PROPERTY INTERFACE_COMPILE_FEATURES '
                               f'cxx_std_{feature_standard} cxx_std_17)\n')
                (source / 'TorchConfig.cmake').write_text(config, encoding='utf-8')
                (source / 'source.cpp').write_text('int probe() { return 0; }\n', encoding='utf-8')
                (source / 'CMakeLists.txt').write_text(
                    'cmake_minimum_required(VERSION 3.22.1)\n'
                    'project(torch_probe LANGUAGES CXX)\n'
                    'set(CMAKE_BUILD_TYPE Release)\n'
                    'set(BUILD_TESTING OFF)\n'
                    'set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}")\n'
                    f'include("{MODULE.as_posix()}")\n'
                    'function(add_nvblox_shared_library target_name)\n'
                    '  add_library(${target_name} STATIC "${PROJECT_SOURCE_DIR}/source.cpp")\n'
                    '  set_nvblox_compiler_options(${target_name})\n'
                    'endfunction()\n'
                    f'add_subdirectory("{ROOT.as_posix()}/nvblox_torch/cpp" wrapper)\n',
                    encoding='utf-8')
                result = subprocess.run([
                    'cmake', '-S',
                    str(source), '-B',
                    str(source / 'build'), f'-DNVBLOX_CXX_STANDARD={standard}'
                ],
                                        capture_output=True,
                                        text=True,
                                        check=False,
                                        timeout=30)
                if max(torch_standard, feature_standard) > standard:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Torch requires C++20', result.stderr)
                    self.assertIn('-DNVBLOX_CXX_STANDARD=20', result.stderr)
                    self.assertIn('-DBUILD_PYTORCH_WRAPPER=OFF', result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(
    shutil.which('cmake') and shutil.which('nvcc'),
    'CMake and nvcc are required for compiler tests')
class CxxStandardCompilerTest(unittest.TestCase):
    """Compile both languages with the real compiler; no GPU is needed."""

    def check_host_and_cuda_standard(self, standard: int) -> None:
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
            source = Path(directory)
            (source / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.22.1)\n'
                'project(standard_probe LANGUAGES CXX CUDA)\n'
                'set(CMAKE_BUILD_TYPE Release)\n'
                f'include("{MODULE.as_posix()}")\n'
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
            configure = [
                'cmake', '-S',
                str(source), '-B',
                str(build), '-DCMAKE_CUDA_ARCHITECTURES=75'
            ]
            if standard == 20:
                configure.append('-DNVBLOX_CXX_STANDARD=20')
            for command in (
                    configure,
                ['cmake', '--build', str(build), '--parallel', '2', '--verbose'],
            ):
                result = subprocess.run(command,
                                        capture_output=True,
                                        text=True,
                                        check=False,
                                        timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_host_and_cuda_default_to_cxx17(self) -> None:
        self.check_host_and_cuda_standard(17)

    def test_host_and_cuda_support_opt_in_cxx20(self) -> None:
        version = subprocess.run(['nvcc', '--version'],
                                 capture_output=True,
                                 text=True,
                                 check=True,
                                 timeout=30).stdout
        if 'release 11.' in version:
            self.skipTest('CUDA 11 does not support C++20')
        self.check_host_and_cuda_standard(20)


if __name__ == '__main__':
    unittest.main()
