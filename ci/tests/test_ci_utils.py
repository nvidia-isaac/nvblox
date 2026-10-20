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
"""CPU-only regression tests for the images selected and validated by CI."""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# pylint: disable=wrong-import-position
import ci_utils
import nvblox_ci


def image_args(cuda: ci_utils.CudaVersion = ci_utils.CudaVersion.CUDA_13,
               ubuntu: ci_utils.UbuntuVersion = ci_utils.UbuntuVersion.UBUNTU_24,
               platform: ci_utils.Platform = ci_utils.Platform.X86_64) -> argparse.Namespace:
    """Supply the same profile fields as the CI command-line parser."""
    return argparse.Namespace(cuda_version=cuda,
                              ubuntu_version=ubuntu,
                              platform=platform,
                              gcc_sanitizer=0,
                              cxx_standard=17,
                              cuda_arch=ci_utils.CudaSmArchitectures.SM_X86_CI_SUPPORTED,
                              max_num_jobs=2,
                              user_build_args=None)


class OsImageSelectionTest(unittest.TestCase):
    """Exercise CI's actual image objects, not the source text of their mappings."""

    def test_cuda13_uses_cuda132_development_images(self) -> None:
        for ubuntu in ci_utils.UbuntuVersion:
            with self.subTest(ubuntu=ubuntu):
                image = ci_utils.OsImage(image_args(ubuntu=ubuntu))
                self.assertEqual(image.image_name(),
                                 f'nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu{ubuntu.value}.04')

    def test_legacy_cuda_images_are_preserved(self) -> None:
        for cuda, ubuntu, version in ((ci_utils.CudaVersion.CUDA_11,
                                       ci_utils.UbuntuVersion.UBUNTU_22, '11.8.0'),
                                      (ci_utils.CudaVersion.CUDA_12,
                                       ci_utils.UbuntuVersion.UBUNTU_22, '12.8.0'),
                                      (ci_utils.CudaVersion.CUDA_12,
                                       ci_utils.UbuntuVersion.UBUNTU_24, '12.8.0')):
            with self.subTest(cuda=cuda, ubuntu=ubuntu):
                image = ci_utils.OsImage(image_args(cuda=cuda, ubuntu=ubuntu))
                self.assertEqual(image.image_name(),
                                 f'nvcr.io/nvidia/cuda:{version}-devel-ubuntu{ubuntu.value}.04')

    def test_unsupported_os_combination_is_rejected(self) -> None:
        image = ci_utils.OsImage(image_args(cuda=ci_utils.CudaVersion.CUDA_11))
        with self.assertRaisesRegex(ValueError, 'No OS image available'):
            image.image_name()

    def test_cuda13_cli_profile_reaches_cuda132_base(self) -> None:
        with patch.object(sys, 'argv', [
                'nvblox_ci.py', '--build-image', 'build', '--cuda-version', '13',
                '--ubuntu-version', '24'
        ]):
            args = nvblox_ci.parse_args()
        build_image = nvblox_ci.BuildImage(args)
        deps_image = build_image.parent_image()
        self.assertIsInstance(deps_image, nvblox_ci.DependenciesImage)
        os_image = deps_image.parent_image()
        self.assertIsNotNone(os_image)
        assert os_image is not None
        self.assertEqual(os_image.image_name(), 'nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04')


class CxxStandardCiTest(unittest.TestCase):
    """Keep the chosen dialect consistent across CI commands and image tags."""

    def parse_args(self, *extra: str) -> argparse.Namespace:
        with patch.object(sys, 'argv', ['nvblox_ci.py', '--build-image', 'build', *extra]):
            return nvblox_ci.parse_args()

    def test_cli_defaults_to_cxx17_and_accepts_explicit_cxx20(self) -> None:
        self.assertEqual(self.parse_args().cxx_standard, 17)
        self.assertEqual(self.parse_args('--cxx-standard', '20').cxx_standard, 20)

    def test_unsupported_dialect_is_rejected(self) -> None:
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            self.parse_args('--cxx-standard', '23')

    def test_cmake_build_receives_selected_dialect(self) -> None:
        for standard in (17, 20):
            with self.subTest(standard=standard):
                args = image_args(cuda=ci_utils.CudaVersion.CUDA_12)
                args.cxx_standard = standard
                build_args = nvblox_ci.BuildImage(args).build_args()
                self.assertIn(f'-DNVBLOX_CXX_STANDARD={standard}', build_args[0].split())

    def test_matrix_jobs_have_distinct_image_tags(self) -> None:
        default = image_args()
        modern = image_args()
        modern.cxx_standard = 20
        for image_type in (nvblox_ci.BuildImage, nvblox_ci.RealsenseImage):
            with self.subTest(image=image_type):
                self.assertEqual(
                    image_type(modern).image_name(),
                    image_type(default).image_name() + '_cxx20')

    def test_modern_bazel_job_tests_both_default_and_opt_in(self) -> None:
        args = image_args()
        default_command = nvblox_ci.BazelTests(args).get_command()
        args.cxx_standard = 20
        modern_command = nvblox_ci.BazelTests(args).get_command()
        self.assertEqual(default_command.split(' && ')[1], 'bazel test -c opt //nvblox/tests/...')
        self.assertEqual(modern_command.split(' && '), [
            'bash ci/install_bazel.sh',
            'bazel test -c opt //nvblox/tests/...',
            'bazel test --config=cxx20 -c opt //nvblox/tests/...',
        ])


class ImageValidationTest(unittest.TestCase):
    """Execute the container's Python version check against fixture torch metadata.

    Only the Docker subprocess and imported torch module are replaced. In particular,
    the assertions sent to the container execute, rather than being string-matched.
    """

    def setUp(self) -> None:
        self.cuda = '13.2'
        self.ubuntu = '24.04'
        self.torch = SimpleNamespace(
            __version__='2.14.1+cu132',
            version=SimpleNamespace(cuda='13.2'),
            cuda=SimpleNamespace(is_available=Mock(side_effect=AssertionError('GPU queried'))))
        self.torch_missing = False
        patcher = patch.object(ci_utils.subprocess, 'run', side_effect=self.run_container_command)
        self.mock_run = patcher.start()
        self.addCleanup(patcher.stop)

    def run_container_command(self, command: list[str], *, check: bool, capture_output: bool,
                              text: bool) -> subprocess.CompletedProcess[str]:
        self.assertTrue(check)
        self.assertTrue(capture_output)
        self.assertTrue(text)
        self.assertEqual(command[:3], ['docker', 'run', '--rm'])
        self.assertNotIn('--gpus', command)
        self.assertNotIn('--privileged', command)
        if command[4:] == ['lsb_release', '-a']:
            stdout = f'Description: Ubuntu {self.ubuntu} LTS\n'
        elif command[4:] == ['nvcc', '--version']:
            stdout = (f'Cuda compilation tools, release {self.cuda}, V{self.cuda}.88\n'
                      f'Build cuda_{self.cuda}.r13.2/compiler.12345_0\n')
        else:
            self.assertEqual(command[4:6], ['/opt/venv/bin/python', '-c'])
            self.assertEqual(len(command), 7)
            with patch.dict(sys.modules, {'torch': None if self.torch_missing else self.torch}), \
                    redirect_stdout(io.StringIO()):
                try:
                    exec(command[6], {})    # pylint: disable=exec-used
                except (AssertionError, ImportError) as error:
                    raise subprocess.CalledProcessError(1, command, stderr=str(error)) from error
            stdout = ''
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    def validate(self, image: ci_utils.DockerImage) -> None:
        with redirect_stdout(io.StringIO()):
            image._validate()    # pylint: disable=protected-access

    def test_cuda132_build_chain_checks_torch_without_a_gpu(self) -> None:
        for ubuntu in ci_utils.UbuntuVersion:
            args = image_args(ubuntu=ubuntu)
            self.ubuntu = f'{ubuntu.value}.04'
            for image in (nvblox_ci.DependenciesImage(args), nvblox_ci.BuildImage(args),
                          nvblox_ci.RealsenseImage(args)):
                with self.subTest(image=type(image).__name__, ubuntu=ubuntu):
                    self.mock_run.reset_mock()
                    self.validate(image)
                    self.assertEqual(self.mock_run.call_count, 3)
                    for call in self.mock_run.call_args_list:
                        self.assertEqual(call.args[0][3], image.image_name())
        self.torch.cuda.is_available.assert_not_called()

    def test_wrong_cuda_minor_is_rejected(self) -> None:
        for cuda in ('13.0', '13.1', '13.20', '13.3', '12.8'):
            with self.subTest(cuda=cuda):
                self.cuda = cuda
                with self.assertRaisesRegex(AssertionError, 'cuda version'):
                    self.validate(nvblox_ci.BuildImage(image_args()))

    def test_malformed_cuda_release_is_rejected(self) -> None:
        self.cuda = '13.not-a-version'
        with self.assertRaisesRegex(AssertionError, 'cuda version'):
            self.validate(nvblox_ci.BuildImage(image_args()))

    def test_wrong_torch_release_is_rejected(self) -> None:
        for version in ('2.14.0+cu132', '2.9.1+cu130', '2.14.10+cu132', '2.14.1rc1+cu132'):
            with self.subTest(version=version):
                self.torch.__version__ = version
                with self.assertRaises(subprocess.CalledProcessError):
                    self.validate(nvblox_ci.BuildImage(image_args()))

    def test_wrong_torch_cuda_build_is_rejected(self) -> None:
        for cuda in ('13.0', '13.20', '12.8', None):
            with self.subTest(cuda=cuda):
                self.torch.version.cuda = cuda
                with self.assertRaises(subprocess.CalledProcessError):
                    self.validate(nvblox_ci.BuildImage(image_args()))

    def test_missing_torch_is_rejected(self) -> None:
        self.torch_missing = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.validate(nvblox_ci.DependenciesImage(image_args()))

    def test_legacy_profiles_do_not_require_new_torch(self) -> None:
        for platform, cuda, torch in ((ci_utils.Platform.X86_64, ci_utils.CudaVersion.CUDA_11,
                                       '2.9.1'), (ci_utils.Platform.X86_64,
                                                  ci_utils.CudaVersion.CUDA_12, '2.9.1'),
                                      (ci_utils.Platform.JETPACK_6, ci_utils.CudaVersion.CUDA_12,
                                       '2.3.0')):
            with self.subTest(platform=platform, cuda=cuda):
                self.cuda = f'{cuda.value}.8'
                self.torch.__version__ = torch
                self.mock_run.reset_mock()
                self.validate(nvblox_ci.DependenciesImage(image_args(cuda=cuda, platform=platform)))
                self.assertEqual(self.mock_run.call_count, 2)

    def test_standalone_docs_image_does_not_require_cuda_or_torch(self) -> None:
        self.validate(nvblox_ci.DocsImage(image_args()))
        self.mock_run.assert_not_called()

    def test_external_os_image_does_not_require_torch(self) -> None:
        self.validate(ci_utils.OsImage(image_args()))
        self.mock_run.assert_not_called()

    def test_wrong_ubuntu_is_rejected(self) -> None:
        self.ubuntu = '22.04'
        with self.assertRaisesRegex(AssertionError, 'ubuntu version'):
            self.validate(nvblox_ci.BuildImage(image_args()))


if __name__ == '__main__':
    unittest.main()
