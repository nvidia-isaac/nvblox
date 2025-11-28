#!/usr/bin/env python3
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
"""Nvblox CI script for building Docker images and running tests.

This script provides a command-line interface for building nvblox Docker images
and running various tests within those images.
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Type

from ci_utils import (
    CudaSmArchitectures,
    CudaVersion,
    DockerImage,
    Platform,
    TestBase,
    UbuntuVersion,
    OsImage,
)
from system_info import print_system_info

DEFAULT_MAX_NUM_JOBS = 8


class DependenciesImage(DockerImage):
    """Nvblox dependencies image"""

    def build_args(self) -> List[str]:
        return []

    def image_name_base(self) -> str:
        return 'nvblox_deps'

    def dockerfile_path(self) -> str:
        if self.args.platform == Platform.X86_64:
            return os.path.join('docker', 'Dockerfile.deps')
        else:
            return os.path.join('docker', 'Dockerfile.jetson_deps')

    def parent_image(self) -> OsImage:
        return OsImage(self.args)


class BuildImage(DockerImage):
    """Nvblox build image containing binaries"""

    def image_name_base(self) -> str:
        return 'nvblox_build'

    def dockerfile_path(self) -> str:
        return os.path.join('docker', 'Dockerfile.build')

    def parent_image(self) -> DockerImage:
        return DependenciesImage(self.args)

    def get_native_cuda_sm_architecture(self) -> str:
        """Get the cuda architecture from nvidia-smi"""
        try:
            command_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv'])
            arch = command_output.decode('utf-8').split()[1].replace('.', '')
            return arch
        except FileNotFoundError:
            print('::error :: nvidia-smi not found. Cannot detect native CUDA SM architecture.')
            raise

    def get_cuda_sm_architecture(self) -> str:
        """Get the CUDA SM architecture"""
        if self.args.cuda_arch == CudaSmArchitectures.SM_NATIVE:
            return self.get_native_cuda_sm_architecture()
        else:
            return self.args.cuda_arch.value

    def build_args(self) -> List[str]:
        cuda_arch = self.get_cuda_sm_architecture()

        # Setup args to cmake
        cmake_args = f'-DCMAKE_CUDA_ARCHITECTURES={cuda_arch} -DWARNING_AS_ERROR=1'
        if self.args.gcc_sanitizer == 1:
            cmake_args += ' -DCMAKE_BUILD_TYPE=Debug -DUSE_SANITIZER=yes'

        # Pytorch is not supported on CUDA 13.
        if self.args.cuda_version == CudaVersion.CUDA_13:
            cmake_args += ' -DBUILD_PYTORCH_WRAPPER=0'
            print(
                '::warning :: Pytorch is not yet supported on CUDA 13. Disabling pytorch wrapper.')

        # Setup args to docker build
        args = [f'CMAKE_ARGS={cmake_args}']
        if self.args.max_num_jobs is not None:
            args += [f'MAX_NUM_JOBS={self.args.max_num_jobs}']

        return args


class RealsenseImage(DockerImage):
    """Nvblox realsense example image"""

    def image_name_base(self) -> str:
        return 'nvblox_realsense_example'

    def dockerfile_path(self) -> str:
        return os.path.join('docker', 'Dockerfile.realsense_example')

    def parent_image(self) -> DockerImage:
        return BuildImage(self.args)

    def build_args(self) -> List[str]:
        return []


class DocsImage(DockerImage):
    """Nvblox documentation image. Does not have any internal dependencies."""

    def image_name_base(self) -> str:
        return 'nvblox_docs'

    def dockerfile_path(self) -> str:
        return os.path.join('docker', 'Dockerfile.docs')

    def parent_image(self) -> None:
        return None

    def build_args(self) -> List[str]:
        return []


class CppUnitTests(TestBase):
    """Run the C++ unit tests"""

    def get_command(self) -> str:
        num_jobs = self.args.max_num_jobs
        return (f'ctest -j{num_jobs} -T test '
                f'--no-compress-output')

    def image(self) -> DockerImage:
        return BuildImage(self.args)

    def get_cwd(self) -> str:
        return '/nvblox/build/nvblox/tests'


class PythonUnitTests(TestBase):
    """Run the Python unit tests"""

    def get_command(self) -> str:

        if self.args.cuda_version == CudaVersion.CUDA_13:
            return 'echo ::warning :: Pytorch not supported on CUDA 13. Skipping Python tests.'
        else:
            cmd = '. /opt/venv/bin/activate && '
            cmd += 'pytest --capture=no /opt/venv/lib/*/site-packages/nvblox_torch'
            return cmd

    def image(self) -> DockerImage:
        return BuildImage(self.args)

    def get_cwd(self) -> str:
        return '/nvblox/'


class CudaSanitizer(TestBase):
    """Run the CUDA Sanitizer tests"""

    def get_command(self) -> str:
        return 'bash ci/compute_sanitizer.sh'

    def image(self) -> DockerImage:
        return BuildImage(self.args)

    def get_cwd(self) -> str:
        return '/nvblox/'


class StabilityTest(TestBase):
    """Run the Stability tests"""

    def get_command(self) -> str:
        return 'bash ci/fuser_redwood_apartment.sh'

    def image(self) -> DockerImage:
        return BuildImage(self.args)

    def get_cwd(self) -> str:
        return '/nvblox/'


# Map cmd line arg to image class.
ARG_TO_IMAGE: Dict[str, Type[DockerImage]] = {
    'deps': DependenciesImage,
    'build': BuildImage,
    'realsense': RealsenseImage,
    'docs': DocsImage,
}

# Map cmd line arg to test class.
ARG_TO_TEST: Dict[str, Type[TestBase]] = {
    'cpp': CppUnitTests,
    'python': PythonUnitTests,
    'cuda-sanitizer': CudaSanitizer,
    'stability': StabilityTest,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build nvblox docker images and run tests inside them.')
    parser.add_argument('--build-image',
                        type=str,
                        choices=ARG_TO_IMAGE.keys(),
                        required=False,
                        help='Docker image to build. Will build the image and then exit.')
    parser.add_argument(
        '--build-and-test',
        type=str,
        choices=ARG_TO_TEST.keys(),
        required=False,
        help=
        'Test to run. Will also build the necessary image (no-op if the image is already built).')
    parser.add_argument('--cuda-version',
                        type=CudaVersion,
                        default=CudaVersion.CUDA_12,
                        help='CUDA version')
    parser.add_argument('--cuda-arch',
                        type=CudaSmArchitectures,
                        required=False,
                        default=CudaSmArchitectures.SM_NATIVE,
                        help='CUDA SM architectures.')
    parser.add_argument('--platform',
                        type=Platform,
                        default=Platform.X86_64,
                        help='Platform to build for.')
    parser.add_argument('--ubuntu-version',
                        type=UbuntuVersion,
                        required=False,
                        default=UbuntuVersion.UBUNTU_24,
                        help='Ubuntu version to build for.')
    parser.add_argument('--user-build-args',
                        type=str,
                        required=False,
                        help='Additional user-provided docker build arguments.')
    parser.add_argument('--max-num-jobs',
                        type=int,
                        required=False,
                        default=DEFAULT_MAX_NUM_JOBS,
                        help='Maximum number of jobs to run in parallel (build and ctest).')
    parser.add_argument('--gcc-sanitizer',
                        type=int,
                        default=False,
                        help='Build in debug mode with gcc sanitizers enabled.(1=yes, 0=no)')
    args = parser.parse_args()

    if args.build_image is None and args.build_and_test is None:
        parser.error('Either image or test must be provided')

    return args


def main() -> int:
    args = parse_args()
    print_system_info()

    if args.build_image is not None:
        image = ARG_TO_IMAGE[args.build_image](args)
        image.build()

    if args.build_and_test is not None:
        test = ARG_TO_TEST[args.build_and_test](args)
        test.run()

    return 0


if __name__ == '__main__':
    sys.exit(main())
