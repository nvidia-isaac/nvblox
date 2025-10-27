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
from system_info import get_native_cuda_sm_architecture, print_system_info


class DependenciesImage(DockerImage):
    """Nvblox dependencies (deps) image"""

    def build_args(self) -> List[str]:
        return []

    def image_name_root(self) -> str:
        return 'nvblox_deps'

    def dockerfile_path(self) -> str:
        """Deps are different for x86 and Jetson platforms."""
        if self.args.platform == Platform.X86_64:
            return os.path.join('docker', 'Dockerfile.deps')
        else:
            return os.path.join('docker', 'Dockerfile.jetson_deps')

    def parent_image(self) -> OsImage:
        return OsImage(self.args)


class BuildImage(DockerImage):
    """Nvblox build image containing compiled binaries and installed python modules."""

    def image_name_root(self) -> str:
        return 'nvblox_build'

    def dockerfile_path(self) -> str:
        return os.path.join('docker', 'Dockerfile.build')

    def parent_image(self) -> DockerImage:
        return DependenciesImage(self.args)

    def get_cuda_sm_architecture(self) -> str:
        """Get the CUDA SM architectures to build for, either from args or detect native"""
        if self.args.cuda_arch == CudaSmArchitectures.SM_NATIVE:
            return get_native_cuda_sm_architecture()
        else:
            return self.args.cuda_arch.value

    def build_args(self) -> List[str]:
        cuda_arch = self.get_cuda_sm_architecture()

        # Setup args to cmake
        cmake_args = f'-DCMAKE_CUDA_ARCHITECTURES={cuda_arch} -DWARNING_AS_ERROR=1'
        if self.args.gcc_sanitizer == 1:
            cmake_args += ' -DCMAKE_BUILD_TYPE=Debug -DUSE_SANITIZER=yes'

        # Setup args to docker build
        args = [f'CMAKE_ARGS={cmake_args}']
        if self.args.max_num_jobs is not None:
            args += [f'MAX_NUM_JOBS={self.args.max_num_jobs}']

        return args


class RealsenseImage(DockerImage):
    """Nvblox image for running the Realsense example test.

    Includes specialized dependencies on top of the build image."""

    def image_name_root(self) -> str:
        return 'nvblox_realsense_example'

    def dockerfile_path(self) -> str:
        return os.path.join('docker', 'Dockerfile.realsense_example')

    def parent_image(self) -> DockerImage:
        return BuildImage(self.args)

    def build_args(self) -> List[str]:
        return []


class DocsImage(DockerImage):
    """Nvblox documentation builder image."""

    def image_name_root(self) -> str:
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
        base_cmd = f'ctest -j{num_jobs} -T test ' f'--no-compress-output'

        # When running tests with gcc sanitizers, we need to disable address space
        # randomization due to bug in libgcc that appears on certain platforms.
        # https://stackoverflow.com/questions/77894856/possible-bug-in-gcc-sanitizers
        if self.args.gcc_sanitizer == 1:
            return f'setarch $(uname -m) --addr-no-randomize {base_cmd}'
        return base_cmd

    def image(self) -> DockerImage:
        return BuildImage(self.args)

    def get_cwd(self) -> str:
        return '/nvblox/build/nvblox/tests'


class PythonUnitTests(TestBase):
    """Run the Python unit tests"""

    def get_command(self) -> str:
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


class RealsenseTest(TestBase):
    """Run the Realsense tests"""

    def get_command(self) -> str:

        # The realsense example test is disabled per default since
        # it requires a dedicated docker image. Here we pass --runxfail to
        # pytest in order to enable it.
        cmd = '. /opt/venv/bin/activate && '
        cmd += 'pytest --runxfail test_realsense_example.py --capture=no'
        return cmd

    def get_cwd(self) -> str:
        return '/nvblox/nvblox_torch/internal_tests/'

    def image(self) -> DockerImage:
        return RealsenseImage(self.args)


# Map cmd line docker image arg to image class.
ARG_TO_IMAGE: Dict[str, Type[DockerImage]] = {
    'deps': DependenciesImage,
    'build': BuildImage,
    'realsense': RealsenseImage,
    'docs': DocsImage,
}

# Map cmd line test arg to test class.
ARG_TO_TEST: Dict[str, Type[TestBase]] = {
    'cpp': CppUnitTests,
    'python': PythonUnitTests,
    'cuda-sanitizer': CudaSanitizer,
    'stability': StabilityTest,
    'realsense': RealsenseTest,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build nvblox docker images and/or run tests inside them.')
    parser.add_argument(
        '--build-image',
        type=str,
        choices=ARG_TO_IMAGE.keys(),
        required=False,
        help='Docker image to build. Will build the image and then exit.',
    )
    parser.add_argument(
        '--build-and-test',
        type=str,
        choices=ARG_TO_TEST.keys(),
        required=False,
        help=
        'Test to run. Will also build the necessary image (no-op if the image is already built).',
    )
    parser.add_argument(
        '--cuda-version',
        type=CudaVersion,
        default=CudaVersion.CUDA_12,
        help='CUDA version',
    )
    parser.add_argument(
        '--cuda-arch',
        type=CudaSmArchitectures,
        required=False,
        default=CudaSmArchitectures.SM_NATIVE,
        help='CUDA SM architecture.',
    )
    parser.add_argument(
        '--platform',
        type=Platform,
        default=Platform.X86_64,
        help='Platform to build for.',
    )
    parser.add_argument(
        '--ubuntu-version',
        type=UbuntuVersion,
        required=False,
        default=UbuntuVersion.UBUNTU_24,
        help='Ubuntu version to build for.',
    )
    parser.add_argument(
        '--user-build-args',
        type=str,
        required=False,
        help='Additional user-provided docker build arguments.',
    )
    parser.add_argument(
        '--max-num-jobs',
        type=int,
        required=False,
        default=8,
        help='Maximum number of jobs to run in parallel (build and ctest).',
    )
    parser.add_argument(
        '--gcc-sanitizer',
        type=int,
        default=False,
        help='Build in debug mode with gcc sanitizers enabled. (1=yes, 0=no)',
    )
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
