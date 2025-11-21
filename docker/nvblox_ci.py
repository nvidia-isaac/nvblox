#!/usr/bin/env python3
"""Nvblox CI script for building Docker images and running tests.

This script provides a command-line interface for building nvblox Docker images
and running various tests within those images.
"""

import argparse
import os
import subprocess
import sys
from enum import Enum
from typing import List

from docker_base import (
    CudaSmArchitectures,
    CudaVersion,
    DockerImage,
    Platform,
    TestBase,
    UbuntuVersion,
    OsImage,
)


class NvbloxImage(Enum):
    DEPS = 'deps'
    BUILD = 'build'
    REALSENSE = 'realsense'
    DOCS = 'docs'
    LINT = 'lint'


class NvbloxTests(Enum):
    CPP = 'cpp'
    PYTHON = 'python'
    LINT = 'lint'
    DOCS = 'docs'
    REALSENSE = 'realsense'


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
            print('ERROR:nvidia-smi not found. Cannot detect native CUDA SM architecture.')
            raise

    def get_cuda_sm_architecture(self) -> str:
        """Get the CUDA SM architecture"""
        if self.args.cuda_arch == CudaSmArchitectures.SM_NATIVE:
            return self.get_native_cuda_sm_architecture()
        else:
            return self.args.cuda_arch.value

    def build_args(self) -> List[str]:
        cuda_arch = self.get_cuda_sm_architecture()
        cmake_args = f'-DCMAKE_VERBOSE_MAKEFILE=1 -DCMAKE_CUDA_ARCHITECTURES={cuda_arch}'
        if self.args.build_debug:
            cmake_args += ' -DCMAKE_BUILD_TYPE=Debug -DUSE_SANITIZER=yes'

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


class LintImage(DockerImage):
    """Nvblox lint image. Does not have any internal dependencies."""

    def image_name_base(self) -> str:
        return 'nvblox_lint'

    def dockerfile_path(self) -> str:
        return os.path.join('docker', 'Dockerfile.lint')

    def parent_image(self) -> None:
        return None

    def build_args(self) -> List[str]:
        return []

    def do_validate_image(self) -> bool:
        """Lint image is not based on ubunut/cuda so cannot be validated."""
        return False


class CppUnitTests(TestBase):
    """Run the C++ unit tests"""

    def get_command(self) -> str:
        num_jobs = self.args.max_num_jobs
        return (f'ctest -j{num_jobs} --verbose -T test '
                f'--no-compress-output')

    def image(self) -> DockerImage:
        return BuildImage(self.args)

    def get_cwd(self) -> str:
        return '/nvblox/build/nvblox/tests'


class PytorchUnitTests(TestBase):
    """Run the Pytorch unit tests"""

    def get_command(self) -> str:
        return 'pytest --capture=no'

    def image(self) -> DockerImage:
        return BuildImage(self.args)

    def get_cwd(self) -> str:
        return '/nvblox/'


class LintTests(TestBase):
    """Run the Lint tests"""

    def get_command(self) -> str:
        return 'bash -c \"ci/lint_nvblox_h.sh && pre-commit run --all-files\"'

    def image(self) -> DockerImage:
        return LintImage(self.args)

    def get_cwd(self) -> str:
        return '/nvblox/'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build nvblox docker images and run tests')
    parser.add_argument('--cuda-version',
                        type=CudaVersion,
                        default=CudaVersion.CUDA_12,
                        help='CUDA version')
    parser.add_argument('--cuda-arch',
                        type=CudaSmArchitectures,
                        required=False,
                        default=CudaSmArchitectures.SM_NATIVE,
                        help='CUDA SM architectures.')
    parser.add_argument('--image',
                        type=NvbloxImage,
                        required=False,
                        help='Docker image to build. Choices: deps, binaries, realsense')
    parser.add_argument('--test',
                        type=NvbloxTests,
                        required=False,
                        help='Test to run. Choices: cpp, python, lint, docs, realsense')
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
                        default=8,
                        help='Maximum number of jobs to run in parallel (build and ctest).')
    parser.add_argument('--build-debug',
                        action='store_true',
                        required=False,
                        help='Build in debug mode.')
    args = parser.parse_args()

    if args.image is None and args.test is None:
        parser.error('Either image or test must be provided')

    return args


def main() -> int:
    args = parse_args()
    if args.image == NvbloxImage.DEPS:
        DependenciesImage(args=args).build()
    elif args.image == NvbloxImage.BUILD:
        BuildImage(args).build()
    elif args.image == NvbloxImage.REALSENSE:
        RealsenseImage(args).build()
    elif args.image == NvbloxImage.DOCS:
        DocsImage(args).build()
    elif args.image == NvbloxImage.LINT:
        LintImage(args).build()
    elif args.test == NvbloxTests.CPP:
        CppUnitTests(args).run()
    elif args.test == NvbloxTests.PYTHON:
        PytorchUnitTests(args).run()
    elif args.test == NvbloxTests.LINT:
        LintTests(args).run()
    else:
        print(f'Invalid test: {args.test}')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
