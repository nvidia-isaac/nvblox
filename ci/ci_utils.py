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
"""Base classes and abstractions for Docker image management.

This module provides abstract base classes for building and testing Docker images,
along with common enumerations used across different CI systems.
"""

import argparse
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple
import re


class Platform(Enum):
    X86_64 = 'x86_64'
    JETPACK_5 = 'jetpack5'
    JETPACK_6 = 'jetpack6'


class CudaVersion(Enum):
    CUDA_11 = '11'
    CUDA_12 = '12'
    CUDA_13 = '13'


class UbuntuVersion(Enum):
    UBUNTU_22 = '22'
    UBUNTU_24 = '24'


class CudaSmArchitectures(Enum):
    SM_X86_CI_SUPPORTED = '120;100;90;89;86;80;75'
    SM_JETPACK_ORIN = '87'
    SM_NATIVE = 'native'


MAX_CONSECUTIVE_IDENTICAL_LOG_LINES = 100


def _try_parse_gcc_output_line(line: str) -> Tuple[Optional[str], Optional[int]]:
    """Try to extract file path and line number from gcc/clang/cmake-style output"""
    # Example: /path/to/file.cpp:LINE:
    match = re.search(r'([^\s:]+):(\d+)(?::\d+)?[ :]', line)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def _try_parse_nvcc_output_line(line: str) -> Tuple[Optional[str], Optional[int]]:
    """Try to extract file path and line number from nvcc output"""
    # Example: /path/to/file.cu(LINE):
    match = re.search(r'([^\s:()]+)\((\d+)\)[ :]', line)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def _maybe_print_github_annotation(line: str) -> None:
    """Print a line as a GitHub Actions annotation if a warning/error keyword is found."""

    error_keywords = [
    # gcc/nvcc
        'error:',
        'fatal error:',
    # cmake
        'cmake error',
        'cmake fatal error',
    ]
    warning_keywords = [
    # gcc
        'warning:',
    # nvcc
        'warning #',
    # Cmake
        'cmake warning',
        'cmake deprecation warning',
    # various
        'permission denied',
    ]

    title = None
    if any(keyword in line.lower() for keyword in error_keywords):
        title = '::error'
    if any(keyword in line.lower() for keyword in warning_keywords):
        title = '::warning'

    # If warning or error found, print as GitHub Actions annotation.
    if title is not None:
        # Try to extract file path and line number from output.
        file_path, line_number = _try_parse_gcc_output_line(line)
        if file_path is None or line_number is None:
            file_path, line_number = _try_parse_nvcc_output_line(line)

        if file_path is None or line_number is None:
            print(f'{title} ::{line.strip()}')
        else:
            # Make file path relative to nvblox root.
            file_path = file_path.replace('/nvblox/', '')
            print(f'{title} file={file_path},line={line_number} ::{line.strip()}')


def _run_and_parse_log(cmd: List[str]) -> None:
    """Run a command and parse its log.
    - The log is printed to console unmodified.
    - Errors and warnings are captured and annotated for GitHub Actions.
    - If there are too many identical lines in a row, the output is truncated.
    """
    # Run the subprocess and redirect stderr to stdout
    with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
    ) as process:

        # Count successive identical lines.
        num_identical = 0
        last_line = None

        # Parse output line by line.
        if process.stdout is not None:
            for line in process.stdout:
                num_identical = num_identical + 1 if line == last_line else 0
                last_line = line

                # Only print if there are not too many identical lines in a row.
                if num_identical < MAX_CONSECUTIVE_IDENTICAL_LOG_LINES:
                    # Print live output
                    print(line, end='')

                    # Print errors and warnings as GitHub Actions annotations.
                    _maybe_print_github_annotation(line)
                elif num_identical == MAX_CONSECUTIVE_IDENTICAL_LOG_LINES:
                    print(
                        '::warning :: Truncating output due to too many identical lines in a row.')

        assert process.wait() == 0, 'Command failed'


class DockerImage(ABC):
    """Abstract base class for Docker images.

    Wraps a dockerfile + build args.
    Support a base image if "FROM ${BASE_IMAGE}" is used in the Dockerfile.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @abstractmethod
    def image_name_root(self) -> str:
        """Root name for the image (without suffix)"""
        pass

    @abstractmethod
    def dockerfile_path(self) -> str:
        """Path to the Dockerfile"""
        pass

    @abstractmethod
    def parent_image(self) -> Optional['DockerImage']:
        """Image can have a single parent image.

        Returns a DockerImage or OSImage.
        """
        pass

    @abstractmethod
    def build_args(self) -> List[str]:
        """Build arguments for the docker build command"""
        return []

    def image_name_suffix(self) -> str:
        """Platform/arch dependent suffix for the image name"""
        suffix = (f'{self.args.platform.value}_cu{self.args.cuda_version.value}'
                  f'_u{self.args.ubuntu_version.value}')
        if self.args.gcc_sanitizer == 1:
            suffix += '_gsan'
        return suffix

    def image_name(self) -> str:
        """Full image name with suffix"""
        return self.image_name_root() + '_' + self.image_name_suffix()

    def build(self) -> None:
        """Build a docker image from a Dockerfile. First builds the parent image if it exists."""

        parent = self.parent_image()
        if parent is not None:
            parent.build()

        image_name = self.image_name()

        # Print build information
        print('=' * 80)
        print(f'BUILDING: {image_name}')
        print('=' * 80)
        print(f'Dockerfile:               {self.dockerfile_path()}')
        if parent is not None:
            print(f'Parent image:             {parent.image_name()}')
        print(f'Platform:                 {self.args.platform.value}')
        print(f'CUDA version:             {self.args.cuda_version.value}')
        print(f'CUDA architecture:        {self.args.cuda_arch.value}')
        print(f'Ubuntu version:           {self.args.ubuntu_version.value}')
        print(f'Max number of jobs:       {self.args.max_num_jobs}')
        build_args_str = ', '.join(self.build_args() or [])
        print(f'Build arguments:          {build_args_str}')
        user_build_args_str = ', '.join(self.args.user_build_args or [])
        print(f'User build arguments:     {user_build_args_str}')
        print('=' * 80)
        print('', flush=True)

        cmd = [
            'docker',
            'build',
            '-f',
            self.dockerfile_path(),
            '-t',
            image_name,
            '--network=host',
            '--progress=plain',
        ]

        if parent is not None:
            parent_name = parent.image_name()
            cmd += ['--build-arg', f'BASE_IMAGE={parent_name}']

        if self.build_args() is not None:
            for arg in self.build_args():
                cmd += ['--build-arg', arg]

        # Add extra docker args from args if provided
        if self.args.user_build_args is not None:
            cmd += self.args.user_build_args

        cmd += ['.']

        print(' '.join(cmd))

        _run_and_parse_log(cmd)

        self._validate()

    def _validate(self) -> None:
        """Validate that the correct cuda/ubuntu version was built"""

        # Check ubuntu version
        lsb_release_result = subprocess.run(
            ['docker', 'run', '--rm',
             self.image_name(), 'lsb_release', '-a'],
            check=True,
            capture_output=True,
            text=True,
        )
        expected_ubuntu = f'Ubuntu {self.args.ubuntu_version.value}'
        assert expected_ubuntu in lsb_release_result.stdout, (
            f'Failed to find the correct ubuntu version. '
            f'Stdout: {lsb_release_result.stdout}')

        # Check cuda version
        cuda_version_result = subprocess.run(
            ['docker', 'run', '--rm',
             self.image_name(), 'nvcc', '--version'],
            check=True,
            capture_output=True,
            text=True,
        )
        expected_cuda = f'cuda_{self.args.cuda_version.value}'
        assert expected_cuda in cuda_version_result.stdout, (
            f'Failed to find the correct cuda version. '
            f'Stdout: {cuda_version_result.stdout}')

        print(f'Successfully validated image: {self.image_name()}')


class TestBase(ABC):
    """Base class for running unit tests in a container"""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @abstractmethod
    def image(self) -> DockerImage:
        """Get the image to run the test on"""
        pass

    @abstractmethod
    def get_command(self) -> str:
        """Get the command to run in the test"""
        pass

    @abstractmethod
    def get_cwd(self) -> str:
        """Get the current working directory"""
        pass

    def run(self) -> None:
        """Build image and run command inside it"""
        self.image().build()
        docker_cmd = ['docker', 'run', '--privileged', '--rm', self.image().image_name()]
        cwd = self.get_cwd()
        cmd = self.get_command()
        full_cmd = docker_cmd + ['bash', '-c'] + [f'cd {cwd} && {cmd}']

        print(' '.join(full_cmd))
        _run_and_parse_log(full_cmd)


class OsImage(DockerImage):
    """External cuda or jetpack OS base image. Used as a parent image for other images."""

    AVAILABLE_OS_IMAGES = {
        Platform.X86_64: {
            CudaVersion.CUDA_11: {
                UbuntuVersion.UBUNTU_22: 'nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04',
            },
            CudaVersion.CUDA_12: {
                UbuntuVersion.UBUNTU_22: 'nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04',
                UbuntuVersion.UBUNTU_24: 'nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu24.04',
            },
            CudaVersion.CUDA_13: {
                UbuntuVersion.UBUNTU_22: 'nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu22.04',
                UbuntuVersion.UBUNTU_24: 'nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04',
            },
        },
        Platform.JETPACK_5: 'nvcr.io/nvidia/l4t-jetpack:r35.4.1',
        Platform.JETPACK_6: 'nvcr.io/nvidia/l4t-jetpack:r36.3.0',
    }

    def get_os_image_name(self) -> str:
        platform_images = self.AVAILABLE_OS_IMAGES.get(self.args.platform, {})
        cuda_images = platform_images.get(self.args.cuda_version, {})
        os_image = cuda_images.get(self.args.ubuntu_version)
        if os_image is None:
            raise ValueError(f'No OS image available for platform {self.args.platform}, '
                             f'cuda version {self.args.cuda_version}, '
                             f'and ubuntu version {self.args.ubuntu_version}')
        return os_image

    def image_name(self) -> str:
        return self.get_os_image_name()

    def image_name_root(self) -> str:
        raise NotImplementedError('OsImage does not have a base name')

    def dockerfile_path(self) -> str:
        raise NotImplementedError('OsImage does not have a dockerfile')

    def parent_image(self) -> None:
        return None

    def build_args(self) -> List[str]:
        return []

    def build(self) -> None:
        """OS images are external and do not need to be built"""
        pass
