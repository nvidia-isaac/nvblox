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
from typing import List, Optional


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


class DockerImage(ABC):
    """Abstract base class for Docker images.

    Wraps a dockerfile + build args. Supports single dependent parent image.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @abstractmethod
    def image_name_base(self) -> str:
        """Base name for the image (without suffix)"""
        pass

    @abstractmethod
    def dockerfile_path(self) -> str:
        """Path to the Dockerfile"""
        pass

    @abstractmethod
    def parent_image(self) -> Optional['DockerImage']:
        """Image can have a single parent image.

        Returns one of the DockerImage subclasses defined in this project.
        """
        pass

    def do_validate_image(self) -> bool:
        """Whether to validate the image after building. Override to disable validation."""
        return True

    @abstractmethod
    def build_args(self) -> List[str]:
        """Build arguments for the docker build command"""
        return []

    def image_name_suffix(self) -> str:
        """Platform/arch dependent suffix for the image name"""
        return (f'{self.args.platform.value}_cu{self.args.cuda_version.value}'
                f'_u{self.args.ubuntu_version.value}')

    def image_name(self) -> str:
        """Full image name with suffix"""
        return self.image_name_base() + '_' + self.image_name_suffix()

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

        cmd = [
            'docker', 'build', '-f',
            self.dockerfile_path(), '-t', image_name, '--network=host', '--progress=plain'
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
        subprocess.run(cmd, check=True)

        if self.do_validate_image():
            self._validate()

    def _validate(self) -> None:
        """Validate that the correct cuda/ubuntu version was built"""

        # Check ubuntu version
        lsb_release_result = subprocess.run(
            ['docker', 'run', '--rm',
             self.image_name(), 'lsb_release', '-a'],
            check=True,
            capture_output=True,
            text=True)
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
            text=True)
        expected_cuda = f'cuda_{self.args.cuda_version.value}'
        assert expected_cuda in cuda_version_result.stdout, (
            f'Failed to find the correct cuda version. '
            f'Stdout: {cuda_version_result.stdout}')

        print(f'Successfully validated image: {self.image_name()}')


class TestBase(ABC):
    """Base class for unit tests"""

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
        docker_cmd = ['docker', 'run', '--rm', self.image().image_name()]
        cwd = self.get_cwd()
        cmd = self.get_command()
        full_cmd = docker_cmd + ['bash', '-c'] + [f'cd {cwd} && {cmd}']
        subprocess.run(full_cmd, check=True)


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
        Platform.JETPACK_6: 'nvcr.io/nvidia/l4t-jetpack:r36.3.0'
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

    def image_name_base(self) -> str:
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
