#!/usr/bin/env python3
"""Helper for building nvblox docker images
"""

import os
import sys
import subprocess
from typing import Optional, List, Dict, Set, Type
from abc import ABC, abstractmethod
import argparse
from enum import Enum


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


class NvbloxImage(Enum):
    DEPS = 'deps'
    BUILD = 'build'
    REALSENSE = 'realsense'
    DOCS = 'docs'
    LINT = 'lint'


class DockerImage(ABC):
    """Abstract base class for Docker images. Wraps a dockerfile + build args. Supports single dependent parent image."""

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
    def parent_image(self):
        """Image can have a single parent image, which is one of the DockerImage subclasses defined in this project."""
        pass

    @abstractmethod
    def build_args(self) -> List[str]:
        """Build arguments for the docker build command"""
        return []

    def image_name_suffix(self) -> str:
        """Platform/arch dependent suffix for the image name"""
        return self.args.platform.value + '_cu' + self.args.cuda_version.value + '_u' + self.args.ubuntu_version.value

    def image_name(self) -> str:
        """Full image name with suffix"""
        return self.image_name_base() + '_' + self.image_name_suffix()

    def build(self) -> None:
        """Build a docker image from a Dockerfile. First builds the parent image if it exists."""

        if self.parent_image() is not None:
            self.parent_image().build()

        image_name = self.image_name()

        # Print build information
        print('=' * 80)
        print(f'BUILDING: {image_name}')
        print('=' * 80)
        print(f'Dockerfile:               {self.dockerfile_path()}')
        print(f'Parent image:             {self.parent_image().image_name()}')
        print(f'Platform:                 {self.args.platform.value}')
        print(f'CUDA version:             {self.args.cuda_version.value}')
        print(f'CUDA architecture:        {self.args.cuda_arch.value}')
        print(f'Ubuntu version:           {self.args.ubuntu_version.value}')
        print(f'Max number of build jobs: {self.args.max_num_build_jobs}')
        print(f'Build arguments:          {", ".join(self.build_args() or [])}')
        print(f'User build arguments:     {", ".join(self.args.user_build_args or [])}')
        print('=' * 80)

        cmd = [
            'docker', 'build', '-f',
            self.dockerfile_path(), '-t', image_name, '--network=host', '--progress=plain'
        ]

        if self.parent_image() is not None:
            cmd += ['--build-arg', f'BASE_IMAGE={self.parent_image().image_name()}']

        if self.build_args() is not None:
            for arg in self.build_args():
                cmd += ['--build-arg', arg]

        # Add extra docker args from args if provided
        if self.args.user_build_args is not None:
            cmd += self.args.user_build_args

        cmd += ['.']

        print(cmd)
        subprocess.run(cmd, check=True)

        self._validate()

    def _validate(self) -> None:
        """Validate that the correct cuda/ubuntu version was built"""

        # Check ubuntu version
        lsb_release_result = subprocess.run(['docker', 'run', '--rm', self.image_name(), 'lsb_release', '-a'], check=True, capture_output=True, text=True)
        assert f"Ubuntu {self.args.ubuntu_version.value}" in lsb_release_result.stdout, f"Failed to find the correct ubuntu version. Stdout: {lsb_release_result.stdout}"
        
        # Check cuda version
        cuda_version_result = subprocess.run(['docker', 'run', '--rm', self.image_name(), 'nvcc', '--version'], check=True, capture_output=True, text=True)
        assert f"cuda_{self.args.cuda_version.value}" in cuda_version_result.stdout, f"Failed to find the correct cuda version. Stdout: {cuda_version_result.stdout}"

        print(f"Successfully validated image: {self.image_name()}")

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
        os_image = self.AVAILABLE_OS_IMAGES.get(self.args.platform,
                                                {}).get(self.args.cuda_version,
                                                        {}).get(self.args.ubuntu_version)
        if os_image is None:
            raise ValueError(
                f'No OS image available for platform {self.args.platform}, cuda version {self.args.cuda_version}, and ubuntu version {self.args.ubuntu_version}'
            )
        return os_image

    def image_name(self) -> str:
        return self.get_os_image_name()

    def image_name_base(self) -> str:
        return None

    def dockerfile_path(self) -> str:
        return None

    def parent_image(self) -> None:
        return None

    def build_args(self) -> List[str]:
        return None

    def build(self):
        return None


class DependenciesImage(DockerImage):
    """Nvblox dependencies image"""

    def build_args(self) -> List[str]:
        return None

    def image_name_base(self) -> str:
        return 'nvblox_deps'

    def dockerfile_path(self) -> str:
        if self.args.platform == Platform.X86_64:
            return os.path.join('docker', 'Dockerfile.deps')
        else:
            return os.path.join('docker', 'Dockerfile.jetson_deps')

    def parent_image(self):
        return OsImage(self.args)


class BuildImage(DockerImage):
    """Nvblox build image containing binaries"""

    def image_name_base(self) -> str:
        return 'nvblox_build'

    def dockerfile_path(self) -> str:
        return os.path.join('docker', 'Dockerfile.build')

    def parent_image(self) -> DockerImage:
        return DependenciesImage(self.args)

    def external_base_image_url(self) -> str:
        return None

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
        args = [f'CMAKE_ARGS=\"-DCMAKE_CUDA_ARCHITECTURES={self.get_cuda_sm_architecture()}\"']
        if self.args.max_num_build_jobs is not None:
            args += [f'MAX_NUM_JOBS={self.args.max_num_build_jobs}']
        return args


class RealsenseImage(DockerImage):
    """Nvblox realsense example image"""

    def image_name_base(self) -> str:
        return 'nvblox_realsense_example'

    def dockerfile_path(self) -> str:
        return os.path.join('docker', 'Dockerfile.realsense_example')

    def parent_image(self) -> DockerImage:
        return BuildImage(self.args)

    def build_args(self) -> None:
        return None


class DocsImage(DockerImage):
    """Nvblox documentation image. Does not have any internal dependencies."""

    def image_name_base(self) -> str:
        return 'nvblox_docs'

    def dockerfile_path(self) -> str:
        return os.path.join('docker', 'Dockerfile.docs')

    def parent_image(self) -> None:
        return None

    def build_args(self) -> None:
        return None


class LintImage(DockerImage):
    """Nvblox lint image. Does not have any internal dependencies."""

    def image_name_base(self) -> str:
        return 'nvblox_lint'

    def dockerfile_path(self) -> str:
        return os.path.join('docker', 'Dockerfile.lint')

    def parent_image(self) -> None:
        return None

    def build_args(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build nvblox docker images')
    parser.add_argument('--cuda-version',
                        type=CudaVersion,
                        default=CudaVersion.CUDA_12,
                        help='CUDA version')
    parser.add_argument('--cuda-arch',
                        type=CudaSmArchitectures,
                        required=False,
                        default=CudaSmArchitectures.SM_NATIVE,
                        help='CUDA SM architectures.')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Docker image to build. Choices are: deps, binaries, realsense-example')
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
    parser.add_argument('--max-num-build-jobs',
                        type=int,
                        required=False,
                        default=8,
                        help='Maximum number of build jobs to run in parallel.')

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.image == NvbloxImage.DEPS.value:
        DependenciesImage(args=args).build()
    elif args.image == NvbloxImage.BUILD.value:
        BuildImage(args).build()
    elif args.image == NvbloxImage.REALSENSE.value:
        RealsenseImage(args).build()
    elif args.image == NvbloxImage.DOCS.value:
        DocsImage(args).build()
    elif args.image == NvbloxImage.LINT.value:
        LintImage(args).build()


if __name__ == '__main__':
    sys.exit(main())
