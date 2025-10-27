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

# Realsense lib needs ubuntu22, so that's our base image
REALSENSE_BASE_IMAGE = 'nvcr.io/nvidia/cuda:12.6.1-devel-ubuntu22.04'
REALSENSE_IMAGE_NAME_SUFFIX = '_cu12_u22'


class Platform(Enum):
    X86_64 = 'x86_64'
    JETPACK_5 = 'jetpack-5'
    JETPACK_6 = 'jetpack-6'


class CudaVersion(Enum):
    CUDA_11 = '11'
    CUDA_12 = '12'
    CUDA_13 = '13'


class UbuntuVersion(Enum):
    UBUNTU_22 = '22'
    UBUNTU_24 = '24'


class CudaSmArchitectures(Enum):
    SM_120 = '120'
    SM_100 = '100'
    SM_90 = '90'
    SM_89 = '89'
    SM_86 = '86'
    SM_80 = '80'
    SM_75 = '75'
    SM_ALL = 'all'
    SM_NATIVE = 'native'


BASE_IMAGES = {
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


def get_base_image(platform: Platform, cuda_version: CudaVersion,
                   ubuntu_version: UbuntuVersion) -> str:
    base_image = BASE_IMAGES.get(platform, {}).get(cuda_version, {}).get(ubuntu_version)
    if base_image is None:
        raise ValueError(
            f'No base image found for platform {platform}, cuda version {cuda_version}, and ubuntu version {ubuntu_version}'
        )
    return base_image


class DockerImage(ABC):
    """Abstract base class for Docker images with dependency management"""

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
        """Parent image, one of the DockerImage subclasses defined in this project."""
        pass

    def external_base_image_url(self) -> str:
        """Full URL to external base image (e.g. from a registry) if applicable"""
        pass

    def image_name_suffix(self) -> str:
        return self.args.platform.value + '_cu' + self.args.cuda_version.value + '_u' + self.args.ubuntu_version.value

    def image_name(self) -> str:
        """Full image name with suffix"""
        return self.image_name_base() + '_' + self.image_name_suffix()

    def build(self, dockerfile_path: str, build_args: Optional[List[str]] = None) -> None:
        """Build a docker image from a Dockerfile"""
        image_name = self.image_name()
        print('=' * 80)
        print(f'Building {image_name} from {dockerfile_path}')
        print('=' * 80)

        cmd = ['docker', 'build', '-f', self.dockerfile_path(), '-t', image_name, '--network=host']

        if build_args:
            cmd += build_args

        if self.external_base_image_url() is not None:
            cmd += ['--build-arg', f'BASE_IMAGE={self.external_base_image_url()}']

        # Add extra docker args from args if provided
        if self.args.extra_docker_args is not None:
            cmd += self.args.extra_docker_args

        cmd += ['.']

        print(cmd)
        subprocess.run(cmd, check=True)


class DependenciesImage(DockerImage):
    """Nvblox dependencies image"""

    def base_image_url(self) -> str:
        return get_base_image(self.args.platform, self.args.cuda_version)

    def image_name_base(self) -> str:
        return 'nvblox_deps'

    def dockerfile_path(self) -> str:
        return os.path.join('docker', 'Dockerfile.deps')

    def parent_image(self) -> str:
        return None


class BuildImage(DockerImage):
    """Nvblox build image containing binaries and pytorch wrapper"""

    def image_name_base(self) -> str:
        return 'nvblox_build'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build nvblox docker images')
    parser.add_argument('--cuda-version', type=CudaVersion, required=True, help='CUDA version')
    parser.add_argument('--cuda-arch',
                        type=CudaSmArchitectures,
                        required=False,
                        help='CUDA SM architectures.')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Docker image to build. Choices are: deps, binaries, realsense-example')
    parser.add_argument('--platform', type=Platform, required=True, help='Platform to build for.')
    parser.add_argument('--ubuntu-version',
                        type=UbuntuVersion,
                        required=False,
                        default=UbuntuVersion.UBUNTU_24,
                        help='Ubuntu version to build for.')
    parser.add_argument('--extra-docker-args',
                        type=List[str],
                        required=False,
                        help='Extra docker build arguments.')

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.image == 'deps':
        DependenciesImage(args=args).build(args)
    elif args.image == 'binaries':
        binaries_image = BinariesImage(args=args)
        binaries_image.build(args)
    elif args.image == 'realsense-example':
        realsense_example_image = RealsenseExampleImage()
        realsense_example_image.build(args)


# # Legacy function compatibility (deprecated - use classes instead)
# def build_deps_image(base_image: Optional[str] = None, image_name_suffix: str = '') -> str:
#     """Build nvblox dependencies (deps) image - DEPRECATED: Use DepsImage class instead"""
#     builder = DockerImageBuilder()
#     deps_image = DepsImage(image_name_suffix=image_name_suffix, base_image=base_image)
#     return deps_image.build(builder)

# def build_binaries_image(base_image: Optional[str] = None,
#                          image_name_suffix: str = '',
#                          cuda_arch: Optional[str] = None,
#                          skip_build_binaries_docker: bool = False,
#                          max_num_build_jobs: Optional[int] = None) -> str:
#     """Build nvblox binaries (.build) image - DEPRECATED: Use BinariesImage class instead"""
#     builder = DockerImageBuilder()
#     binaries_image = BinariesImage(
#         image_name_suffix=image_name_suffix,
#         base_image=base_image,
#         cuda_arch=cuda_arch,
#         skip_build_binaries_docker=skip_build_binaries_docker,
#         max_num_build_jobs=max_num_build_jobs
#     )
#     return binaries_image.build(builder)

# def build_realsense_example_image(base_image: Optional[str] = None,
#                                   image_name_suffix: str = '') -> str:
#     """Build nvblox realsense example image - DEPRECATED: Use RealsenseExampleImage class instead"""
#     builder = DockerImageBuilder()
#     realsense_image = RealsenseExampleImage()
#     return realsense_image.build(builder)

if __name__ == '__main__':
    sys.exit(main())
