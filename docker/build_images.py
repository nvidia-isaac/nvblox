#!/usr/bin/env python3
"""Helper for building nvblox docker images

"""

import os
import sys
import subprocess
from typing import Optional, List
import argparse

# Realsense lib needs ubuntu22, so that's our base image
REALSENSE_BASE_IMAGE = 'nvcr.io/nvidia/cuda:12.6.1-devel-ubuntu22.04'
REALSENSE_IMAGE_NAME_SUFFIX = '_cu12_u22'


def build_image(dockerfile_path: str,
                image_name: str,
                base_image: Optional[str] = None,
                extra_build_args: Optional[List[str]] = None) -> None:
    """Build a docker image from a Dockerfile"""

    print('=' * 80)
    print(f'Building {image_name} from {dockerfile_path}')
    print(f'with base image {base_image}')
    print('=' * 80)

    cmd = ['docker', 'build', '-f', dockerfile_path, '-t', image_name, '--network=host']

    if base_image:
        cmd += ['--build-arg', f'BASE_IMAGE={base_image}']

    if extra_build_args:
        cmd += extra_build_args

    cmd += ['.']

    print(cmd)
    subprocess.run(cmd, check=True)


def build_deps_image(base_image: Optional[str] = None, image_name_suffix: str = '') -> str:
    """Build nvblox dependencies (deps) image"""
    image_name = 'nvblox_deps' + image_name_suffix
    build_image(dockerfile_path=os.path.join('docker', 'Dockerfile.deps'),
                image_name=image_name,
                base_image=base_image,
                extra_build_args=[])
    return image_name


def build_binaries_image(base_image: Optional[str] = None,
                         image_name_suffix: str = '',
                         cuda_arch: Optional[str] = None,
                         skip_build_binaries_docker: bool = False,
                         max_num_build_jobs: Optional[int] = None) -> str:
    """Build nvblox binaries (.build) image"""
    image_name = 'nvblox_build' + image_name_suffix

    if cuda_arch is None:
        cuda_arch = get_cuda_arch()

    extra_build_args = ['--build-arg', f"CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}'"]

    if max_num_build_jobs is not None:
        extra_build_args.extend(['--build-arg', f'MAX_NUM_JOBS={max_num_build_jobs}'])

    if not skip_build_binaries_docker:
        build_image(dockerfile_path=os.path.join('docker', 'Dockerfile.build'),
                    image_name=image_name,
                    base_image=base_image,
                    extra_build_args=extra_build_args)

    return image_name


def build_realsense_example_image(base_image: Optional[str] = None,
                                  image_name_suffix: str = '') -> str:
    """Build nvblox realsense example image"""
    image_name = 'nvblox_realsense_example' + image_name_suffix
    build_image(dockerfile_path=os.path.join('docker', 'Dockerfile.realsense_example'),
                image_name=image_name,
                base_image=base_image,
                extra_build_args=[])

    return image_name


def get_cuda_arch() -> str:
    """Get the cuda architecture from nvidia-smi"""
    try:
        command_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv'])
        arch = command_output.decode('utf-8').split()[1].replace('.', '')
        return arch
    except FileNotFoundError:
        print('ERROR:nvidia-smi not found. If on aarch64, provide architecture from command line.')
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-deps-image',
                        '-d',
                        action='store_true',
                        help='Whether to build the dependencies (deps) image')
    parser.add_argument('--build-binaries-image',
                        '-b',
                        action='store_true',
                        help='Whether to build the binaries (build) image')
    parser.add_argument('--build-realsense-example-image',
                        '-r',
                        action='store_true',
                        help='Whether to build the realsense (realsense) image')
    parser.add_argument(
        '--skip-build-binaries-docker',
        '-n',
        action='store_true',
        help='Whether to skip building the binary (build) image and rely a previously built image.')
    parser.add_argument('--base-image',
                        type=str,
                        default=None,
                        help='Base image. Will be taken from Dockerfile.deps if not given')
    parser.add_argument('--image_name_suffix',
                        type=str,
                        default='',
                        help='Suffix for created docker image name.')
    parser.add_argument('--cuda-arch',
                        type=str,
                        help='Optionally input cuda architectures to build for as a '
                        'semicolon separated list. e.g. "90;89" ')
    parser.add_argument('--max-num-build-jobs',
                        type=int,
                        help='Max number of build jobs',
                        default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.build_deps_image:
        build_deps_image(base_image=args.base_image, image_name_suffix=args.image_name_suffix)

    if args.build_binaries_image:
        # need deps
        deps_image = build_deps_image(base_image=args.base_image,
                                      image_name_suffix=args.image_name_suffix)
        build_binaries_image(base_image=deps_image,
                             image_name_suffix=args.image_name_suffix,
                             cuda_arch=args.cuda_arch,
                             skip_build_binaries_docker=args.skip_build_binaries_docker,
                             max_num_build_jobs=args.max_num_build_jobs)

    if args.build_realsense_example_image:
        # need deps + binaries
        assert args.base_image is None, 'Cannot provide base image for realsense example'
        deps_image = build_deps_image(base_image=REALSENSE_BASE_IMAGE,
                                      image_name_suffix=REALSENSE_IMAGE_NAME_SUFFIX)
        binaries_image = build_binaries_image(
            base_image=deps_image,
            image_name_suffix=REALSENSE_IMAGE_NAME_SUFFIX,
            cuda_arch=args.cuda_arch,
            skip_build_binaries_docker=args.skip_build_binaries_docker,
            max_num_build_jobs=args.max_num_build_jobs)

        build_realsense_example_image(base_image=binaries_image,
                                      image_name_suffix=REALSENSE_IMAGE_NAME_SUFFIX)

    return 0


if __name__ == '__main__':
    sys.exit(main())
