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

import subprocess
import re
import platform
import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TorchVersion:
    """Store information of a pythorch version."""

    platform: str
    cuda_version: str

    pytorch_version: str
    torchvision_version: str
    index_url: Optional[str] = None

    # Special treatment needed for Jetson:
    # * Need to explicitly install urls.
    # * Wheel urls has to be renamed to be installable. Hence we need the filename as well
    # TODO(dtingdahl): Remove these variables by using the pip install method described on
    #  https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html.
    #  So far, this method cause linkage errors in our containers due to wrong version of cuDNN.
    pytorch_url: Optional[str] = None
    torchvision_url: Optional[str] = None
    torchvision_filename: Optional[str] = None
    pytorch_filename: Optional[str] = None


# List of supported pytorch versions in this project.
TORCH_VERSIONS = [
    TorchVersion(
        platform='x86_64',
        cuda_version='12',
        index_url='https://download.pytorch.org/whl/cu128',
        pytorch_version='2.9.1',
        torchvision_version='0.24.1',
    ),
    TorchVersion(
        platform='x86_64',
        cuda_version='13',
        index_url='https://download.pytorch.org/whl/cu130',
        pytorch_version='2.9.1',
        torchvision_version='0.24.1',
    ),
    TorchVersion(
        platform='aarch64',
        cuda_version='12',
        pytorch_version='2.3.0',
        torchvision_version='0.18.0',
    # pylint: disable=line-too-long
        pytorch_url='https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl',
        pytorch_filename='torch-2.3.0-cp310-cp310-linux_aarch64.whl',
    # pylint: disable=line-too-long
        torchvision_url='https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl',
        torchvision_filename='torchvision-0.18.0-cp310-cp310-linux_aarch64.whl',
    ),
]


def get_cuda_version() -> str:
    """Get cuda version of the system"""
    # use re.search to find the cuda version
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
    match = re.search(r'release (\d+\.\d+)', result.stdout)
    if not match:
        raise ValueError(f'Failed to find cuda version in {result.stdout}')
    return match.group(1).split('.')[0]


def get_pytorch_version_for_this_machine() -> Optional[TorchVersion]:
    """Get the pytorch version for the current system or None if not supported."""

    print(f'platform.machine(): {platform.machine()}')
    print(f'get_cuda_version(): {get_cuda_version()}')

    result = [
        v for v in TORCH_VERSIONS
        if v.platform == platform.machine() and v.cuda_version == get_cuda_version()
    ]

    if not result:
        print(f'No pytorch version found for {platform.machine()} '
              f'with cuda version: {get_cuda_version()}')
        return None
    print(f'pytorch version: {result}')
    assert len(result) <= 1, 'Expected 1 pytorch version'
    return result[0]


def download_and_rename_wheel(url: Optional[str], filename: Optional[str]) -> Optional[str]:
    """Download the wheel and rename it"""
    subprocess.run(f'wget {url} -O {filename}', shell=True, check=True)
    return filename


def install_pytorch_if_supported_for_this_machine() -> None:

    pytorch_version = get_pytorch_version_for_this_machine()
    if pytorch_version is None:
        print('pytorch not supported on this system')
        return

    script = """
    set -ex
    umask 000
    . /opt/venv/bin/activate
    pip install --ignore-installed --upgrade pip --no-cache-dir
    """

    if pytorch_version.platform == 'x86_64':
        script += f"""
            pip install \
                --no-cache-dir \
                --index-url {pytorch_version.index_url} \
                torch=={pytorch_version.pytorch_version} \
                torchvision=={pytorch_version.torchvision_version}
            """
    else:
        # If on aarch64, we need to download the wheel blob and rename it.
        # Otherwise pip won't install it.
        assert pytorch_version.pytorch_filename is not None
        assert pytorch_version.torchvision_filename is not None
        pytorch_file = download_and_rename_wheel(pytorch_version.pytorch_url,
                                                 pytorch_version.pytorch_filename)
        torchvision_file = download_and_rename_wheel(pytorch_version.torchvision_url,
                                                     pytorch_version.torchvision_filename)

        script += f"""
            pip install --no-cache-dir {pytorch_file}
            pip install --no-cache-dir {torchvision_file}
            """

    subprocess.run(script, shell=True, check=True)


def install_nvblox_torch_if_supported_for_this_machine() -> None:

    pytorch_version = get_pytorch_version_for_this_machine()
    if pytorch_version is None:
        print('nvblox torch not supported on this system')
        return

    script = f"""
    set -ex
    umask 000
    . /opt/venv/bin/activate
    pip install --ignore-installed --upgrade pip --no-cache-dir
    # Need to force the torch version to prevent accidental upgrades.
    pip install /nvblox/nvblox_torch/ torch=={pytorch_version.pytorch_version}
    """

    subprocess.run(script, shell=True, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--install-nvblox-torch-if-supported',
        action='store_true',
        help='Install nvblox torch if supported',
    )
    parser.add_argument(
        '--install-pytorch-if-supported',
        action='store_true',
        help='Install pytorch if supported',
    )

    args = parser.parse_args()

    if (not args.install_pytorch_if_supported and not args.install_nvblox_torch_if_supported):
        parser.error('Either --install-pytorch-if-supported or '
                     '--install-nvblox-torch-if-supported must be provided')

    return args


def main() -> None:
    """Platform dependent installation of pytorch and nvblox torch.

    Note that the pypi version of pytorch is locked to a specific
    CUDA version (12 at the time of writing). Therefore we need this custom install script.
    """
    args = parse_args()
    if args.install_pytorch_if_supported:
        install_pytorch_if_supported_for_this_machine()
    if args.install_nvblox_torch_if_supported:
        install_nvblox_torch_if_supported_for_this_machine()


if __name__ == '__main__':
    main()
