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

# pylint: disable=line-too-long
X86_CUDA_VERSION_TO_PYTORCH_PIP_URL = {
    # Latest version of pytorch compatible with CUDA 11
    '11': '--index-url https://download.pytorch.org/whl/cu118/torch/ --extra-index-url https://pypi.org/simple torch==2.7.1',
    # Default pypi version uses CUDA 12
    '12': 'torch==2.9.1',
    '13': '--index-url https://download.pytorch.org/whl/cu130/torch/ --extra-index-url https://pypi.org/simple torch==2.9.1',
}

# We currently only support Jetpack 6 (Cuda12) for jetson. These are nvidia-hosted builds of JP6 wheels.
JETSON_CUDA_VERSION_TO_TORCH = {
    '12': {
        'url': 'https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl',
        'filename': 'torch-2.3.0-cp310-cp310-linux_aarch64.whl'
    },
}

# On Jetson we also need to explicitly install torchvision.
JETSON_CUDA_VERSION_TO_TORCHVISION = {
    '12': {
        'url': 'https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl',
        'filename': 'torchvision-0.18.0-cp310-cp310-linux_aarch64.whl'
    },
}
# pylint: enable=line-too-long


def get_cuda_version() -> str:
    """Get cuda version of the system"""
    # use re.search to find the cuda version
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
    match = re.search(r'release (\d+\.\d+)', result.stdout)
    if not match:
        raise ValueError(f'Failed to find cuda version in {result.stdout}')
    return match.group(1).split('.')[0]


def install_pytorch_x86() -> None:

    cuda_version = get_cuda_version()
    if cuda_version not in X86_CUDA_VERSION_TO_PYTORCH_PIP_URL:
        raise ValueError(f'Unsupported CUDA version: {cuda_version}')

    pytorch_install = X86_CUDA_VERSION_TO_PYTORCH_PIP_URL[cuda_version]
    print(f'installing pytorch from {pytorch_install}')

    script = f"""
    set -ex
    umask 000
    . /opt/venv/bin/activate
    python3 -m pip install --ignore-installed --upgrade pip --no-cache-dir
    python3 -m pip install --no-cache-dir {pytorch_install}
    """

    subprocess.run(script, shell=True, check=True)


def install_pytorch_jetson() -> None:

    cuda_version = get_cuda_version()
    if cuda_version not in JETSON_CUDA_VERSION_TO_TORCH:
        print('warning: Unsupported CUDA version: {cuda_version}. Skipping pytorch installation.')
        return

    torch_url = JETSON_CUDA_VERSION_TO_TORCH[cuda_version]['url']
    torch_filename = JETSON_CUDA_VERSION_TO_TORCH[cuda_version]['filename']
    torchvision_url = JETSON_CUDA_VERSION_TO_TORCHVISION[cuda_version]['url']
    torchvision_filename = JETSON_CUDA_VERSION_TO_TORCHVISION[cuda_version]['filename']

    script = f"""
    set -ex
    umask 000
    . /opt/venv/bin/activate
    python3 -m pip install --ignore-installed --upgrade pip --no-cache-dir
    wget {torch_url} -O  /{torch_filename}
    pip install /{torch_filename}
    rm /{torch_filename}
    wget {torchvision_url} -O /{torchvision_filename}
    pip install /{torchvision_filename}
    rm /{torchvision_filename}
    """
    subprocess.run(script, shell=True, check=True)


def install_nvblox_torch() -> None:

    cuda_version = get_cuda_version()
    supported_cuda_versions = list(JETSON_CUDA_VERSION_TO_TORCH.keys()
                                   | X86_CUDA_VERSION_TO_PYTORCH_PIP_URL.keys())
    if cuda_version not in supported_cuda_versions:
        print(f'warning: Unsupported CUDA version: {cuda_version}. Skipping nvblox torch install.')
        return

    script = """
    set -ex
    umask 000
    . /opt/venv/bin/activate
    python3 -m pip install --ignore-installed --upgrade pip --no-cache-dir
    """
    # Need to force the torch version for cuda 11 to prevent upgrade.
    if cuda_version == '11':
        script += 'pip install /nvblox/nvblox_torch/ "torch==2.7.1"'
    else:
        script += 'pip install /nvblox/nvblox_torch/'
    subprocess.run(script, shell=True, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--install-nvblox-torch-if-supported',
                        action='store_true',
                        help='Install nvblox torch if supported')
    parser.add_argument('--install-pytorch-if-supported',
                        action='store_true',
                        help='Install pytorch if supported')

    args = parser.parse_args()

    if not args.install_pytorch_if_supported and not args.install_nvblox_torch_if_supported:
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
        if platform.machine() == 'x86_64':
            install_pytorch_x86()
        elif platform.machine() == 'aarch64':    # Jetson
            install_pytorch_jetson()
        else:
            raise ValueError(f'Unsupported system: {platform.machine()}')
    elif args.install_nvblox_torch_if_supported:
        install_nvblox_torch()


if __name__ == '__main__':
    main()
