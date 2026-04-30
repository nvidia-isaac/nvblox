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
import re
from typing import List, Any

from sphinx.application import Sphinx

UNKNOWN_VERSION = 'unknown'

WHEEL_BASE_URL = 'https://github.com/nvidia-isaac/nvblox/releases/download'


def get_wheel_url_0_0_8(cuda_version: str, ubuntu_version: str) -> str:
    """Get the wheel URL for version 0.0.8.

    It is a special case because it has build number in the wheel filename.
    """
    # pylint: disable=line-too-long
    return f'{WHEEL_BASE_URL}/v0.0.8/nvblox_torch-0.0.8rc5+cu{cuda_version}ubuntu{ubuntu_version}-863-py3-none-linux_x86_64.whl'


def get_wheel_url_general(version: str, cuda_version: str, ubuntu_version: str) -> str:
    """Get the wheel URL for a given version, CUDA version, and Ubuntu version.
    """
    # pylint: disable=line-too-long
    return f'{WHEEL_BASE_URL}/v{version}/nvblox_torch-{version}+cu{cuda_version}ubuntu{ubuntu_version}-py3-none-linux_x86_64.whl'


def get_wheel_url(version: str, cuda_version: str, ubuntu_version: str) -> str:
    """Get the wheel URL for a given version, CUDA version, and Ubuntu version.
    """
    if version == '0.0.8':
        return get_wheel_url_0_0_8(cuda_version, ubuntu_version)
    else:
        return get_wheel_url_general(version, cuda_version, ubuntu_version)


def get_smv_version_number(app: Sphinx) -> str:
    """Get the version number from the sphinx-multiversion current version.

    """
    smv_current_version = getattr(app.config, 'smv_current_version', None)
    if smv_current_version is None:
        return UNKNOWN_VERSION

    # Extract 0.0.9 from branch name like v0.0.9-docs_test
    match = re.search(r'v(\d+\.\d+\.\d+)', smv_current_version)
    if not match:
        return UNKNOWN_VERSION
    version = match.group(1)

    return version


def nvblox_torch_pip_install_code_block(app: Sphinx, _: Any, source: List[str]) -> None:
    """Replaces the :nvblox_torch_pip_install_code_block: directive with a code block.

    The output pip command depends on whether we're in release or internal mode.
    We also generate pip commands for the two different CUDA versions.

    """

    def replacer(_: Any) -> str:

        version = get_smv_version_number(app)

        wheel_name_ubuntu_24_cuda_12 = get_wheel_url(version, '12', '24')
        wheel_name_ubuntu_22_cuda_12 = get_wheel_url(version, '12', '22')
        wheel_name_ubuntu_24_cuda_13 = get_wheel_url(version, '13', '24')


        pip_install_target_ubuntu_24_cuda_12 = \
            f'{wheel_name_ubuntu_24_cuda_12}'
        pip_install_target_ubuntu_22_cuda_12 = \
            f'{wheel_name_ubuntu_22_cuda_12}'
        pip_install_target_ubuntu_24_cuda_13 = \
            f'{wheel_name_ubuntu_24_cuda_13}'

        rst_string = f"""

To install ``nvblox_torch`` via ``pip`` on a supported platform, run the following commands:

.. tabs::
    .. tab:: Ubuntu 24.04 + CUDA 12.8

        .. code-block:: bash

            sudo apt-get install python3-pip libglib2.0-0 libgl1 # Open3D dependencies
            pip3 install {pip_install_target_ubuntu_24_cuda_12}

    .. tab:: Ubuntu 22.04 + CUDA 12.6

        .. code-block:: bash

            sudo apt-get install python3-pip libglib2.0-0 libgl1 # Open3D dependencies
            pip3 install {pip_install_target_ubuntu_22_cuda_12}

"""
        # Only add the CUDA 13.0 tab if the version is not 0.0.8.
        # TODO(dtingdahl) handle this in a more elegant way to support future releases.
        if version != '0.0.8':
            rst_string += f"""
    .. tab:: Ubuntu 24.04 + CUDA 13.0

        .. code-block:: bash

            sudo apt-get install python3-pip libglib2.0-0 libgl1 # Open3D dependencies
            pip3 install torch==2.9.1+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130/
            pip3 install {pip_install_target_ubuntu_24_cuda_13}

"""
        return rst_string

    source[0] = re.sub(r':nvblox_torch_pip_install_code_block:', replacer, source[0])


def nvblox_torch_git_clone_code_block(app: Sphinx, _: Any, source: List[str]) -> None:
    """Replaces the :nvblox_torch_git_clone_code_block: directive with a code block.

    The output git clone command depends on whether we're in release or internal mode.

    """

    def replacer(_: Any) -> str:
        release_state = app.config.nvblox_torch_docs_config['released']
        internal_git_url = app.config.nvblox_torch_docs_config['internal_git_url']
        external_git_url = app.config.nvblox_torch_docs_config['external_git_url']
        if release_state:
            git_clone_target = external_git_url
        else:
            git_clone_target = internal_git_url
        return f"""
.. code-block:: bash

    git clone {git_clone_target}

"""

    source[0] = re.sub(r':nvblox_torch_git_clone_code_block:', replacer, source[0])


# pylint: disable=unused-argument
def download_sun3d_test_dataset(sphinx: Sphinx, _: Any, source: List[str]) -> None:
    """Replaces the :download_sun3d_test_dataset: directive with a code block.
    """

    def replacer(_: Any) -> str:

        return """

Download an example SUN3D dataset by running the following command:

.. code-block:: bash

    wget http://3dvision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-mit_76_studyroom-76-1studyroom2.zip
    unzip sun3d-mit_76_studyroom-76-1studyroom2.zip

"""

    source[0] = re.sub(r':download_sun3d_test_dataset:', replacer, source[0])


# pylint: disable=unused-argument
def download_replica_test_dataset(sphinx: Sphinx, _: Any, source: List[str]) -> None:
    """Replaces the :download_replica_test_dataset: directive with a code block.
    """

    def replacer(_: Any) -> str:

        return """

Download an example Replica dataset by running the following command:

.. code-block:: bash

    wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
    unzip Replica.zip

"""

    source[0] = re.sub(r':download_replica_test_dataset:', replacer, source[0])


def nvblox_code_link(app: Sphinx, _: Any, source: List[str]) -> None:
    """Replaces the :nvblox_code_link: directive with a code block.

    The output link is either gitlab (internal) or github (external) depending on the release state.

    """

    def replacer(match: re.Match) -> str:
        relative_path = match.group('relative_path')
        release_state = app.config.nvblox_torch_docs_config['released']
        internal_code_link_base_url = app.config.nvblox_torch_docs_config[
            'internal_code_link_base_url']
        external_code_link_base_url = app.config.nvblox_torch_docs_config[
            'external_code_link_base_url']
        # Extract the file name
        file_name = relative_path.split('/')[-1]
        if release_state:
            code_link_base_url = external_code_link_base_url
        else:
            code_link_base_url = internal_code_link_base_url
        return f'`{file_name} <{code_link_base_url}/{relative_path}>`_'

    source[0] = re.sub(r':nvblox_code_link:`<(?P<relative_path>.*)>`', replacer, source[0])


def current_version_name(app: Sphinx, _: Any, source: List[str]) -> None:
    """Replaces the :current_version_name: directive with the current version name.

    This uses the sphinx-multiversion context if available, otherwise falls back
    to the Sphinx version config value.

    Usage in RST:
        Current Version: :current_version_name:

    """

    def replacer(_: Any) -> str:
        # Try to get the version from sphinx-multiversion's environment
        # When sphinx-multiversion builds, it sets the 'smv_current_version' in the environment
        smv_current_version = getattr(app.config, 'smv_current_version', None)
        if smv_current_version:

            # Extract version number from string containing v0.0.9
            version = get_smv_version_number(app)
            return version
        else:
            raise ValueError('Failed to get current version name. Build with make multi-doc.')

    source[0] = re.sub(r':current_version_name:', replacer, source[0])


def setup(app: Sphinx) -> None:
    app.connect('source-read', nvblox_torch_pip_install_code_block)
    app.connect('source-read', nvblox_torch_git_clone_code_block)
    app.connect('source-read', nvblox_code_link)
    app.connect('source-read', download_replica_test_dataset)
    app.connect('source-read', download_sun3d_test_dataset)
    app.connect('source-read', current_version_name)
    app.add_config_value('nvblox_torch_docs_config', {}, 'env')
