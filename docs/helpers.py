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
import dataclasses
import datetime
import os
import re


def to_datetime(date_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(date_str, '%d.%m.%Y')


def is_expired(start_date: datetime.datetime, days: int) -> bool:
    today = datetime.datetime.now()
    delta = datetime.timedelta(days=days)
    return today > (start_date + delta)


@dataclasses.dataclass
class TemporaryLinkcheckIgnore:
    url: str
    start_date: datetime.datetime
    days: int


def get_version_from_multiversion_env() -> str:
    """Get version number from sphinx-multiversion environment.

    When building with sphinx-multiversion, extract version from
    SPHINX_MULTIVERSION_NAME environment variable or from the checked-out
    branch's setup.py file (e.g., "v0.0.8" -> "0.0.8").

    Falls back to reading from setup.py for local single-version builds.

    Returns:
        Version string like "0.0.8" or "0.0.9"
    """
    # Check if running under sphinx-multiversion
    smv_name = os.environ.get('SPHINX_MULTIVERSION_NAME')

    # Debug output
    print(f'[VERSION DEBUG] SPHINX_MULTIVERSION_NAME = {smv_name}')
    print(f'[VERSION DEBUG] Current directory: {os.getcwd()}')
    print(f'[VERSION DEBUG] __file__ location: {os.path.abspath(__file__)}')

    if smv_name:
        # Parse version from branch/tag name
        # Handles: "v0.0.8-docs", "v0.0.9-docs", with or without 'v' prefix
        match = re.match(r'v?(\d+\.\d+\.\d+)', smv_name)
        if match:
            version = match.group(1)
            print(f'[VERSION DEBUG] Extracted version from env: {version}')
            return version

        # For "public" branch, read from current setup.py
        if smv_name == 'public':
            version = _read_version_from_setup()
            print(f'[VERSION DEBUG] Public branch, read from setup.py: {version}')
            return version

    # Fallback: read from setup.py in the current working directory
    # When sphinx-multiversion builds, it checks out each version to a temp dir,
    # so we should read from THAT version's setup.py
    version = _read_version_from_setup()
    print(f'[VERSION DEBUG] Reading from setup.py: {version}')
    return version


def _read_version_from_setup() -> str:
    """Read version from setup.py by parsing file content.

    This avoids Python import caching issues.

    Uses current working directory instead of __file__ location because
    sphinx-multiversion changes CWD to the checked-out version's directory,
    but __file__ still points to the original file location.
    """
    # Use current working directory (which sphinx-multiversion sets correctly)
    setup_path = os.path.join(os.getcwd(), '..', 'nvblox_torch', 'setup.py')
    setup_path = os.path.abspath(setup_path)

    print(f'[VERSION DEBUG] Reading version from: {setup_path}')

    with open(setup_path, 'r', encoding='utf-8') as f:
        content = f.read()

    match = re.search(r"NVBLOX_VERSION_NUMBER\s*=\s*['\"]([^'\"]+)['\"]", content)
    if match:
        version = match.group(1)
        print(f'[VERSION DEBUG] Found version in setup.py: {version}')
        return version

    raise ValueError(f'Could not find NVBLOX_VERSION_NUMBER in {setup_path}')
