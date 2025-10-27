#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import sys
import os
import argparse
import io
import contextlib

# Dig up the module that we're testing
CI_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'ci')
sys.path.insert(0, CI_PATH)
# pylint: disable=wrong-import-position
import ship_it


def make_args(build_number: str = '1234',
              print_staging_url: bool = False,
              print_release_url: bool = False,
              build_package: bool = True) -> argparse.Namespace:
    args = argparse.Namespace()
    args.build_number = build_number
    args.print_staging_url = print_staging_url
    args.print_release_url = print_release_url
    args.build_package = build_package
    return args


def test_build_package() -> None:
    _run_and_check_output(make_args(build_package=True), expected_output='Finished')


def test_print_staging_url() -> None:
    _run_and_check_output(make_args(print_staging_url=True), expected_output='urm.nvidia.com')


def test_print_release_url() -> None:
    _run_and_check_output(make_args(print_release_url=True), expected_output='urm.nvidia.com')


def _run_and_check_output(args: argparse.Namespace, expected_output: str) -> None:
    """Run the ship_it.main function and capture the output."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        ship_it.main(args)

    assert expected_output in stdout_buffer.getvalue(), (
        f'Ship it did not finish. Missing string: {expected_output}\n'
        f'STDOUT:\n{stdout_buffer.getvalue()}\n'
        f'STDERR:\n{stderr_buffer.getvalue()}')
