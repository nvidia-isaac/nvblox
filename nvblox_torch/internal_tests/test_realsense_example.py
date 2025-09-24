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

import numpy as np
from unittest.mock import patch
import io
import sys
import pytest

from contextlib import redirect_stdout
from typing import Any
import torch

xfail = pytest.mark.xfail

MAX_STEPS = 10
WIDTH = 640
HEIGHT = 640


class MockIntrinsics:
    """Mock class for realsense intrinsics."""

    def __init__(self, _: Any = None) -> None:
        self.fx = 320.0
        self.fy = 320.0
        self.ppx = 160.0
        self.ppy = 160.0
        self.width = WIDTH
        self.height = HEIGHT


class MockExtrinsics:
    """Mock class for realsense extrinsics."""

    def __init__(self, _: Any = None) -> None:
        self.rotation = np.eye(3, 3)
        self.translation = np.zeros(3)


class MockRealsenseDataloader:
    """Mock class for realsense dataloader."""

    def __init__(self, max_steps: int) -> None:
        assert max_steps == MAX_STEPS
        self.max_steps = max_steps
        self.current_step = 0

    def __len__(self) -> int:
        return self.max_steps

    def __iter__(self) -> 'MockRealsenseDataloader':
        return self

    def __next__(self) -> dict:
        self.current_step += 1
        print(f'current_step: {self.current_step}')
        if self.current_step >= self.max_steps:
            raise StopIteration

        return {
            'left_infrared_image': 255 * np.random.randint(0, 256, (HEIGHT, WIDTH), dtype=np.uint8),
            'right_infrared_image': 255 * np.random.randint(0, 256,
                                                            (HEIGHT, WIDTH), dtype=np.uint8),
            'depth': torch.tensor(np.random.rand(HEIGHT, WIDTH), dtype=torch.float32,
                                  device='cuda'),
            'rgb': torch.randint(low=0,
                                 high=256,
                                 size=(HEIGHT, WIDTH, 3),
                                 dtype=torch.uint8,
                                 device='cuda'),
            'timestamp': self.current_step
        }

    def left_infrared_intrinsics(self) -> MockIntrinsics:
        return MockIntrinsics()

    def right_infrared_intrinsics(self) -> MockIntrinsics:
        return MockIntrinsics()

    def depth_intrinsics(self) -> MockIntrinsics:
        return MockIntrinsics()

    def color_intrinsics(self) -> MockIntrinsics:
        return MockIntrinsics()

    # pylint: disable=C0103
    def T_C_left_infrared_C_color(self) -> MockExtrinsics:
        return MockExtrinsics()

    def T_C_left_infrared_C_right_infrared(self) -> MockExtrinsics:
        return MockExtrinsics()


@patch('nvblox_torch.examples.realsense.run_realsense_mapper.RealsenseDataloader',
       MockRealsenseDataloader)
@patch('nvblox_torch.examples.realsense.realsense_dataloader.rs.pyrealsense2.intrinsics',
       MockIntrinsics)
@patch('nvblox_torch.examples.realsense.realsense_dataloader.rs.pyrealsense2.extrinsics',
       MockExtrinsics)
@xfail(run=False,
       reason='''This test requres a dedicated docker image and is therefore disabled per default.
    To enable it, run "pytest --runxfail''')
def test_realsense_example() -> None:
    # This import will fail outside the realsense docker. Since the test is marked with "xfail" it
    # will not break the non-realsense test suite
    # pylint: disable=import-outside-toplevel
    from nvblox_torch.examples.realsense import run_realsense_mapper

    # We pass/get CLI input/output by:
    # - Pass CLI args by using the unittest.mock.patch.
    # - Redirect stdout to a buffer for inspection.
    test_args = [
        'run_realsense_mapper.py',
        '--max_frames',
        str(MAX_STEPS),
    ]
    buffer = io.StringIO()
    with patch.object(sys, 'argv', test_args):
        with redirect_stdout(buffer):
            assert run_realsense_mapper.main() == 0
    assert 'Done' in buffer.getvalue()
