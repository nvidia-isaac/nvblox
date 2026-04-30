# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Run fuse_3dmatch multiple times on the test dataset"""

import os
import subprocess

BINARY_PATH = 'build/executables/fuse_3dmatch'

TIMEOUT_S = 120


def test_run_fuser_multiple_times() -> None:
    """Run fuse_3dmatch multiple times on the test dataset"""
    assert os.path.isfile(BINARY_PATH), f'{BINARY_PATH} not found'

    for _ in range(0, 100):
        output = subprocess.check_output([f'{BINARY_PATH}', 'executables/tests/data/3dmatch'],
                                         timeout=TIMEOUT_S)

        print(output.decode('utf-8'))


if __name__ == '__main__':
    test_run_fuser_multiple_times()
