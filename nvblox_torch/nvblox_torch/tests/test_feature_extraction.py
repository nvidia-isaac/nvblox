# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from unittest.mock import Mock, patch

import torch

from nvblox_torch.examples.utils.feature_extraction import RadioFeatureExtractor
from nvblox_torch.constants import constants


def test_radio_hub_load_explicitly_trusts_the_fixed_nvidia_repository() -> None:
    """A clean Torch Hub cache must not prompt during unattended feature mapping."""
    model = Mock()
    with patch.object(torch.hub, 'load', return_value=model) as load_model:
        with patch.object(constants, 'feature_array_num_elements', return_value=1536):
            extractor = RadioFeatureExtractor()

    load_model.assert_called_once_with('NVlabs/RADIO',
                                       'radio_model',
                                       version='e-radio_v2',
                                       progress=True,
                                       pretrained=True,
                                       skip_validation=True,
                                       trust_repo=True)
    assert extractor.model is model
    model.cuda.assert_called_once_with()
    model.cuda.return_value.eval.assert_called_once_with()
