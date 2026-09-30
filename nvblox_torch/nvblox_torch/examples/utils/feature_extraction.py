# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import math
from typing import Tuple, Optional
import torch

from nvblox_torch.constants import constants


def get_random_projection_matrix(input_channel_num: int,
                                 output_channel_num: int,
                                 device: str,
                                 preserve_norm: bool = True) -> torch.Tensor:
    """Get a matrix that projects from an input space to an output space.

    Projection should right multiply the input tensor, such that:

        output = input @ projection_matrix

    Projection is implemented as a random projection which (by default)
    attempts to preserve the norm of the input vectors.

    Args:
        input_channel_num (int): Dimension of the input space
        output_channel_num (int): Dimension of the output space
        device (str): torch device
        preserve_norm (bool): Whether to preserve the norm of the input vectors. Default is True.

    Returns:
        torch.Tensor: A random projection matrix
                      of shape (initial_channel_num, output_channel_num)
    """
    # Uniform random matrix with elements in the range [-1, 1]
    projection_matrix = torch.rand(input_channel_num, output_channel_num, device=device) * 2.0 - 1.0
    if preserve_norm:
        # NOTE(alexmillane): The variance of the a uniform random variable between
        # -1 and 1 is 1/3 leading to the appearance of a factor of sqrt(3) in the
        # normalization.
        projection_matrix = projection_matrix * math.sqrt(3.0 / output_channel_num)
    return projection_matrix


class RadioFeatureExtractor:
    """
    Feature extractor using the RADIO model.

    Extracts visual features from RGB images using a pretrained RADIO model.
    Supports optional upscaling of input images to preserve details.
    """

    def __init__(self, upscale_n_times: int = 1):
        """Initialize the RADIO feature extractor.

        Args:
            upscale_n_times: Factor to upscale input images before feature extraction
        """
        self.upscale_n_times = upscale_n_times
        self.radio_embedding_dim = 1536
        self.model = torch.hub.load(
            'NVlabs/RADIO',
            'radio_model',
            version='e-radio_v2',    # embedding dim = 1536
            progress=True,
            pretrained=True,
            skip_validation=True,
        # Trust only this fixed NVIDIA repository; avoid interactive Hub prompts.
            trust_repo=True)
        self.model.cuda().eval()

        if self.radio_embedding_dim > constants.feature_array_num_elements():
            print('The embedding dimension of the RADIO model is larger than the maximum '
                  'feature array size that nvblox supports: '
                  f'{constants.feature_array_num_elements()}. The feature extractor will '
                  'randomly project the output of the model to a lower dimension.')
            self.projection_matrix = get_random_projection_matrix(
                self.radio_embedding_dim, constants.feature_array_num_elements(), device='cuda')
        else:
            self.projection_matrix = None

    def embedding_dim(self) -> int:
        """Get the dimension of the feature embeddings."""
        if self.projection_matrix is not None:
            return constants.feature_array_num_elements()
        else:
            return self.radio_embedding_dim

    def compute(
        self,
        rgb: torch.Tensor,
        desired_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features from an RGB image.

        Args:
            rgb: Input RGB image tensor of shape (H, W, 3)
            desired_size: Optional target output size (H, W)

        Returns:
            Tuple containing:
                - features: Feature tensor of shape (H, W, F) where F is the feature dimension
        """
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3

        # Optional upscaling of the input
        if self.upscale_n_times is not None:
            upscale_size = (rgb.shape[0] * self.upscale_n_times,
                            rgb.shape[1] * self.upscale_n_times)
            rgb, _ = self._upscale_image(rgb.float(), upscale_size)

        # Extract features at model's required resolution
        features = self._extract_features(rgb)

        # Nvblox feature mapper expects a dimension of a certain size
        features = self._get_zero_padded_features(features)

        # Resize features back to input resolution if no desired size specified
        if desired_size is None:
            desired_size = rgb.shape[0:2]

        # Resize features to desired size
        features = self._upscale_image(features, desired_size)[0]

        if self.projection_matrix is not None:
            features = features @ self.projection_matrix

        return features

    def _extract_features(self, rgb_bhw3: torch.Tensor) -> torch.Tensor:
        """Extract features using the RADIO model."""
        # Reshape input for model
        rgb_b3hw = rgb_bhw3.permute(2, 0, 1)

        # Normalize to [0, 1]
        if torch.max(rgb_b3hw) > 1.0:
            rgb_b3hw = rgb_b3hw / 255.0

        # Get model's required resolution
        nearest_res = self.model.get_nearest_supported_resolution(*rgb_b3hw.shape[-2:])
        assert rgb_b3hw.shape[1] == nearest_res[0]
        assert rgb_b3hw.shape[2] == nearest_res[1]

        # Compute features
        _, features = self.model(rgb_b3hw.unsqueeze(0))

        # Reshape output into image
        output_size_height = int(nearest_res[0] / self.model.patch_size)
        output_size_width = int(nearest_res[1] / self.model.patch_size)
        return features.view(output_size_height, output_size_width, -1)

    def _upscale_image(self,
                       tensor: torch.Tensor,
                       target_size: Tuple[int, int],
                       mode: str = 'bilinear') -> Tuple[torch.Tensor, Tuple[float, float]]:
        """
        Upscale a tensor to the specified target size.

        Args:
            tensor: Input tensor of shape (H, W, C)
            target_size: Desired output size (H, W)
            mode: Interpolation mode

        Returns:
            Tuple containing:
                - Upscaled tensor of shape (H_new, W_new, C)
                - Scale factors (scale_h, scale_w)
        """
        scale_h = target_size[0] / tensor.shape[0]
        scale_w = target_size[1] / tensor.shape[1]

        # Convert from [H, W, C] to [N, C, H, W] for interpolation
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(tensor,
                                                 size=target_size,
                                                 mode=mode,
                                                 align_corners=False)
        # Convert back to [H, W, C]
        tensor = tensor.squeeze(0).permute(1, 2, 0)
        return tensor, (scale_h, scale_w)

    def _get_zero_padded_features(self, features: torch.Tensor) -> torch.Tensor:
        """Pad features with zeros to match nvblox's expected feature dimension."""
        feature_side_length_height = features.shape[0]
        feature_side_length_width = features.shape[1]
        zeros = torch.zeros(feature_side_length_height, feature_side_length_width,
                            self.num_excess_features()).to(features.device)
        return torch.cat((features, zeros), dim=2)

    def num_excess_features(self) -> int:
        """Return number of zeros that we need to append to the end of each feature in order to
        comply with nvblox lib"""
        num_excess = constants.feature_array_num_elements() - self.embedding_dim()
        assert num_excess >= 0, (
            f"Embedding dim: {self.embedding_dim()} is more than nvblox's max feature size: "
            f'{constants.feature_array_num_elements()}. Rebuild nvblox with a larger feature size.')
        return num_excess
