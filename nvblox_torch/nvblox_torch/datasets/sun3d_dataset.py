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
import os

import imageio.v3
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.dataset import Dataset
from transforms3d.quaternions import quat2mat
from scipy.spatial.transform import Rotation
from typing import List, Dict, Any

from nvblox_torch.sensor import Sensor


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that handles Sensor objects.

    The Sensor object cannot be batched by default_collate, so we handle it separately.

    Args:
        batch: List of dictionaries from the dataset

    Returns:
        Dictionary with batched tensors and list of Sensor objects
    """
    # Extract sensor objects separately
    sensors = [item.pop('sensor') for item in batch]

    # Use default collate for the remaining tensors
    collated = default_collate(batch)

    # Add sensors back as a list
    collated['sensor'] = sensors

    return collated


class Sun3dDataset(Dataset):
    """Sun3d dataset for testing and evaluation.
    """

    def __init__(self, root_dir: str, sequence_name: str, device: str = 'cuda') -> None:
        super().__init__()
        self.root = root_dir
        self.device = device

        # Load all floats from camera intrinsics file
        # File has variable columns per row (3x3 matrix, then single values for distortion)
        intrinsics_path = os.path.join(self.root, 'camera-intrinsics.txt')

        self.sequence_name = sequence_name
        self.seq_dir = os.path.join(self.root, self.sequence_name)

        self.frame_names = list(sorted({f.split('.')[0] for f in os.listdir(self.seq_dir)}))

        # Load first frame to determine image dimensions and create sensor
        first_frame_name = self.frame_names[0]
        first_rgb_np = self._load_color(first_frame_name)
        height, width = first_rgb_np.shape[:2]

        # Create sensor object from camera intrinsics and distortion coefficients
        self.sensor = Sensor.from_file(intrinsics_path, width, height)

    @staticmethod
    def create_dataloader(root_dir: str, sequence_name: str) -> DataLoader:
        """Create a dataloader for the SUN3D dataset."""
        return DataLoader(Sun3dDataset(root_dir=root_dir, sequence_name=sequence_name),
                          batch_size=1,
                          shuffle=False,
                          num_workers=0,
                          collate_fn=collate_batch)

    def __len__(self) -> int:
        return len(self.frame_names)

    def _load_color(self, frame_name: str) -> np.ndarray:
        """Load color image from disk."""
        return imageio.imread(os.path.join(self.seq_dir, f'{frame_name}.color.png'))

    def _load_depth(self, frame_name: str) -> np.ndarray:
        """Load depth image from disk and convert to meters."""
        depth_np = imageio.imread(os.path.join(self.seq_dir, f'{frame_name}.depth.png'))
        return depth_np.astype(np.float32) / 1000

    def _load_pose(self, frame_name: str) -> np.ndarray:
        """Load pose from disk."""
        pose_params = np.loadtxt(os.path.join(self.seq_dir, f'{frame_name}.pose.txt'))
        if pose_params.shape == (4, 4):
            return torch.tensor(pose_params, dtype=torch.float32)
        elif pose_params.shape == (7, ):
            translation = pose_params[:3]
            quaternion = pose_params[3:]
            w, x, y, z = quaternion
            rotation_mx = Rotation.from_quat([x, y, z, w]).as_matrix()
            pose = np.eye(4)
            pose[:3, :3] = rotation_mx
            pose[:3, 3] = translation
            return torch.tensor(pose, dtype=torch.float32).reshape(4, 4)
        else:
            raise ValueError(f'Invalid pose shape: {pose_params.shape}')

    def __getitem__(self, index: int) -> dict:
        """rgba: HxWx4, depth HxWx1"""
        # Load the raw data
        frame_name = self.frame_names[index]
        rgb_np = self._load_color(frame_name)
        depth_np = self._load_depth(frame_name)
        pose_np = self._load_pose(frame_name)

        # Color
        rgb = torch.tensor(rgb_np, device=self.device)

        # Depth
        depth = torch.tensor(depth_np, device=self.device)
        depth = depth.squeeze()
        depth = depth.unsqueeze(dim=-1)

        # Pose
        # Conversion in nvblox:
        # Rotate the world frame since Y is up in the normal 3D match datasets.
        # Eigen::Quaternionf q_L_O = Eigen::Quaternionf::FromTwoVectors(Vector3f(0, 1, 0),
        # Vector3f(0, 0, 1));
        pose = torch.tensor(pose_np, device=self.device, dtype=torch.float32)
        eigen_quat = [0.707106769, 0.707106769, 0, 0]
        sun3d_to_nvblox_T = torch.eye(4, device=self.device, dtype=torch.float32)
        sun3d_to_nvblox_T[:3, :3] = torch.tensor(quat2mat(eigen_quat), device=self.device)
        nvblox_pose = sun3d_to_nvblox_T @ pose

        # Post-conditions
        assert rgb.shape[-1] == 3, 'Only 3-channel RGB images supported by nvblox'
        assert depth.shape[-1] == 1, 'Only 1-channel depth images supported by nvblox'
        assert rgb.dtype == torch.uint8, 'Only 8-bit RGB images supported'
        assert depth.dtype == torch.float, 'CPP-side conversions assume 32-bit float tensors'
        assert pose.dtype == torch.float, 'CPP-side conversions assume 32-bit float tensors'

        return {
            'rgb': rgb,
            'depth': depth,
            'pose': nvblox_pose,
            'sensor': self.sensor,
        }
