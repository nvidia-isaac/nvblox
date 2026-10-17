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
"""Sensor classes for nvblox_torch integration.

This module provides a unified Sensor interface that wraps different sensor types
(Camera, Lidar) using nvblox's TypeIndexedStore infrastructure.
"""

from typing import Optional
import torch
import numpy as np

from nvblox_torch.lib.utils import get_nvblox_torch_class


class Sensor:
    """Unified sensor class that can represent Camera or Lidar sensors.

    This provides a single entry point for all sensor types, using factory methods
    to create the appropriate internal representation. The C++ implementation uses
    nvblox::TypeIndexedStore to dispatch to the correct sensor type at runtime.

    Examples:
        # Create a pinhole camera
        camera = Sensor.from_camera(fu=525.0, fv=525.0, cu=320.0, cv=240.0,
                                    width=640, height=480)

        # Create a camera with distortion using from_camera_matrix
        K = torch.tensor([[1000.0, 0.0, 960.0],
                          [0.0, 1000.0, 540.0],
                          [0.0, 0.0, 1.0]], dtype=torch.float32)
        radial_dist = torch.tensor([-0.2, 0.1, -0.01, 0.0, 0.0, 0.0], dtype=torch.float32)
        tangential_dist = torch.tensor([0.001, -0.001], dtype=torch.float32)
        camera_distorted = Sensor.from_camera_matrix(K, 1920, 1080, radial_dist, tangential_dist)

        # Create a lidar sensor
        lidar = Sensor.from_lidar(num_azimuth_divisions=1800,
                                  num_elevation_divisions=16,
                                  vertical_fov_rad=0.524, min_valid_range_m=0.5)
    """

    def __init__(self, c_sensor: torch.classes.Sensor) -> None:
        """Internal constructor - use factory methods instead.

        Args:
            c_sensor: The wrapped C++ PySensor object.
        """
        self._c_sensor = c_sensor

    @classmethod
    def from_camera(cls, fu: float, fv: float, cu: float, cv: float, width: int,
                    height: int) -> 'Sensor':
        """Create a pinhole camera sensor (no distortion).

        Args:
            fu: Focal length in pixels (x-direction).
            fv: Focal length in pixels (y-direction).
            cu: Principal point x-coordinate.
            cv: Principal point y-coordinate.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Sensor object wrapping a Camera.
        """
        c_sensor = get_nvblox_torch_class('Sensor').from_camera(fu, fv, cu, cv, width, height)
        return cls(c_sensor)

    @classmethod
    def from_camera_matrix(cls,
                           intrinsics: torch.Tensor,
                           width: int,
                           height: int,
                           radial_distortion: Optional[torch.Tensor] = None,
                           tangential_distortion: Optional[torch.Tensor] = None) -> 'Sensor':
        """Create a camera from intrinsics matrix and optional distortion coefficients.

        Args:
            intrinsics (3, 3; device=CPU): Camera intrinsics matrix.
            width: Image width in pixels.
            height: Image height in pixels.
            radial_distortion (6; device=CPU): Optional radial distortion coefficients:
                                                    [k1, k2, k3, k4, k5, k6]
            tangential_distortion (2; device=CPU): Optional tangential distortion: [p1, p2].

        Returns:
            Sensor object wrapping a Camera (with or without distortion).
        """
        fu = float(intrinsics[0, 0].item())
        fv = float(intrinsics[1, 1].item())
        cu = float(intrinsics[0, 2].item())
        cv = float(intrinsics[1, 2].item())

        # Check if any distortion is provided
        no_distortion = radial_distortion is None and tangential_distortion is None

        if no_distortion:
            return cls.from_camera(fu, fv, cu, cv, width, height)

        # Parse radial distortion coefficients (3 or 6 elements)
        if radial_distortion is not None:
            assert len(radial_distortion) == 6, 'Radial distortion must be a 6-element tensor'
            k1 = float(radial_distortion[0].item())
            k2 = float(radial_distortion[1].item())
            k3 = float(radial_distortion[2].item())
            k4 = float(radial_distortion[3].item())
            k5 = float(radial_distortion[4].item())
            k6 = float(radial_distortion[5].item())

        # Parse tangential distortion coefficients (2 elements)
        p1 = p2 = 0.0
        if tangential_distortion is not None:
            assert len(
                tangential_distortion) == 2, 'Tangential distortion must be a 2-element tensor'
            p1 = float(tangential_distortion[0].item())
            p2 = float(tangential_distortion[1].item())

        c_sensor = get_nvblox_torch_class('Sensor').from_camera_distorted(
            fu, fv, cu, cv, width, height, k1, k2, k3, k4, k5, k6, p1, p2)
        return cls(c_sensor)

    @classmethod
    def from_lidar(cls,
                   num_azimuth_divisions: int,
                   num_elevation_divisions: int,
                   vertical_fov_rad: float,
                   min_valid_range_m: float = 0.001) -> 'Sensor':
        """Create a lidar sensor.

        Args:
            num_azimuth_divisions: Number of azimuth divisions (horizontal resolution).
            num_elevation_divisions: Number of elevation divisions (vertical channels).
            vertical_fov_rad: Vertical field of view in radians.
            min_valid_range_m: Minimum valid range in meters.

        Returns:
            Sensor object wrapping a Lidar.
        """
        c_sensor = get_nvblox_torch_class('Sensor').from_lidar(num_azimuth_divisions,
                                                               num_elevation_divisions,
                                                               vertical_fov_rad, min_valid_range_m)
        return cls(c_sensor)

    @classmethod
    def from_lidar_asymmetric(cls,
                              num_azimuth_divisions: int,
                              num_elevation_divisions: int,
                              min_angle_below_zero_elevation_rad: float,
                              max_angle_above_zero_elevation_rad: float,
                              min_valid_range_m: float = 0.001) -> 'Sensor':
        """Create a lidar sensor with an asymmetric elevation envelope.

        Args:
            num_azimuth_divisions: Number of azimuth divisions (horizontal resolution).
            num_elevation_divisions: Number of elevation divisions (vertical channels).
            min_angle_below_zero_elevation_rad: Angle below zero of the lowest beam (positive value).
            max_angle_above_zero_elevation_rad: Angle above zero of the highest beam (positive value).
            min_valid_range_m: Minimum valid range in meters.

        Returns:
            Sensor object wrapping a Lidar.
        """
        c_sensor = get_nvblox_torch_class('Sensor').from_lidar_asymmetric(
            num_azimuth_divisions, num_elevation_divisions,
            min_angle_below_zero_elevation_rad, max_angle_above_zero_elevation_rad,
            min_valid_range_m)
        return cls(c_sensor)

    @property
    def modality(self) -> str:
        """Get the sensor modality.

        Returns:
            String describing the sensor type.
        """
        return self._c_sensor.get_sensor_modality()

    @property
    def width(self) -> int:
        """Get sensor width.

        Returns:
            Width in pixels or divisions.
        """
        return self._c_sensor.width()

    @property
    def height(self) -> int:
        """Get sensor height.

        Returns:
            Height in pixels or divisions.
        """
        return self._c_sensor.height()

    def get_c_sensor(self) -> torch.classes.Sensor:
        """Get the wrapped C++ sensor object."""
        return self._c_sensor

    @classmethod
    def from_file(cls, intrinsics_path: str, width: int, height: int) -> 'Sensor':
        """Create a sensor from a file.

        Args:
            intrinsics_path: Path to the intrinsics file.

        Returns:
            Sensor object wrapping a Camera.
        """
        params = []
        with open(intrinsics_path, 'r', encoding='utf-8') as f:
            for line in f:
                params.extend([float(x) for x in line.split()])
        params = np.array(params)

        # First 9 values are the 3x3 camera intrinsics matrix
        camera_intrinsics = torch.from_numpy(params[:9].reshape(3, 3)).float()

        # Remaining values (if present) are distortion: k1-k6 (radial), p1-p2 (tangential)
        radial_distortion = torch.from_numpy(params[9:15]).float() if len(params) >= 15 else None
        tangential_distortion = torch.from_numpy(
            params[15:17]).float() if len(params) >= 17 else None

        print(camera_intrinsics)
        print(radial_distortion)
        print(tangential_distortion)

        return cls.from_camera_matrix(camera_intrinsics, width, height, radial_distortion,
                                      tangential_distortion)
