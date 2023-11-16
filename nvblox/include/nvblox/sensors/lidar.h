/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include "nvblox/core/types.h"

namespace nvblox {

/// Helper class for handling input LIDAR pointclouds and storing the LIDAR
/// intrinsics. This helps convert a LIDAR pointcloud into a depth image.
class Lidar {
 public:
  Lidar() = delete;

  /// LiDAR constructor. This constructor assumes beams distributed evenly
  /// around zero.
  /// @param num_azimuth_divisions Number of samples per rotation for each beam.
  /// @param num_elevation_divisions Number of samples in azimuth per beam.
  /// @param vertical_fov_rad The angular distance in elevation between the top
  /// and bottom beam.
  __host__ __device__ inline Lidar(int num_azimuth_divisions,
                                   int num_elevation_divisions,
                                   float vertical_fov_rad);
  /// LiDAR constructor. This constructor does not assume beams distributed
  /// evenly around zero. If you have evenly distributed beams use the
  /// constructor above.
  /// @param num_azimuth_divisions Number of samples per rotation for each beam.
  /// @param num_elevation_divisions Number of samples in azimuth per beam.
  /// @param min_angle_below_zero_elevation_rad The angle below zero of the
  /// lowest beam (specified as a positive number).
  /// @param max_angle_above_zero_elevation_rad The angle above zero of the
  /// highest beam (specified as a positive number).
  __host__ __device__ inline Lidar(int num_azimuth_divisions,
                                   int num_elevation_divisions,
                                   float min_angle_below_zero_elevation_rad,
                                   float max_angle_above_zero_elevation_rad);
  __host__ __device__ inline ~Lidar() = default;

  /// Projects a 3D point to the (floating-point) image plane
  __host__ __device__ inline bool project(const Vector3f& p_C,
                                          Vector2f* u_C) const;

  /// Projects a 3D point to the (index-based) image plane
  __host__ __device__ inline bool project(const Vector3f& p_C,
                                          Index2D* u_C) const;

  /// Gets the depth of a point
  __host__ __device__ inline float getDepth(const Vector3f& p_C) const;

  /// Back projection (image plane point to 3D point)
  __host__ __device__ inline Vector3f unprojectFromImagePlaneCoordinates(
      const Vector2f& u_C, const float depth) const;
  __host__ __device__ inline Vector3f unprojectFromPixelIndices(
      const Index2D& u_C, const float depth) const;

  /// Back projection (image plane point to ray)
  /// NOTE(alexmillane): These return normalized vectors
  __host__ __device__ inline Vector3f vectorFromImagePlaneCoordinates(
      const Vector2f& u_C) const;
  __host__ __device__ inline Vector3f vectorFromPixelIndices(
      const Index2D& u_C) const;

  /// Conversions between pixel indices and image plane coordinates
  __host__ __device__ inline Vector2f pixelIndexToImagePlaneCoordsOfCenter(
      const Index2D& u_C) const;
  __host__ __device__ inline Index2D imagePlaneCoordsToPixelIndex(
      const Vector2f& u_C) const;

  /// View
  __host__ inline AxisAlignedBoundingBox getViewAABB(
      const Transform& T_L_C,
      const float,  // for compatibility with camera interface
      const float max_depth) const;

  __host__ __device__ inline int num_azimuth_divisions() const;
  __host__ __device__ inline int num_elevation_divisions() const;
  __host__ __device__ inline float vertical_fov_rad() const;
  __host__ __device__ inline float start_polar_angle_rad() const;
  __host__ __device__ inline int numel() const;
  __host__ __device__ inline int rows() const;
  __host__ __device__ inline int cols() const;

  /// Equality
  __host__ inline friend bool operator==(const Lidar& lhs, const Lidar& rhs);

  /// Hash
  struct Hash {
    __host__ inline size_t operator()(const Lidar& lidar) const;
  };

 private:
  // Core parameters
  int num_azimuth_divisions_;
  int num_elevation_divisions_;
  float vertical_fov_rad_;
  float start_polar_angle_rad_;

  // Dependent parameters
  float start_azimuth_angle_rad_;
  float elevation_pixels_per_rad_;
  float azimuth_pixels_per_rad_;
  float rads_per_pixel_elevation_;
  float rads_per_pixel_azimuth_;
};

// Equality
__host__ inline bool operator==(const Lidar& lhs, const Lidar& rhs);

// Stream LiDAR as text
__host__ inline std::ostream& operator<<(std::ostream& os, const Lidar& camera);

}  // namespace nvblox

#include "nvblox/sensors/internal/impl/lidar_impl.h"
