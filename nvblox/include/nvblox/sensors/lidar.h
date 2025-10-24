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
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/sensor.h"

namespace nvblox {

/// Helper class for handling input LIDAR pointclouds and storing the LIDAR
/// intrinsics. This helps convert a LIDAR pointcloud into a depth image.
class Lidar : public SensorBase {
 public:
  __host__ __device__ inline Lidar() = default;
  __host__ __device__ inline ~Lidar() = default;

  /// LiDAR constructor. This constructor assumes beams distributed evenly
  /// around zero.
  /// @param num_azimuth_divisions Number of samples per rotation for each beam.
  /// @param num_elevation_divisions Number of samples in azimuth per beam.
  /// @param min_valid_range_m Minimum valid range value in meters.
  /// @param max_valid_range_m Maximum valid range value in meters.
  /// @param vertical_fov_rad The angular distance in elevation between the top
  /// and bottom beam.
  __host__ __device__ inline Lidar(int num_azimuth_divisions,
                                   int num_elevation_divisions,
                                   float min_valid_range_m,
                                   float max_valid_range_m,
                                   float vertical_fov_rad);
  /// LiDAR constructor. This constructor does not assume beams distributed
  /// evenly around zero. If you have evenly distributed beams use the
  /// constructor above.
  /// @param num_azimuth_divisions Number of samples per rotation for each beam.
  /// @param num_elevation_divisions Number of samples in azimuth per beam.
  /// @param min_valid_range_m Minimum valid range value in meters.
  /// @param max_valid_range_m Minimum valid range value in meters.
  /// @param min_angle_below_zero_elevation_rad The angle below zero of the
  /// lowest beam (specified as a positive number).
  /// @param max_angle_above_zero_elevation_rad The angle above zero of the
  /// highest beam (specified as a positive number).
  __host__ __device__ inline Lidar(int num_azimuth_divisions,
                                   int num_elevation_divisions,
                                   float min_valid_range_m,
                                   float max_valid_range_m,
                                   float min_angle_below_zero_elevation_rad,
                                   float max_angle_above_zero_elevation_rad);

  /// Returns if the point is in the valid range of the lidar
  __host__ __device__ inline bool isInValidRange(const Vector3f& p_C) const;

  /// Projects a 3D point to the (floating-point) image plane
  __host__ __device__ inline bool project(
      const Vector3f& p_C, Vector2f* u_C, const float unused = -1.F,
      const bool check_viewport = true) const;

  /// Projects a 3D point to the (index-based) image plane
  __host__ __device__ inline bool project(
      const Vector3f& p_C, Index2D* u_C, const float unused = -1.F,
      const bool check_viewport = true) const;

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

  /// Interpolation that takes the sparsity of a lidar-generated depth image
  /// into account.
  __host__ __device__ inline bool interpolateDepthImage(
      const DepthImageConstView frame, const Vector2f& u_px,
      const Vector3f& p_voxel_center_C, const float voxel_size,
      float* image_value, Index2D* u_px_closest_ptr = nullptr) const;

  __host__ __device__ inline int num_azimuth_divisions() const;
  __host__ __device__ inline int num_elevation_divisions() const;
  __host__ __device__ inline float min_valid_range_m() const;
  __host__ __device__ inline float max_valid_range_m() const;
  __host__ __device__ inline float vertical_fov_rad() const;
  __host__ __device__ inline float start_polar_angle_rad() const;
  __host__ __device__ inline int numel() const;
  __host__ __device__ inline int rows() const;
  __host__ __device__ inline int cols() const;
  __host__ __device__ inline int height() const;
  __host__ __device__ inline int width() const;

  /// Get the sensor modality identifier
  /// @return The sensor modality (kLidar).
  __host__ __device__ static constexpr SensorModality sensor_modality() {
    return SensorModality::kLidar;
  }

  /// A parameter getter
  __host__ __device__ float linear_interpolation_max_allowable_difference_vox()
      const {
    return linear_interpolation_max_allowable_difference_vox_;
  }

  /// A parameter setter
  __host__ __device__ void linear_interpolation_max_allowable_difference_vox(
      const float value) {
    linear_interpolation_max_allowable_difference_vox_ = value;
  }

  /// A parameter getter
  __host__ __device__ float
  nearest_interpolation_max_allowable_dist_to_ray_vox() const {
    return nearest_interpolation_max_allowable_dist_to_ray_vox_;
  }

  /// A parameter setter
  __host__ __device__ void nearest_interpolation_max_allowable_dist_to_ray_vox(
      const float value) {
    nearest_interpolation_max_allowable_dist_to_ray_vox_ = value;
  }

  /// Equality
  __host__ inline friend bool operator==(const Lidar& lhs, const Lidar& rhs);

  /// Hash
  struct Hash {
    __host__ inline size_t operator()(const Lidar& lidar) const;
  };

 private:
  // Core parameters
  int num_azimuth_divisions_ = 0;
  int num_elevation_divisions_ = 0;
  float min_valid_range_m_ = 0.F;
  float max_valid_range_m_ = 0.F;
  float vertical_fov_rad_ = 0.F;
  float start_polar_angle_rad_ = 0.F;

  // Dependent parameters
  float start_azimuth_angle_rad_ = 0.F;
  float elevation_pixels_per_rad_ = 0.F;
  float azimuth_pixels_per_rad_ = 0.F;
  float rads_per_pixel_elevation_ = 0.F;
  float rads_per_pixel_azimuth_ = 0.F;

  // Interpolation parameters
  float linear_interpolation_max_allowable_difference_vox_ = 2.0f;
  float nearest_interpolation_max_allowable_dist_to_ray_vox_ = 0.5f;
};

// Equality
__host__ inline bool operator==(const Lidar& lhs, const Lidar& rhs);

// Stream LiDAR as text
__host__ inline std::ostream& operator<<(std::ostream& os, const Lidar& camera);

}  // namespace nvblox

#include "nvblox/sensors/internal/impl/lidar_impl.h"
