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

#include "nvblox/core/image.h"
#include "nvblox/core/types.h"

namespace nvblox {

// NOTE(jjiao):
// image coordinates to pixel indices
// return u_C.array().floor().cast<int>();
// image angle order:
// from left to right: 0       -> 2pi
// fron bottom to top: phi_min -> phi_max
class OSLidar {
 public:
  __host__ __device__ inline OSLidar() = default;
  __host__ __device__ inline OSLidar(int num_azimuth_divisions,
                                     int num_elevation_divisions,
                                     float horizontal_fov_rad,
                                     float vertical_fov_rad,
                                     DepthImage* depth_image_ptr,
                                     DepthImage* z_image_ptr);
  __host__ __device__ inline ~OSLidar() = default;

  // TODO(jjiao): This function is used to check whether p_C is projected on the
  // the image plane, or outside the OSLidar's FOV
  // Projects a 3D point to the (floating-point) image plane
  __host__ __device__ inline bool project(const Vector3f& p_C,
                                          Vector2f* u_C) const;

  // Projects a 3D point to the (index-based) image plane
  __host__ __device__ inline bool project(const Vector3f& p_C,
                                          Index2D* u_C) const;

  // Gets the depth of a point
  __host__ __device__ inline float getDepth(const Vector3f& p_C) const;

  // Back projection (image plane point to 3D point)
  // NOTE(jjiao): for OUSTER lidars, 3D points cannot be calculated
  // directly by angles. They should be retrieved from the coord_image_ptr
  __host__ __device__ inline Vector3f unprojectFromImagePlaneCoordinates(
      const Vector2f& u_C, const float depth) const;
  __host__ __device__ inline Vector3f unprojectFromPixelIndices(
      const Index2D& u_C, const float depth) const;

  // Back projection (image plane point to ray)
  __host__ __device__ inline Vector3f vectorFromImagePlaneCoordinates(
      const Vector2f& u_C) const;
  __host__ __device__ inline Vector3f vectorFromPixelIndices(
      const Index2D& u_C) const;

  // Conversions between pixel indices and image plane coordinates
  __host__ __device__ inline Vector2f pixelIndexToImagePlaneCoordsOfCenter(
      const Index2D& u_C) const;
  __host__ __device__ inline Index2D imagePlaneCoordsToPixelIndex(
      const Vector2f& u_C) const;

  // View
  __host__ inline AxisAlignedBoundingBox getViewAABB(
      const Transform& T_L_C, const float min_depth,
      const float max_depth) const;

  __host__ __device__ inline int num_azimuth_divisions() const;
  __host__ __device__ inline int num_elevation_divisions() const;
  __host__ __device__ inline float vertical_fov_rad() const;
  __host__ __device__ inline int numel() const;
  __host__ __device__ inline int rows() const;
  __host__ __device__ inline int cols() const;

  // Equality
  __host__ inline friend bool operator==(const OSLidar& lhs,
                                         const OSLidar& rhs);

  // Hash
  struct Hash {
    __host__ inline size_t operator()(const OSLidar& OSLidar) const;
  };

 private:
  // Core parameters
  int num_azimuth_divisions_;
  int num_elevation_divisions_;
  float horizontal_fov_rad_;
  float vertical_fov_rad_;

  // Dependent parameters
  float start_polar_angle_rad_;
  float start_azimuth_angle_rad_;
  float elevation_pixels_per_rad_;
  float azimuth_pixels_per_rad_;
  float rads_per_pixel_elevation_;
  float rads_per_pixel_azimuth_;

  DepthImage* depth_image_ptr_;
  DepthImage* z_image_ptr_;
};

// Equality
__host__ inline bool operator==(const OSLidar& lhs, const OSLidar& rhs);

}  // namespace nvblox

#include "nvblox/core/impl/oslidar_impl.h"
