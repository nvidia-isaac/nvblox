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
// NOTE(gogojjh):
// image coordinates to pixel indices
// return u_C.array().floor().cast<int>();
// image angle order:
// from left to right: 0       -> 2pi
// fron bottom to top: phi_min -> phi_max
class OSLidar {
 public:
  __host__ __device__ inline OSLidar() = default;
  __host__ __device__ inline OSLidar(
      int num_azimuth_divisions, int num_elevation_divisions,
      float horizontal_fov_rad, float vertical_fov_rad,
      float start_azimuth_angle_rad, float end_azimuth_angle_rad,
      float start_elevation_angle_rad, float end_elevation_angle_rad);
  __host__ __device__ inline ~OSLidar();

  // __host__ __device__ inline void setIntrinsics();
  __host__ __device__ inline void printIntrinsics() const;

  // NOTE(gogojjh): This function is used to check whether p_C (in the camera
  // coordinate) is projected on the the image plane, or outside the
  // OSLidar's FOV Projects a 3D point to the (floating-point) image plane
  __host__ __device__ inline bool project(const Vector3f& p_C,
                                          Vector2f* u_C) const;

  // Projects a 3D point to the (index-based) image plane
  __host__ __device__ inline bool project(const Vector3f& p_C,
                                          Index2D* u_C) const;

  // Gets the depth of a point
  __host__ __device__ inline float getDepth(const Vector3f& p_C) const;

  // Gets the normal vector of a point
  __host__ __device__ inline Vector3f getNormalVector(const Index2D& u_C) const;

  // NOTE(gogojjh): This function is used to unproject a pixel to a 3D point
  // given the 2D coordinate of an image Back projection (image plane point to
  // 3D point) represented in the lidar coordinate system
  __host__ __device__ inline Vector3f unprojectFromImagePlaneCoordinates(
      const Vector2f& u_C, const float depth) const;
  __host__ __device__ inline Vector3f unprojectFromPixelIndices(
      const Index2D& u_C, const float depth) const;
  __host__ __device__ inline Vector3f unprojectFromImageIndex(
      const Index2D& u_C) const;

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

  __host__ __device__ void setDepthFrameCUDA(float* depth_image_ptr_cuda) {
    depth_image_ptr_cuda_ = depth_image_ptr_cuda;
  }

  __host__ __device__ void setHeightFrameCUDA(float* height_image_ptr_cuda) {
    height_image_ptr_cuda_ = height_image_ptr_cuda;
  }

  __host__ __device__ void setNormalFrameCUDA(float* normal_image_ptr_cuda) {
    normal_image_ptr_cuda_ = normal_image_ptr_cuda;
  }

  __host__ __device__ inline float* getDepthFrameCUDA() const {
    return depth_image_ptr_cuda_;
  }

  __host__ __device__ inline float* getHeightFrameCUDA() const {
    return height_image_ptr_cuda_;
  }

  __host__ __device__ inline float* getNormalFrameCUDA() const {
    return normal_image_ptr_cuda_;
  }

  // View
  __host__ inline AxisAlignedBoundingBox getViewAABB(
      const Transform& T_L_C, const float min_depth,
      const float max_depth) const;

  __host__ __device__ inline int num_azimuth_divisions() const;
  __host__ __device__ inline int num_elevation_divisions() const;
  __host__ __device__ inline float horizontal_fov_rad() const;
  __host__ __device__ inline float vertical_fov_rad() const;
  __host__ __device__ inline float rads_per_pixel_elevation() const;
  __host__ __device__ inline float rads_per_pixel_azimuth() const;
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
  float* depth_image_ptr_cuda_;
  float* height_image_ptr_cuda_;
  float* normal_image_ptr_cuda_;

  // Core parameters
  int num_azimuth_divisions_;
  int num_elevation_divisions_;
  float horizontal_fov_rad_;
  float vertical_fov_rad_;

  // Angular distance between pixels
  // Note(alexmillane): Note the difference in division by N vs. (N-1) below.
  // This is because in the azimuth direction there's a wrapping around. The
  // point at pi/-pi is not double sampled, generating this difference.
  float start_azimuth_angle_rad_;
  float end_azimuth_angle_rad_;

  // ********************* elevation_angle
  // ****** the start elevation_angle indicate the direction: x=0, +z
  // ****** the end elevation_angle indicate the direction: x=0, -z
  // ********************* azimuth_angle
  // ****** the start and end azimuth_angle: clockwise
  // -x, y=0 -> +x, y=0
  float start_elevation_angle_rad_;
  float end_elevation_angle_rad_;

  // Dependent parameters
  float elevation_pixels_per_rad_;
  float azimuth_pixels_per_rad_;
  float rads_per_pixel_elevation_;
  float rads_per_pixel_azimuth_;
};

// Equality
__host__ inline bool operator==(const OSLidar& lhs, const OSLidar& rhs);

}  // namespace nvblox

#include "nvblox/core/impl/oslidar_impl.h"
