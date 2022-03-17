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

#include <memory>

#include "nvblox/core/camera.h"
#include "nvblox/core/image.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"

namespace nvblox {

// Helper class for frustum calculations.
class FrustumCalculator {
 public:
  FrustumCalculator();
  ~FrustumCalculator();

  // Gets blocks which fall into the camera view up to: max camera distance
  static std::vector<Index3D> getBlocksInView(const Transform& T_L_C,
                                              const Camera& camera,
                                              const float block_size,
                                              const float max_distance);

  // Gets blocks which fall into the camera view up to: max distance in the
  // depth frame
  static std::vector<Index3D> getBlocksInImageView(
      const DepthImage& depth_frame, const Transform& T_L_C,
      const Camera& camera, const float block_size,
      const float truncation_distance_m,
      const float max_integration_distance_m);

  // Performs ray casting to get the blocks in view
  // NOTE(helenol) The only non-static member function of this class.
  std::vector<Index3D> getBlocksInImageViewCuda(
      const DepthImage& depth_frame, const Transform& T_L_C,
      const Camera& camera, const float block_size,
      const float truncation_distance_m,
      const float max_integration_distance_m);

  // Params
  unsigned int raycast_subsampling_factor() const;
  void raycast_subsampling_factor(unsigned int raycast_subsampling_rate);

 private:
  // Raycasts to all corners of all blocks touched by the endpoints of depth
  // rays.
  void getBlocksByRaycastingCorners(
      const Transform& T_L_C,                  // NOLINT
      const Camera& camera,                    // NOLINT
      const DepthImage& depth_frame,           // NOLINT
      float block_size,                        // NOLINT
      const float truncation_distance_m,       // NOLINT
      const float max_integration_distance_m,  // NOLINT
      const Index3D& min_index,                // NOLINT
      const Index3D& aabb_size,                // NOLINT
      bool* aabb_updated_cuda);

  // Raycasts through (possibly subsampled) pixels in the image.
  void getBlocksByRaycastingPixels(
      const Transform& T_L_C,                  // NOLINT
      const Camera& camera,                    // NOLINT
      const DepthImage& depth_frame,           // NOLINT
      float block_size,                        // NOLINT
      const float truncation_distance_m,       // NOLINT
      const float max_integration_distance_m,  // NOLINT
      const Index3D& min_index,                // NOLINT
      const Index3D& aabb_size,                // NOLINT
      bool* aabb_updated_cuda);

  // A 3D grid of bools, one for each block in the AABB, which indicates if it
  // is in the view. The 3D grid is represented as a flat vector.
  device_vector<bool> aabb_device_buffer_;
  host_vector<bool> aabb_host_buffer_;

  // Whether to use the single kernel or double kernel computation for the
  // CUDA version. Single kernel raycasts to every pixel, double kernel
  // raycasts to every block corner.
  bool raycast_to_pixels_ = true;
  // The rate at which we subsample pixels to raycast. Note that we always
  // raycast the extremes of the frame, no matter the subsample rate.
  unsigned int raycast_subsampling_factor_ = 1;

  cudaStream_t cuda_stream_;
};

}  // namespace nvblox