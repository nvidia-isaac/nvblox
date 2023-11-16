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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/lidar.h"

namespace nvblox {

/// A class that calculates the blocks in the camera view, given intrinsic and
/// extrinsic parameters.
class ViewCalculator {
 public:
  ViewCalculator();
  ViewCalculator(std::shared_ptr<CudaStream> cuda_stream);
  ~ViewCalculator() = default;

  /// Gets blocks which fall into the camera view (without using an image)
  /// Operates by checking if voxel block corners fall inside the pyramid formed
  /// by the 4 images sides and the max distance plane.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param camera The camera (intrinsics) model.
  /// @param block_size The size of the blocks in the layer.
  /// @param max_distance The maximum distance of blocks considered.
  /// @return a vector of the 3D indices of the blocks in view.
  static std::vector<Index3D> getBlocksInViewPlanes(const Transform& T_L_C,
                                                    const Camera& camera,
                                                    const float block_size,
                                                    const float max_distance);

  /// Gets blocks which fall into the camera view (using a depth image)
  /// Operates by checking if voxel block corners fall inside the pyrimid formed
  /// by the 4 images sides and the max distance plane. The max distance is the
  /// smaller of either max_integration_distance_m or the maximum value in the
  /// depth image + truncation distance.
  /// @param depth_frame the depth image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param camera The camera (intrinsics) model.
  /// @param block_size The size of the blocks in the layer.
  /// @param max_integration_distance_behind_surface_m The truncation distance.
  /// @param max_integration_distance_m The max integration distance.
  /// @return a vector of the 3D indices of the blocks in view.
  static std::vector<Index3D> getBlocksInImageViewPlanes(
      const DepthImage& depth_frame, const Transform& T_L_C,
      const Camera& camera, const float block_size,
      const float max_integration_distance_behind_surface_m,
      const float max_integration_distance_m);

  /// Gets blocks which fall into the camera view (using a depth image)
  /// Performs ray casting to get the blocks in view
  /// Operates by ray through the grid returning the blocks traversed in the ray
  /// casting process. The number of pixels on the image plane raycast is
  /// determined by the class parameter raycast_subsampling_factor.
  /// @param depth_frame the depth image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param camera The camera (intrinsics) model.
  /// @param block_size The size of the blocks in the layer.
  /// @param max_integration_distance_behind_surface_m The truncation distance.
  /// @param max_integration_distance_m The max integration distance.
  /// @return a vector of the 3D indices of the blocks in view.
  std::vector<Index3D> getBlocksInImageViewRaycast(
      const DepthImage& depth_frame, const Transform& T_L_C,
      const Camera& camera, const float block_size,
      const float max_integration_distance_behind_surface_m,
      const float max_integration_distance_m);

  /// Gets blocks which fall into the lidar view (using a depth image)
  /// Performs ray casting to get the blocks in view
  /// Operates by ray through the grid returning the blocks traversed in the ray
  /// casting process. The number of pixels on the image plane raycast is
  /// determined by the class parameter raycast_subsampling_factor.
  /// @param depth_frame the depth image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param Lidar The lidar (intrinsics) model.
  /// @param block_size The size of the blocks in the layer.
  /// @param max_integration_distance_behind_surface_m The truncation distance.
  /// @param max_integration_distance_m The max integration distance.
  /// @return a vector of the 3D indices of the blocks in view.
  std::vector<Index3D> getBlocksInImageViewRaycast(
      const DepthImage& depth_frame, const Transform& T_L_C, const Lidar& lidar,
      const float block_size,
      const float max_integration_distance_behind_surface_m,
      const float max_integration_distance_m);

  /// A parameter getter
  /// The rate at which we subsample pixels to raycast. Note that we always
  /// raycast the edges of the frame, no matter the subsample rate. For example,
  /// for a 100px by 100px image with a subsampling factor of 4, 25x25 rays are
  /// traced, and the blocks those rays touch are returned.
  /// @returns the ray casting subsampling factor
  unsigned int raycast_subsampling_factor() const;

  /// A parameter setter
  /// See raycast_subsampling_factor()
  /// @param raycast_subsampling_rate the ray casting subsampling factor
  void raycast_subsampling_factor(unsigned int raycast_subsampling_rate);

  /// A parameter getter
  /// If this parameter is true, calls to getBlocksInImageViewRaycast() finds
  /// block by raycasting through (subsampled) pixels. If false blocks are
  /// generated by raycasting to the corners of blocks touched by ray end
  /// points.
  bool raycast_to_pixels() const;

  /// A parameter setter
  /// See raycast_to_pixels()
  /// @param raycast_to_pixels whether to raycast to block corners.
  void raycast_to_pixels(bool raycast_to_pixels);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 private:
  // Raycasts to all corners of all blocks touched by the endpoints of depth
  // rays.
  template <typename SensorType>
  void getBlocksByRaycastingCorners(
      const Transform& T_L_C,                                 // NOLINT
      const SensorType& camera,                               // NOLINT
      const DepthImage& depth_frame,                          // NOLINT
      float block_size,                                       // NOLINT
      const float max_integration_distance_behind_surface_m,  // NOLINT
      const float max_integration_distance_m,                 // NOLINT
      const Index3D& min_index,                               // NOLINT
      const Index3D& aabb_size,                               // NOLINT
      bool* aabb_updated_cuda);

  // Raycasts through (possibly subsampled) pixels in the image.
  template <typename SensorType>
  void getBlocksByRaycastingPixels(
      const Transform& T_L_C,                                 // NOLINT
      const SensorType& camera,                               // NOLINT
      const DepthImage& depth_frame,                          // NOLINT
      float block_size,                                       // NOLINT
      const float max_integration_distance_behind_surface_m,  // NOLINT
      const float max_integration_distance_m,                 // NOLINT
      const Index3D& min_index,                               // NOLINT
      const Index3D& aabb_size,                               // NOLINT
      bool* aabb_updated_cuda);

  // Templated version of the public getBlocksInImageViewRaycast() methods.
  // Internally we use this templated version of this function called with
  // Camera and Lidar classes.
  template <typename SensorType>
  std::vector<Index3D> getBlocksInImageViewRaycastTemplate(
      const DepthImage& depth_frame, const Transform& T_L_C,
      const SensorType& camera, const float block_size,
      const float max_integration_distance_behind_surface_m,
      const float max_integration_distance_m);

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
  unsigned int raycast_subsampling_factor_ = 4;

  // CUDA stream on which to execute work.
  std::shared_ptr<CudaStream> cuda_stream_;
};

}  // namespace nvblox
