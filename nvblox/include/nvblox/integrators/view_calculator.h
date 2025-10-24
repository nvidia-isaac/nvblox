/*
Copyright 2022-2024 NVIDIA CORPORATION

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

#include <deque>
#include <memory>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/geometry/workspace_bounds.h"
#include "nvblox/integrators/view_calculator_params.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/type_indexed_store.h"

namespace nvblox {

// Forward declare the cache. See below for the full definition.
class ViewpointCache;

/// A class that calculates the blocks in the camera view, given intrinsic and
/// extrinsic parameters.
class ViewCalculator {
 public:
  ViewCalculator();
  ViewCalculator(std::shared_ptr<CudaStream> cuda_stream);
  ~ViewCalculator() = default;

  /// @brief The type of calculation used to get the view.
  enum class CalculationType { kPlanes, kRaycasting };

  /// Gets blocks which fall into the camera view (without using an image)
  /// Operates by checking if voxel block corners fall inside the space formed
  /// by the 4 images sides and the max distance plane.
  /// @param T_L_C The pose of the sensor. Supplied as a Transform mapping
  /// points in the sensor frame (C) to the layer frame (L).
  /// @param sensor The sensor (intrinsics) model.
  /// @param block_size The size of the blocks in the layer.
  /// @param max_distance The maximum distance of blocks considered.
  /// @return a vector of the 3D indices of the blocks in view.
  template <typename SensorType>
  std::vector<Index3D> getBlocksInImageViewProjection(const Transform& T_L_C,
                                                      const SensorType& sensor,
                                                      const float block_size,
                                                      const float max_distance);

  /// Gets blocks which fall into the sensor's view (using a depth image)
  /// Performs ray casting to get the blocks in view
  /// Operates by ray through the grid returning the blocks traversed in the ray
  /// casting process. The number of pixels on the image plane raycast is
  /// determined by the class parameter raycast_subsampling_factor.
  /// @param depth_frame the depth image.
  /// @param T_L_C The pose of the sensor. Supplied as a Transform mapping
  /// points in the sensor frame (C) to the layer frame (L).
  /// @param sensor The Image sensors's (intrinsic) model.
  /// @param block_size The size of the blocks in the layer.
  /// @param max_integration_distance_behind_surface_m The truncation distance.
  /// @param max_integration_distance_m The max integration distance.
  /// @return a vector of the 3D indices of the blocks in view.
  template <typename SensorType>
  std::vector<Index3D> getBlocksInImageViewRaycast(
      const MaskedDepthImageConstView& depth_frame, const Transform& T_L_C,
      const SensorType& sensor, const float block_size,
      const float max_integration_distance_behind_surface_m,
      const float max_integration_distance_m);

  /// A parameter getter
  /// The rate at which we subsample pixels to raycast. Note that we always
  /// raycast the edges of the frame, no matter the subsample rate. For
  /// example, for a 100px by 100px image with a subsampling factor of 4,
  /// 25x25 rays are traced, and the blocks those rays touch are returned.
  /// @returns the ray casting subsampling factor
  unsigned int raycast_subsampling_factor() const;

  /// A parameter setter
  /// See raycast_subsampling_factor()
  /// @param raycast_subsampling_rate the ray casting subsampling factor
  void raycast_subsampling_factor(unsigned int raycast_subsampling_rate);

  /// @brief A parameter getter
  /// @return The type of the workspace bounds
  WorkspaceBoundsType workspace_bounds_type() const;

  /// @brief A parameter setter
  /// @param workspace_bounds_type See workspace_bounds_type()
  void workspace_bounds_type(WorkspaceBoundsType workspace_bounds_type);

  /// @brief A parameter getter
  /// @return The min corner of the workspace.
  Vector3f workspace_bounds_min_corner_m() const;

  /// @brief A parameter setter
  /// @param workspace_bounds_min_corner_m See workspace_bounds_min_corner_m()
  void workspace_bounds_min_corner_m(
      const Vector3f& workspace_bounds_min_corner_m);

  /// @brief A parameter getter
  /// @return The max corner of the workspace.
  Vector3f workspace_bounds_max_corner_m() const;

  /// @brief A parameter setter
  /// @param workspace_bounds_max_corner_m See workspace_bounds_max_corner_m()
  void workspace_bounds_max_corner_m(
      const Vector3f& workspace_bounds_max_corner_m);

  /// @brief A parameter getter
  /// @return Whether or not to avoid re-computing viewpoints for repeated
  /// queries.
  bool cache_last_viewpoint() const;

  /// @brief A parameter setter
  /// @param cache_last_viewpoint See cache_last_viewpoint()
  void cache_last_viewpoint(const bool cache_last_viewpoint);

  /// @brief Gets the viewpoint cache
  /// @param calculation_type For each calculation type there is a separate
  /// cache. Specifying the calculation type selects the respective cache to
  /// return.
  /// @return The viewpoint cache of the specified type.
  std::shared_ptr<ViewpointCache> get_viewpoint_cache(
      const CalculationType calculation_type) const;

  /// @brief Sets the viewpoint cache. Useful for allowing sharing of caches
  /// between several ViewpointCalculators.
  /// @param viewpoint_cache The viewpoint cache to use.
  /// @param calculation_type For each calculation type there is a separate
  /// cache. Specifying the calculation type selects the respective cache to
  /// set.
  void set_viewpoint_cache(std::shared_ptr<ViewpointCache> viewpoint_cache,
                           const CalculationType calculation_type);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 private:
  // Raycasts through (possibly subsampled) pixels in the image.
  template <typename SensorType>
  void getBlocksByRaycastingPixelsAsync(
      const Transform& T_L_C,                                 // NOLINT
      const SensorType& sensor,                               // NOLINT
      const MaskedDepthImageConstView& depth_frame,           // NOLINT
      float block_size,                                       // NOLINT
      const float max_integration_distance_behind_surface_m,  // NOLINT
      const float max_integration_distance_m,                 // NOLINT
      const Index3D& min_index,                               // NOLINT
      const Index3D& aabb_size,                               // NOLINT
      bool* aabb_updated_cuda);

  // Filter out non-visible blocks by projecting them into the image plane.
  template <typename SensorType>
  std::vector<Index3D> getVisibleBlocksByProjection(
      const std::vector<Index3D>& block_indices, const SensorType& sensor,
      const Transform& T_C_L, const float block_size, const float min_distance);

  // A 3D grid of bools, one for each block in the AABB, which indicates if it
  // is in the view. The 3D grid is represented as a flat vector.
  device_vector<bool> aabb_device_buffer_;
  host_vector<bool> aabb_host_buffer_;

  // Parameters.
  unsigned int raycast_subsampling_factor_ =
      kRaycastSubsamplingFactorDesc.default_value;
  WorkspaceBoundsType workspace_bounds_type_ =
      kWorkspaceBoundsTypeDesc.default_value;
  float workspace_bounds_min_height_m_ =
      kWorkspaceBoundsMinHeightDesc.default_value;
  float workspace_bounds_max_height_m_ =
      kWorkspaceBoundsMaxHeightDesc.default_value;
  float workspace_bounds_min_corner_x_m_ =
      kWorkspaceBoundsMinCornerXDesc.default_value;
  float workspace_bounds_max_corner_x_m_ =
      kWorkspaceBoundsMaxCornerXDesc.default_value;
  float workspace_bounds_min_corner_y_m_ =
      kWorkspaceBoundsMinCornerYDesc.default_value;
  float workspace_bounds_max_corner_y_m_ =
      kWorkspaceBoundsMaxCornerYDesc.default_value;

  // Caching the last viewpoint calculation.
  bool cache_last_viewpoint_ = true;
  std::shared_ptr<ViewpointCache> raycasting_viewpoint_cache_;
  std::shared_ptr<ViewpointCache> projection_viewpoint_cache_;

  // CUDA stream on which to execute work.
  std::shared_ptr<CudaStream> cuda_stream_;
};

// Specialization for Camera that speeds up computation by making use of
// normalized image coordinates.
template <>
std::vector<Index3D> ViewCalculator::getVisibleBlocksByProjection<Camera>(
    const std::vector<Index3D>& block_indices, const Camera& camera,
    const Transform& T_C_L, const float block_size, const float min_distance);

class ViewpointCache {
 public:
  /// Gets a cached result of previous getBlocksInViewCalls, if the viewpoint
  /// and intrinsics were the same.
  /// @param T_L_C The pose of the sensor.
  /// @param sensor The intrinsics of the sensor.
  /// @return The cached blocks in view if the viewpoint is the same,
  /// otherwise std::nullopt.
  template <typename SensorType>
  std::optional<std::vector<Index3D>> getCachedResult(
      const Transform& T_L_C, const SensorType& sensor) const;

  /// Stores the result of a call to getBlocksInView*() in the cache.
  /// @param T_L_C The pose of the sensor.
  /// @param sensor The intrinsics of the sensor.
  /// @param blocks_in_view The calculated blocks in view list.
  template <typename SensorType>
  void storeResultInCache(const Transform& T_L_C, const SensorType& sensor,
                          const std::vector<Index3D>& blocks_in_view);

 private:
  std::deque<TypeIndexedStore> sensor_cache_;
  std::deque<Transform> pose_cache_;
  std::deque<std::vector<Index3D>> blocks_in_view_cache_;

  /// Maximum number of views to store in the cache.
  /// A cache size of 1 is sufficient to cache a view between the static and
  /// dynamic mapper as they are guaranteed to integrate the frame in
  /// sequence.
  /// Currently we set cache size to 2 to also allow caching 2 static sensors.
  /// If you want to support caching for n static sensors,
  /// you want to increase the maximum cache size to n.
  static constexpr int kMaxCacheSize = 2;
};

}  // namespace nvblox

#include "nvblox/integrators/internal/impl/view_calculator_impl.h"
