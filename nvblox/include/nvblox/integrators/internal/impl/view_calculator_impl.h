/*
Copyright 2025 NVIDIA CORPORATION

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
#include "nvblox/core/hash.h"
#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/geometry/transforms.h"
#include "nvblox/integrators/view_calculator.h"
#include "nvblox/rays/ray_caster.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

template <typename SensorType>
std::vector<Index3D> ViewCalculator::getBlocksInImageViewProjection(
    const Transform& T_L_C, const SensorType& sensor, const float block_size,
    const float max_distance) {
  CHECK_GT(max_distance, 0.0f);
  timing::Timer("view_calculator/get_blocks_in_view_projection");
  // Check cache
  CHECK_NOTNULL(projection_viewpoint_cache_);
  if (cache_last_viewpoint_) {
    if (auto cached_result =
            projection_viewpoint_cache_->getCachedResult(T_L_C, sensor);
        cached_result.has_value()) {
      return cached_result.value();
    }
  }

  // Project all block centers into the image and check if they are
  // inside the image viewport.

  // Coarse bound: AABB
  constexpr float kMinDistance = 1E-6f;
  AxisAlignedBoundingBox aabb_L =
      sensor.getViewAABB(T_L_C, kMinDistance, max_distance);

  // Apply the workspace bounds,
  // i.e. make sure we only return blocks that are within the workspace limits.
  if (!applyWorkspaceBounds(aabb_L, workspace_bounds_type_,
                            workspace_bounds_min_corner_m(),
                            workspace_bounds_max_corner_m(), &aabb_L)) {
    // Return an empty vector of blocks to update if the workspace is not valid
    // (i.e. empty).
    return std::vector<Index3D>();
  }
  const std::vector<Index3D> block_indices_in_aabb =
      getBlockIndicesTouchedByBoundingBox(block_size, aabb_L);

  // Get the transform to sensor from layer
  const Transform T_C_L = T_L_C.inverse();

  // Filter out blocks not visible
  std::vector<Index3D> block_indices_in_view = getVisibleBlocksByProjection(
      block_indices_in_aabb, sensor, T_C_L, block_size, kMinDistance);

  // Cache
  if (cache_last_viewpoint_) {
    projection_viewpoint_cache_->storeResultInCache(T_L_C, sensor,
                                                    block_indices_in_view);
  }
  return block_indices_in_view;
}

// Margin padded to the viewport when checking if a pixel is visible.
inline float getViewportMargin(const float sensor_height) {
  return sensor_height / 20.F;  // Rather arbitrary chosen.
}

template <typename SensorType>
std::vector<Index3D> ViewCalculator::getVisibleBlocksByProjection(
    const std::vector<Index3D>& block_indices, const SensorType& sensor,
    const Transform& T_C_L, const float block_size, const float min_distance) {
  // Extract rotation and translation component
  const Eigen::Matrix3f rotation_C_L = T_C_L.rotation();
  const Eigen::Vector3f translation_C_L = T_C_L.translation();

  std::vector<Index3D> block_indices_in_view;

  for (const Index3D& block_index : block_indices) {
    // Transform the block center into sensor frame
    const Eigen::Vector3f p3d_layer =
        getCenterPositionFromBlockIndex(block_size, block_index);
    const Eigen::Vector3f p3d_cam = rotation_C_L * p3d_layer + translation_C_L;

    if (p3d_cam[2] > min_distance) {
      // Project into normalized sensor coordinates
      Eigen::Vector2f u_c;
      constexpr bool kNoCheckViewport = false;
      sensor.project(p3d_cam, &u_c, SensorType::kDefaultMinProjectionDepth,
                     kNoCheckViewport);

      // We add a small margin to the viewport before checking. This is to
      // include more blocks that are partially in view.
      const int margin_px = getViewportMargin(sensor.rows());
      if (u_c.x() > -margin_px && u_c.x() < sensor.cols() + margin_px &&
          u_c.y() > -margin_px && u_c.y() < sensor.rows() + margin_px) {
        block_indices_in_view.push_back(block_index);
      }
    }
  }
  return block_indices_in_view;
}

template <typename SensorType>
std::optional<std::vector<Index3D>> ViewpointCache::getCachedResult(
    const Transform& T_L_C, const SensorType& sensor) const {
  CHECK_EQ(sensor_cache_.size(), pose_cache_.size());
  CHECK_EQ(sensor_cache_.size(), blocks_in_view_cache_.size());

  if (pose_cache_.empty() || sensor_cache_.empty()) {
    return std::nullopt;
  }

  // Iterate through the cache and check if anything fits the
  // current pose and sensor.
  bool cache_hit = false;
  size_t cache_hit_idx = 0;
  for (size_t i = 0; i < sensor_cache_.size(); i++) {
    NVBLOX_CHECK(sensor_cache_[i].hasType<SensorType>(), "Empty sensor");

    constexpr float kTranslationToleranceM = 0.001f;
    constexpr float kAngularToleranceDeg = 0.1f;
    const bool same_extrinsics = arePosesClose(
        T_L_C, pose_cache_[i], kTranslationToleranceM, kAngularToleranceDeg);
    const bool same_sensors = (sensor == sensor_cache_[i].get<SensorType>());

    if (same_extrinsics && same_sensors) {
      cache_hit = true;
      cache_hit_idx = i;
      break;
    }
  }

  // Return the cached result if there is any.
  if (!cache_hit) {
    return std::nullopt;
  }
  return blocks_in_view_cache_[cache_hit_idx];
}

template <typename SensorType>
void ViewpointCache::storeResultInCache(
    const Transform& T_L_C, const SensorType& sensor,
    const std::vector<Index3D>& blocks_in_view) {
  CHECK_EQ(sensor_cache_.size(), pose_cache_.size());
  CHECK_EQ(sensor_cache_.size(), blocks_in_view_cache_.size());
  if (sensor_cache_.size() == kMaxCacheSize) {
    // Remove the oldest element.
    pose_cache_.pop_back();
    sensor_cache_.pop_back();
    blocks_in_view_cache_.pop_back();
  }
  TypeIndexedStore store;
  store.set(sensor);
  sensor_cache_.push_front(std::move(store));
  pose_cache_.push_front(T_L_C);
  blocks_in_view_cache_.push_front(blocks_in_view);
}

}  // namespace nvblox
