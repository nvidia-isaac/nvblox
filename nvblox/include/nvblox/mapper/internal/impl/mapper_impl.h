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
#pragma once

#include "nvblox/sensors/pointcloud_to_depth_conversion.h"
#include "nvblox/utils/timing.h"

namespace nvblox {
template <typename SensorType>
void Mapper::integrateDepth(const DepthImage& depth_frame,
                            const Transform& T_L_C, const SensorType& sensor) {
  integrateDepth(MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere),
                 T_L_C, sensor);
}

template <typename SensorType>
void Mapper::integrateDepth(const MaskedDepthImageConstView& depth_frame,
                            const Transform& T_L_C, const SensorType& sensor) {
  static_assert(is_sensor_interface<SensorType>::value,
                "Sensor does not match the required interface");

  CHECK(projective_layer_type_ != ProjectiveLayerType::kNone)
      << "You are trying to update on an inexistent projective layer.";
  // If requested, we perform preprocessing of the depth image. At the moment
  // this is just (optional) dilation of the invalid regions.
  MaskedDepthImageConstView depth_image_for_integration = depth_frame;
  if (do_depth_preprocessing_) {
    depth_image_for_integration = MaskedDepthImageConstView(
        preprocessDepthImageAsync(depth_frame), depth_frame.mask());
  }

  // Call the integrator.
  std::vector<Index3D> updated_blocks;
  if (hasTsdfLayer(projective_layer_type_)) {
    getTsdfIntegrator<SensorType>().integrateFrame(
        MaskedDepthImageConstView(depth_image_for_integration), T_L_C, sensor,
        layers_.getPtr<TsdfLayer>(), &updated_blocks);

    layers_.getPtr<TsdfLayer>()->updateGpuHash(*cuda_stream_);
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    getOccupancyIntegrator<SensorType>().integrateFrame(
        depth_image_for_integration, T_L_C, sensor,
        layers_.getPtr<OccupancyLayer>(), &updated_blocks);

    layers_.getPtr<OccupancyLayer>()->updateGpuHash(*cuda_stream_);
  }

  // Save the view data for use in view-based exclusion.
  if (!last_posed_depth_image_.hasType<PosedDepthImage<SensorType>>()) {
    LOG(INFO) << "Allocating space for last depth view for sensor type";
    last_posed_depth_image_.set(PosedDepthImage<SensorType>(
        T_L_C, sensor,
        DepthImage(depth_image_for_integration.rows(),
                   depth_image_for_integration.cols(), MemoryType::kDevice)));
  }

  // Update the stored view data
  auto* view_data =
      last_posed_depth_image_.getMutablePtr<PosedDepthImage<SensorType>>();
  view_data->T_L_C = T_L_C;
  view_data->sensor = sensor;
  view_data->depth_image.resizeAsync(depth_image_for_integration.rows(),
                                     depth_image_for_integration.cols(),
                                     *cuda_stream_);
  view_data->depth_image.copyFromAsync(depth_image_for_integration,
                                       *cuda_stream_);

  blocks_to_update_tracker_.addBlocksToUpdate(updated_blocks);
}

template <typename SensorType>
void Mapper::integrateDepth(const Pointcloud& pointcloud,
                            const Transform& T_L_S_scanStart,
                            const SensorType& lidar_sensor,
                            bool use_lidar_motion_compensation,
                            const std::optional<Transform>& T_L_S_scanEnd,
                            const std::optional<Time>& scan_duration_ms) {
  CHECK(lidar_sensor.sensor_modality() == SensorModality::kLidar)
      << "Pointcloud integration is only intended for lidar sensors";
  // Direct pointcloud integration is not supported,
  // therefore we convert the pointcloud to a spherical depth image first.
  // During this conversion, we run motion compensation if enabled.
  depthImageFromPointcloudGPU(pointcloud, T_L_S_scanStart, lidar_sensor,
                              use_lidar_motion_compensation, T_L_S_scanEnd,
                              scan_duration_ms, &depth_frame_from_pointcloud_,
                              *cuda_stream_);

  // Integrate the depth image.
  integrateDepth(depth_frame_from_pointcloud_, T_L_S_scanStart, lidar_sensor);
}

template <typename SensorType>
void Mapper::integrateColor(const ColorImage& color_frame,
                            const Transform& T_L_C, const SensorType& sensor) {
  integrateColor(MaskedColorImageConstView(color_frame, kMaskActiveEverywhere),
                 T_L_C, sensor);
}

template <typename SensorType>
void Mapper::integrateColor(const ColorImage& color_frame,
                            const DepthImage& depth_frame,
                            const Transform& T_L_C, const SensorType& sensor) {
  static_assert(is_sensor_interface<SensorType>::value,
                "Sensor does not match the required interface");

  if (hasTsdfLayer(projective_layer_type_)) {
    std::vector<Index3D> updated_blocks;
    color_integrator_.integrateFrame(
        MaskedColorImageConstView(color_frame, kMaskActiveEverywhere),
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_L_C,
        sensor, layers_.get<TsdfLayer>(), layers_.getPtr<ColorLayer>(),
        &updated_blocks);

    layers_.getPtr<ColorLayer>()->updateGpuHash(*cuda_stream_);
    blocks_to_update_tracker_.addBlocksToUpdate(
        updated_blocks,
        {BlocksToUpdateType::kColorMesh, BlocksToUpdateType::kLayerStreamer});
  }
}

template <typename SensorType>
void Mapper::integrateColor(const MaskedColorImageConstView& color_frame,
                            const Transform& T_L_C, const SensorType& sensor) {
  static_assert(is_sensor_interface<SensorType>::value,
                "Sensor does not match the required interface");

  // Color is only integrated for Tsdf layers (not for occupancy)
  if (hasTsdfLayer(projective_layer_type_)) {
    std::vector<Index3D> updated_blocks;
    color_integrator_.integrateFrame(
        color_frame, T_L_C, sensor, layers_.get<TsdfLayer>(),
        layers_.getPtr<ColorLayer>(), &updated_blocks);

    layers_.getPtr<ColorLayer>()->updateGpuHash(*cuda_stream_);
    blocks_to_update_tracker_.addBlocksToUpdate(
        updated_blocks,
        {BlocksToUpdateType::kColorMesh, BlocksToUpdateType::kLayerStreamer});
  }
}

template <typename SensorType>
void Mapper::integrateFeatures(const MaskedFeatureImageConstView& feature_frame,
                               const Transform& T_L_C,
                               const SensorType& sensor) {
  static_assert(is_sensor_interface<SensorType>::value,
                "Sensor does not match the required interface");

  // Features are only integrated for Tsdf layers (not for occupancy)
  if (hasTsdfLayer(projective_layer_type_)) {
    std::vector<Index3D> updated_blocks;
    feature_integrator_.integrateFrame(
        feature_frame, T_L_C, sensor, layers_.get<TsdfLayer>(),
        layers_.getPtr<FeatureLayer>(), &updated_blocks);

    layers_.getPtr<FeatureLayer>()->updateGpuHash(*cuda_stream_);
    blocks_to_update_tracker_.addBlocksToUpdate(
        updated_blocks,
        {BlocksToUpdateType::kFeatureMesh, BlocksToUpdateType::kLayerStreamer});
  }
}

template <typename SensorType>
void Mapper::updateFreespace(Time update_time_ms, const Transform& T_L_C,
                             const SensorType& sensor,
                             const DepthImage& depth_frame,
                             UpdateFullLayer update_full_layer) {
  // The freespace integrator only updates voxel that are in view and within the
  // negative truncation distance. Due to noisy depth measurements, a voxel
  // might occasionaly end up on the "wrong" side of the truncation distance
  // and would thus not be updated. To mitigate the effect of this on/off
  // switching, we inflate the truncation distance.
  constexpr float kTruncationDistanceMultiplier = 2.F;

  updateFreespace<SensorType>(
      update_time_ms,
      DepthObservationSpace<SensorType>(
          T_L_C, sensor, depth_frame,
          getTsdfIntegrator<SensorType>().max_integration_distance_m(),
          kTruncationDistanceMultiplier *
              getTsdfIntegrator<SensorType>().get_truncation_distance_m(
                  voxel_size_m_)),
      update_full_layer);
}

template <typename SensorType>
void Mapper::updateFreespace(
    Time update_time_ms,
    std::optional<DepthObservationSpace<SensorType>> view_to_update,
    UpdateFullLayer update_full_layer) {
  CHECK(hasFreespaceLayer(projective_layer_type_))
      << "Trying to update the freespace layer while it is not enabled.";

  const std::vector<Index3D> blocks_to_update =
      getBlocksToUpdate(BlocksToUpdateType::kFreespace, update_full_layer);

  freespace_integrator_.updateFreespaceLayer(
      blocks_to_update, update_time_ms, layers_.get<TsdfLayer>(),
      view_to_update, layers_.getPtr<FreespaceLayer>());

  blocks_to_update_tracker_.markBlocksAsUpdated(BlocksToUpdateType::kFreespace);
  layers_.getPtr<FreespaceLayer>()->updateGpuHash(*cuda_stream_);
}

template <typename SensorType>
void Mapper::decayTsdfExcludeLastView() {
  if (last_posed_depth_image_.hasType<PosedDepthImage<SensorType>>()) {
    const auto& posed_depth_image =
        last_posed_depth_image_.get<PosedDepthImage<SensorType>>();
    decayTsdfInternal<SensorType>(
        std::make_optional<DepthObservationSpace<SensorType>>(
            posed_depth_image.toDepthObservationSpace(
                getTsdfIntegrator<SensorType>().max_integration_distance_m(),
                getTsdfIntegrator<SensorType>().get_truncation_distance_m(
                    voxel_size_m_))));
  } else {
    LOG(INFO) << "Last view not set for sensor type. Decaying all voxels";
    decayTsdfInternal<SensorType>(std::nullopt);
  }
}

template <typename SensorType>
void Mapper::decayTsdfInternal(
    const std::optional<DepthObservationSpace<SensorType>>& inclusion_data) {
  // TODO(remos): In the future we could exclude the blocks not decayed, from
  // the blocks requiring an update.
  blocks_to_update_tracker_.addAllBlocksToUpdate();

  // Decay
  std::vector<Index3D> removed_blocks =
      tsdf_decay_integrator_.decay<SensorType>(layers_.getPtr<TsdfLayer>(),
                                               std::nullopt, inclusion_data,
                                               *cuda_stream_);

  // Clear the blocks that got removed in the tsdf layer also in the esdf,
  // freespace and mesh layers.
  clearBlocksInLayers(removed_blocks);
  layers_.getPtr<TsdfLayer>()->updateGpuHash(*cuda_stream_);
}

template <typename SensorType>
void Mapper::decayOccupancyExcludeLastView() {
  if (last_posed_depth_image_.hasType<PosedDepthImage<SensorType>>()) {
    const auto& posed_depth_image =
        last_posed_depth_image_.get<PosedDepthImage<SensorType>>();
    decayOccupancyInternal(
        std::make_optional<DepthObservationSpace<SensorType>>(
            posed_depth_image.toDepthObservationSpace(
                getOccupancyIntegrator<SensorType>()
                    .max_integration_distance_m(),
                getOccupancyIntegrator<SensorType>().get_truncation_distance_m(
                    voxel_size_m_))));
  } else {
    LOG(INFO) << "Last view not set for sensor type. Decaying all voxels";
    decayOccupancyInternal<SensorType>(std::nullopt);
  }
}

template <typename SensorType>
void Mapper::decayOccupancyInternal(
    const std::optional<DepthObservationSpace<SensorType>>& inclusion_data) {
  // TODO(remos): In the future we could exclude the blocks not decayed, from
  // the blocks requiring an update.
  blocks_to_update_tracker_.addAllBlocksToUpdate();

  // Decay
  std::vector<Index3D> removed_blocks =
      occupancy_decay_integrator_.decay<SensorType>(
          layers_.getPtr<OccupancyLayer>(), std::nullopt, inclusion_data,
          *cuda_stream_);

  // Clear the blocks that got removed in the occupancy layer also in the esdf,
  // freespace and mesh layers.
  clearBlocksInLayers(removed_blocks);
  layers_.getPtr<OccupancyLayer>()->updateGpuHash(*cuda_stream_);
}

template <typename AppearanceVoxelType>
void Mapper::updateFlatMesh() {
  if (!hasTsdfLayer(projective_layer_type_)) return;
  updateFlatMeshImpl<AppearanceVoxelType>(
      layers_.get<TsdfLayer>().getAllBlockIndices());
}

template <typename AppearanceVoxelType>
void Mapper::updateFlatMesh(const Camera& camera, const Transform& T_L_C,
                            float max_depth) {
  if (!hasTsdfLayer(projective_layer_type_)) return;
  timing::Timer frustum_timer("mapper/update_flat_mesh/frustum_cull");
  const auto block_indices =
      getFrustumFilteredIndices(camera, T_L_C, max_depth);
  frustum_timer.Stop();
  updateFlatMeshImpl<AppearanceVoxelType>(block_indices);
}

template <typename AppearanceVoxelType>
void Mapper::updateFlatMeshImpl(const std::vector<Index3D>& block_indices) {
  timing::Timer timer("mapper/update_flat_mesh");
  static_assert(std::is_same_v<AppearanceVoxelType, ColorVoxel> ||
                    std::is_same_v<AppearanceVoxelType, FeatureVoxel>,
                "Unsupported appearance voxel type for flat mesh");
  using AppearanceLayerType = VoxelBlockLayer<AppearanceVoxelType>;
  if constexpr (std::is_same_v<AppearanceVoxelType, ColorVoxel>) {
    color_flat_mesh_integrator_.integrateBlocks(
        layers_.get<TsdfLayer>(), layers_.get<AppearanceLayerType>(),
        block_indices, &flat_color_mesh_);
  } else {
    feature_flat_mesh_integrator_.integrateBlocks(
        layers_.get<TsdfLayer>(), layers_.get<AppearanceLayerType>(),
        block_indices, &flat_feature_mesh_);
  }
}

}  // namespace nvblox
