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

namespace nvblox {

template <typename SensorType>
void MultiMapper::integrateDepth(const DepthImage& depth_frame,
                                 const Transform& T_L_CD,
                                 const SensorType& sensor,
                                 const std::optional<Time>& update_time_ms) {
  if (!isDynamicMapping(mapping_type_)) {
    // For static mapping only integrate to the background mapper
    background_mapper_->integrateDepth(depth_frame, T_L_CD, sensor);
  } else {
    CHECK(update_time_ms.has_value());

    // To speed-up execution, the functions called here are grouped into blocks
    // that can be launched in parallel without race conditions.
    //
    // The blocks are determined by tabulating the R/W access of each
    // function/resource and then group functions that does not read and write
    // to the same resource. We also select a flow that doesn't require storing
    // any data from the previous iteration.
    //
    // FUNCTION  	         TL  FL  OL  DI  FD  DD
    // integrateTsdf             W           R
    // updateFreespace	         R   W
    // computeDynamics	             R       R       W
    // removeSmall                               W   R
    // integrateOccupancy                W       R
    //
    // TL: TSDF layer
    // FL: Freespace Layer
    // OL: Occupacy Layer
    // DI: Depth image
    // FD: Foreground depth image
    // DD: DynamicsDetector
    //
    // The selected blocks:
    //    1. IntegrateTsdf + computeDynamics
    //    2. removeSmallConnectedComponents
    //    3. updateFreespace + integrateOccupancy

    {  // Block 1
      timing::Timer timer("multi_mapper/integrate_depth/dynamic_block1");

      // TODO(dtingdahl) Reduce overhead by recycling threads instead of
      // re-creating them.
      std::vector<std::thread> threads;

      // Integrate TSDF
      threads.push_back(std::thread([&]() {
        background_mapper_->integrateDepth(depth_frame, T_L_CD, sensor);
      }));

      // Compute dynamic mask. Note that we're using the freespace layer
      // computed during the previous update. This should be fine since the
      // freespace layer is designed to reacting slow to changes.
      threads.push_back(std::thread([&]() {
        dynamic_detector_.computeDynamics(
            depth_frame, background_mapper_->freespace_layer(), sensor, T_L_CD);
      }));

      // sync threads
      std::for_each(threads.begin(), threads.end(),
                    [](std::thread& t) { t.join(); });
      threads.clear();
    }

    {  // Block 2 (This block is launched synchronously since there is only one
       // call)
      timing::Timer timer("multi_mapper/integrate_depth/dynamic_block2");

      // Update dynamic mask
      const MonoImage& dynamic_mask = dynamic_detector_.getDynamicMaskImage();

      // Remove small components (assumed to be noise) from the mask
      if (params_.remove_small_connected_components) {
        mask_preprocessor_.removeSmallConnectedComponents(
            dynamic_mask, params_.connected_mask_component_size_threshold,
            &cleaned_dynamic_mask_);
      } else {
        cleaned_dynamic_mask_.copyFromAsync(dynamic_mask, *cuda_stream_);
        cuda_stream_->synchronize();
      }
    }

    // Block 3
    timing::Timer timer("multi_mapper/integrate_depth/dynamic_block3");
    std::vector<std::thread> threads;

    // Update occupancy
    threads.push_back(std::thread([&]() {
      foreground_mapper_->integrateDepth(
          MaskedDepthImageConstView(depth_frame, cleaned_dynamic_mask_), T_L_CD,
          sensor);
    }));

    // Update freespace.
    threads.push_back(std::thread([&]() {  // NOLINT
      background_mapper_->updateFreespace(update_time_ms.value(), T_L_CD,
                                          sensor, depth_frame);
    }));

    // sync threads
    std::for_each(threads.begin(), threads.end(),
                  [](std::thread& t) { t.join(); });
  }
}

template <typename SensorType>
void MultiMapper::integrateDepth(const DepthImage& depth_frame,
                                 const MonoImage& mask, const Transform& T_L_CD,
                                 const Transform& T_CM_CD,
                                 const SensorType& depth_sensor,
                                 const SensorType& mask_sensor) {
  CHECK(isHumanMapping(mapping_type_))
      << "Passing a mask to integrateDepth is only valid for human "
         "mapping.";

  // Remove small components (assumed to be noise) from the mask
  if (params_.remove_small_connected_components) {
    mask_preprocessor_.removeSmallConnectedComponents(
        mask, params_.connected_mask_component_size_threshold,
        &cleaned_semantic_mask_);
  } else {
    cleaned_semantic_mask_.copyFromAsync(mask, *cuda_stream_);
  }
  // Split into foreground and background depth frame
  image_masker_.splitImageOnGPU(
      depth_frame, cleaned_semantic_mask_, T_CM_CD, depth_sensor, mask_sensor,
      &depth_frame_background_, &depth_frame_foreground_,
      &foreground_depth_overlay_);

  // Integrate the frames to the respective layer cake
  background_mapper_->integrateDepth(depth_frame_background_, T_L_CD,
                                     depth_sensor);
  foreground_mapper_->integrateDepth(depth_frame_foreground_, T_L_CD,
                                     depth_sensor);
}

template <typename SensorType>
void MultiMapper::integrateDepth(const Pointcloud& pointcloud,
                                 const Transform& T_L_S_scanStart,
                                 const SensorType& lidar_sensor,
                                 bool use_lidar_motion_compensation,
                                 const std::optional<Transform>& T_L_S_scanEnd,
                                 const std::optional<Time>& scan_duration_ms,
                                 const std::optional<Time>& update_time_ms) {
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
  integrateDepth(depth_frame_from_pointcloud_, T_L_S_scanStart, lidar_sensor,
                 update_time_ms);
}

template <typename SensorType>
void MultiMapper::integrateColor(const ColorImage& color_frame,
                                 const Transform& T_L_C,
                                 const SensorType& sensor) {
  // TODO(remos): For kDynamic we should split the image and only integrate
  // background pixels. As the dynamic mask is not a direct overlay of the
  // color image, this requires implementing a new splitImageOnGPU for color
  // images.
  background_mapper_->integrateColor(color_frame, T_L_C, sensor);
}

template <typename SensorType>
void MultiMapper::integrateColor(const ColorImage& color_frame,
                                 const MonoImage& foreground_mask,
                                 const Transform& T_L_C,
                                 const SensorType& sensor) {
  CHECK(isHumanMapping(mapping_type_))
      << "Passing a mask to integrateColor is only valid for human "
         "mapping.";
  if (mapping_type_ == MappingType::kHumanWithStaticOccupancy) {
    // We do nothing because color integration is only implemented for
    // static tsdf.
    return;
  }

  // Remove small components (assumed to be noise) from the mask
  // We do this again in case the mask is not synced with the depth mask
  if (params_.remove_small_connected_components) {
    mask_preprocessor_.removeSmallConnectedComponents(
        foreground_mask, params_.connected_mask_component_size_threshold,
        &cleaned_semantic_mask_);
  } else {
    cleaned_semantic_mask_.copyFromAsync(foreground_mask, *cuda_stream_);
  }

  // Integrate the frames to the respective layer cake
  foreground_mapper_->integrateColor(
      MaskedColorImageConstView(color_frame, cleaned_semantic_mask_,
                                MaskMode::kNonInverted),
      T_L_C, sensor);
  background_mapper_->integrateColor(
      MaskedColorImageConstView(color_frame, cleaned_semantic_mask_,
                                MaskMode::kInverted),
      T_L_C, sensor);
}

template <typename AppearanceVoxelType>
void MultiMapper::updateFlatMesh() {
  background_mapper_->template updateFlatMesh<AppearanceVoxelType>();
}

template <typename AppearanceVoxelType>
void MultiMapper::updateFlatMesh(const Camera& camera, const Transform& T_L_C,
                                 float max_depth) {
  background_mapper_->template updateFlatMesh<AppearanceVoxelType>(
      camera, T_L_C, max_depth);
}

}  // namespace nvblox
