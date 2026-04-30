/*
Copyright 2024 NVIDIA CORPORATION

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

#include "nvblox/experimental/ground_plane/ground_plane_estimator_params.h"
#include "nvblox/experimental/ground_plane/ransac_plane_fitter_params.h"
#include "nvblox/utils/params.h"

namespace nvblox {

// ======= MASK PRE-PROCESSING =======
constexpr Param<int>::Description kConnectedMaskComponentSizeThresholdParamDesc{
    "connected_mask_component_size_threshold", 2000,
    "Connected components smaller than this threshold will be removed from the "
    "mask image."};

constexpr Param<bool>::Description kRemoveSmallConnectedComponentsParamDesc{
    "remove_small_connected_components", true,
    "If set to true, small connected components will be removed from the "
    "segmentation mask."};

constexpr Param<bool>::Description kExperimentalUseGroundPlaneEstimationDesc{
    "experimental_use_ground_plane_estimation", false,
    "If set to true, estimate the ground plane and use it to slice the ESDF."};

constexpr Param<float>::Description
    kSegmentationMaskModeProximityThresholdParamDesc{
        "segmentation_mask_mode_proximity_threshold", 0.2f,
        "Distance in meters to the depth histogram mode for a depth "
        "pixel to be classified as masked."};

/// A structure containing the multi-mapper parameters.
struct MultiMapperParams {
  Param<int> connected_mask_component_size_threshold{
      kConnectedMaskComponentSizeThresholdParamDesc};
  Param<bool> remove_small_connected_components{
      kRemoveSmallConnectedComponentsParamDesc};
  float segmentation_mask_mode_proximity_threshold =
      kSegmentationMaskModeProximityThresholdParamDesc.default_value;
  Param<bool> experimental_use_ground_plane_estimation{
      kExperimentalUseGroundPlaneEstimationDesc};

  RansacPlaneFitterParams ransac_plane_fitter_params;
  GroundPlaneEstimatorParams ground_plane_estimator_params;
};

}  // namespace nvblox
