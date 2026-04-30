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

#include "nvblox/integrators/weighting_function.h"
#include "nvblox/utils/params.h"

namespace nvblox {

// ======= SHARED PARAMS =======
constexpr Param<
    float>::Description kProjectiveIntegratorMaxIntegrationDistanceMParamDesc{
    "projective_integrator_max_integration_distance_m", 7.f,
    "The maximum distance, in meters, to integrate the depth or color image "
    "values."};

constexpr Param<float>::Description
    kLidarProjectiveIntegratorMaxIntegrationDistanceMParamDesc{
        "lidar_projective_integrator_max_integration_distance_m", 10.f,
        "The maximum distance, in meters, to integrate the depth values for "
        "LiDAR scans."};

constexpr Param<
    float>::Description kProjectiveIntegratorTruncationDistanceVoxParamDesc{
    "projective_integrator_truncation_distance_vox", 4.f,
    "The truncation distance, in units of voxels, for the TSDF or occupancy "
    "map."};

constexpr Param<WeightingFunctionType>::Description
    kProjectiveIntegratorWeightingModeParamDesc{
        "projective_integrator_weighting_mode",
        WeightingFunctionType::kInverseSquareWeight,
        "The weighting mode, applied to TSDF and color integrations.  "
        "Options: [0:constant, 1:constant_dropoff, 2:inverse_square, "
        "3:inverse_square_dropoff, 4:inverse_square_tsdf_distance_penalty]"};

constexpr Param<float>::Description kProjectiveIntegratorMaxWeightParamDesc{
    "projective_integrator_max_weight", 5.f,
    "Maximum weight for the TSDF and color integrations. Setting this number "
    "higher will lead to higher-quality reconstructions but worse "
    "performance in dynamic scenes."};

constexpr Param<
    float>::Description kProjectiveTsdfIntegratorInvalidDepthDecayFactor{
    "projective_tsdf_integrator_invalid_depth_decay_factor", -1.0,
    "Whenever a voxel projects into an invalid (<=0) depth image pixel, we "
    "decay the voxel with this factor instead of integrating. This allow us to "
    "rapidly prune outliers stemming from dynamic objects and/or invalid "
    "sensor data. A negative value for this parameter disables the effect,"
    "i.e. no decay takes place."};

constexpr Param<float>::Description
    kProjectiveAppearanceIntegratorMeasurementWeightParamDesc{
        "projective_appearance_integrator_measurement_weight", 0.8f,
        "How much weight that should be given to the measurement when fusing "
        "with an existing estimate in color and feature integrators."
        "With alpha as the value of this parameter, a new estimate x is "
        "computed "
        "as follows:"
        "x_new = alpha * x_measured + (1 - alpha) * x_old"};

constexpr Param<float>::Description
    kProjectiveDynamicTsdfIntegratorDiscrepancyThresholdMParamDesc{
        "projective_dynamic_tsdf_integrator_discrepancy_threshold_m", -1.0f,
        "The discrepancy threshold in meters. When a voxel's stored TSDF "
        "distance differs from the current depth observation by more than "
        "this threshold, the voxel is invalidated (TSDF and weight set to "
        "zero). "
        "A negative value disables the dynamic discrepancy check."};

constexpr Param<float>::Description
    kProjectiveDynamicTsdfIntegratorDynamicDiscrepancyMinWeightParamDesc{
        "projective_dynamic_tsdf_integrator_dynamic_discrepancy_min_weight",
        2.0f,
        "Minimum voxel weight required before the dynamic discrepancy "
        "check is applied. Voxels below this weight are treated as "
        "establishing and are fused normally without comparison."};

struct ProjectiveIntegratorParams {
  Param<float> projective_integrator_max_integration_distance_m{
      kProjectiveIntegratorMaxIntegrationDistanceMParamDesc};
  Param<float> lidar_projective_integrator_max_integration_distance_m{
      kLidarProjectiveIntegratorMaxIntegrationDistanceMParamDesc};
  Param<float> projective_integrator_truncation_distance_vox{
      kProjectiveIntegratorTruncationDistanceVoxParamDesc};
  Param<WeightingFunctionType> projective_integrator_weighting_mode{
      kProjectiveIntegratorWeightingModeParamDesc};
  Param<float> projective_integrator_max_weight{
      kProjectiveIntegratorMaxWeightParamDesc};
  Param<float> projective_tsdf_integrator_invalid_depth_decay_factor{
      kProjectiveTsdfIntegratorInvalidDepthDecayFactor};
  Param<float> projective_appearance_integrator_measurement_weight{
      kProjectiveAppearanceIntegratorMeasurementWeightParamDesc};
  Param<float> projective_dynamic_tsdf_integrator_discrepancy_threshold_m{
      kProjectiveDynamicTsdfIntegratorDiscrepancyThresholdMParamDesc};
  Param<float>
      projective_dynamic_tsdf_integrator_dynamic_discrepancy_min_weight{
          kProjectiveDynamicTsdfIntegratorDynamicDiscrepancyMinWeightParamDesc};
};

}  // namespace nvblox
