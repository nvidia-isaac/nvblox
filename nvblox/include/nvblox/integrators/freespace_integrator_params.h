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

#include "nvblox/utils/params.h"

namespace nvblox {

constexpr Param<float>::Description kMaxTsdfDistanceForOccupancyMParamDesc{
    "max_tsdf_distance_for_occupancy_m", .15f,
    "TSDF distance below which we assume a voxel to be occupied (non "
    "freespace)."};

constexpr Param<
    Time>::Description kMaxUnobservedToKeepConsecutiveOccupancyMsParamDesc{
    "max_unobserved_to_keep_consecutive_occupancy_ms", Time(200),
    "Maximum duration of no observed occupancy to keep consecutive occupancy "
    "alive."};

constexpr Param<Time>::Description
    kMinDurationSinceOccupiedForFreespaceMsParamDesc{
        "min_duration_since_occupied_for_freespace_ms", Time(1000),
        "Minimum duration since last observed occupancy to consider voxel as "
        "free."};

constexpr Param<Time>::Description
    kMinConsecutiveOccupancyDurationForResetMsParamDesc{
        "min_consecutive_occupancy_duration_for_reset_ms", Time(2000),
        "Minimum duration of consecutive occupancy to turn a high confidence "
        "free voxel back to occupied."};

constexpr Param<bool>::Description kCheckNeighborhoodParamDesc{
    "check_neighborhood", true,
    "Whether to check the occupancy of the neighboring voxels for the high "
    "confidence freespace update."};

constexpr Param<bool>::Description
    kInitializeToHighConfidenceFreespaceParamDesc{
        "initialize_to_high_confidence_freespace", false,
        "Whether to initialize voxels to high confidence freespace."};

struct FreespaceIntegratorParams {
  Param<float> max_tsdf_distance_for_occupancy_m{
      kMaxTsdfDistanceForOccupancyMParamDesc};
  Param<Time> max_unobserved_to_keep_consecutive_occupancy_ms{
      kMaxUnobservedToKeepConsecutiveOccupancyMsParamDesc};
  Param<Time> min_duration_since_occupied_for_freespace_ms{
      kMinDurationSinceOccupiedForFreespaceMsParamDesc};
  Param<Time> min_consecutive_occupancy_duration_for_reset_ms{
      kMinConsecutiveOccupancyDurationForResetMsParamDesc};
  Param<bool> check_neighborhood{kCheckNeighborhoodParamDesc};
  Param<bool> initialize_to_high_confidence_freespace{
      kInitializeToHighConfidenceFreespaceParamDesc};
};

}  // namespace nvblox
