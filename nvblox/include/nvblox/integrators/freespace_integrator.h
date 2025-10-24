/*
Copyright 2023 NVIDIA CORPORATION

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

#include <optional>

#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/time.h"
#include "nvblox/core/types.h"
#include "nvblox/integrators/depth_observation_space.h"
#include "nvblox/integrators/freespace_integrator_params.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"

namespace nvblox {

/// The FreespaceIntegrator class updates the freespace layer that
/// classifies voxels as high confidence freespace or occupied.
/// Freespace information can be used for dynamic detection as objects moving
/// into freespace can be labelled dynamic.
///
/// The algorithm implemented in this class is based on the dynablox
/// paper: https://ieeexplore.ieee.org/document/10218983
class FreespaceIntegrator {
 public:
  FreespaceIntegrator();
  FreespaceIntegrator(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~FreespaceIntegrator() = default;

  /// @brief Updates a freespace layer according to a tsdf layer.
  /// @param block_indices_to_update The block indices that should be updated.
  /// @param update_time_ms  The current time in miliseconds.
  /// @param tsdf_layer The tsdf layer that is used to check wether voxels
  /// are occupied or not.
  /// @param freespace_layer_ptr The freespace layer that will be updated.
  //   void updateFreespaceLayer(const std::vector<Index3D>&
  //   block_indices_to_update,
  //                             Time update_time_ms, const TsdfLayer&
  //                             tsdf_layer, FreespaceLayer*
  //                             freespace_layer_ptr);

  /// @brief Updates a freespace layer according to a tsdf layer (in view).
  /// @param block_indices_to_update The block indices that should be updated.
  /// @param update_time_ms  The current time in miliseconds.
  /// @param tsdf_layer The tsdf layer that is used to check wether voxels
  /// are occupied or not.
  /// @param view_to_update Describes the view of the scene in which voxels
  /// should be updated.
  /// @param freespace_layer_ptr The freespace layer that will be updated.
  template <typename SensorType>
  void updateFreespaceLayer(
      const std::vector<Index3D>& block_indices_to_update, Time update_time_ms,
      const TsdfLayer& tsdf_layer,
      const std::optional<DepthObservationSpace<SensorType>>& view,
      FreespaceLayer* freespace_layer_ptr);

  /// A parameter getter
  /// Tsdf distance below which we assume a voxel to be occupied.
  /// Note: if set to greater than tsdf truncation distance everything will be
  /// occupied. Corresponds to tau_d in dynablox.
  /// @returns the maximum tsdf distance for occupancy in meters
  float max_tsdf_distance_for_occupancy_m() const;

  /// A parameter setter
  /// See max_tsdf_distance_for_occupancy_m().
  /// @param value the new maximum tsdf distance for occupancy in meters.
  void max_tsdf_distance_for_occupancy_m(float value);

  /// A parameter getter
  /// Maximum duration of no observed occupancy to keep consecutive occupancy
  /// alive. Corresponds to the sparsity compensation duration tau_s in
  /// dynablox.
  /// @returns the maximum unobserved duration in ms to keep consecutive
  /// occupancy
  Time max_unobserved_to_keep_consecutive_occupancy_ms() const;

  /// A parameter setter
  /// See max_unobserved_to_keep_consecutive_occupancy_ms().
  /// @param value the new maximum unobserved duration in ms to keep consecutive
  /// occupancy.
  void max_unobserved_to_keep_consecutive_occupancy_ms(Time value);

  /// A parameter getter
  /// Minimum duration since last observed occupancy to consider voxel as free.
  /// Corresponds to tau_w in dynablox.
  /// @returns the minimum duration since occupied for freespace in ms
  Time min_duration_since_occupied_for_freespace_ms() const;

  /// A parameter setter
  /// See min_duration_since_occupied_for_freespace_ms().
  /// @param value the new minimum duration since occupied for freespace in ms.
  void min_duration_since_occupied_for_freespace_ms(Time value);

  /// A parameter getter
  /// Minimum duration of consecutive occupancy to turn a high confidence free
  /// voxel back to occupied. Corresponds to tau_r in dynablox.
  /// @returns the minimum consecutive occupancy duration for a reset in ms
  Time min_consecutive_occupancy_duration_for_reset_ms() const;

  /// A parameter setter
  /// See min_consecutive_occupancy_duration_for_reset_ms().
  /// @param value the new minimum consecutive occupancy duration for a reset in
  /// ms.
  void min_consecutive_occupancy_duration_for_reset_ms(Time value);

  /// A parameter getter
  /// Whether to check the occupancy of the neighboring voxels for the high
  /// confidence freespace update.
  /// @returns whether to check the neighboring voxels
  bool check_neighborhood() const;

  /// A parameter setter
  /// See check_neighborhood().
  /// @param value whether to check the neighboring voxels
  void check_neighborhood(bool value);

  /// A parameter getter
  /// Whether to initialize voxels to high confidence freespace
  /// @returns whether to initialize to high confidence freespace
  bool initialize_to_high_confidence_freespace() const;

  /// A parameter setter
  /// See initialize_to_high_confidence_freespace().
  /// @param value whether to initialize to high confidence freespace
  void initialize_to_high_confidence_freespace(bool value);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 protected:
  template <typename SensorType>
  void launchKernel(
      Time update_time_ms,
      const std::optional<DepthObservationSpace<SensorType>>& view,
      FreespaceLayer* freespace_layer_ptr);

  // Parameters (see getters for description)
  // Note: See comment behind each parameter for corresponding dynablox
  // parameter name
  float max_tsdf_distance_for_occupancy_m_{
      kMaxTsdfDistanceForOccupancyMParamDesc.default_value};  // tau_d
  Time max_unobserved_to_keep_consecutive_occupancy_ms_{
      kMaxUnobservedToKeepConsecutiveOccupancyMsParamDesc
          .default_value};  // tau_s
  Time min_duration_since_occupied_for_freespace_ms_{
      kMinDurationSinceOccupiedForFreespaceMsParamDesc.default_value};  // tau_w
  Time min_consecutive_occupancy_duration_for_reset_ms_{
      kMinConsecutiveOccupancyDurationForResetMsParamDesc
          .default_value};  // tau_r
  bool check_neighborhood_ = kCheckNeighborhoodParamDesc.default_value;
  bool initialize_to_high_confidence_freespace_ =
      kInitializeToHighConfidenceFreespaceParamDesc.default_value;

  // Time
  Time last_update_time_ms_{0};

  // Block index buffers
  host_vector<Index3D> block_indices_to_update_host_;
  device_vector<Index3D> block_indices_to_update_device_;

  // Block ptr buffers
  host_vector<FreespaceBlock*> freespace_blocks_to_update_host_;
  device_vector<FreespaceBlock*> freespace_blocks_to_update_device_;
  host_vector<const TsdfBlock*> tsdf_blocks_to_update_host_;
  device_vector<const TsdfBlock*> tsdf_blocks_to_update_device_;

  // CUDA stream to process integration on
  std::shared_ptr<CudaStream> cuda_stream_;
};

}  // namespace nvblox
