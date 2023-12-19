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
#include "nvblox/core/log_odds.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/integrators/internal/decay_integrator_base.h"
#include "nvblox/integrators/internal/decayer.h"
#include "nvblox/map/common_names.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

/// Decay a Tsdf layer. The decay operation for a TsdfLayer reduces the weight,
/// until (optionally) a block is deallocated or set to some (freespace)
/// distance.
class TsdfDecayIntegrator : public DecayIntegratorBase<TsdfLayer> {
 public:
  static constexpr float kDefaultTsdfDecayFactor = 0.95;
  static constexpr bool kDefaultSetFreeDistanceOnDecayed = false;
  static constexpr float kDefaultDecayedWeightThreshold = 1e-3;
  static constexpr float kDefaultFreeDistanceVox = 4.0;

  explicit TsdfDecayIntegrator(DecayMode decay_mode = kDefaultDecayMode);
  virtual ~TsdfDecayIntegrator() = default;

  TsdfDecayIntegrator(const TsdfDecayIntegrator&) = delete;
  TsdfDecayIntegrator& operator=(const TsdfDecayIntegrator&) const = delete;
  TsdfDecayIntegrator(TsdfDecayIntegrator&&) = delete;
  TsdfDecayIntegrator& operator=(const TsdfDecayIntegrator&&) const = delete;

  /// Decay all blocks. Fully decayed blocks (weight close to zero) will be
  /// deallocated if deallocate_decayed_blocks is true.
  ///
  /// @param layer_ptr    Layer to decay
  /// @param cuda_stream  Cuda stream for GPU work
  /// @return A vector containing the indices of the blocks deallocated.
  virtual std::vector<Index3D> decay(TsdfLayer* layer_ptr,
                                     const CudaStream cuda_stream) override;

  /// Decay blocks. Blocks to decay can be excluded based on block index and/or
  /// distance to point.
  ///
  /// @param layer_ptr                 Layer to decay
  /// @param block_exclusion_options   Blocks to be excluded from decay
  /// @param cuda_stream               Cuda stream for GPU work
  /// @return A vector containing the indices of the blocks deallocated.
  virtual std::vector<Index3D> decay(
      TsdfLayer* layer_ptr,
      const DecayBlockExclusionOptions& block_exclusion_options,
      const CudaStream cuda_stream) override;

  /// Decay blocks. Voxels can be excluded based on being in view.
  /// @param layer_ptr              Layer to decay
  /// @param view_exclusion_options Specifies view in which to exclude voxels
  /// @param cuda_stream            Cuda stream for GPU work.
  /// @return A vector containing the indices of the blocks deallocated.
  virtual std::vector<Index3D> decay(
      TsdfLayer* layer_ptr,
      const DecayViewExclusionOptions& view_exclusion_options,
      const CudaStream cuda_stream) override;

  /// Decay blocks. Optional block and voxel view exclusion.
  /// @param layer_ptr               Layer to decay
  /// @param block_exclusion_options Specifies blocks to be excluded from decay
  /// @param view_exclusion_options  Specifies view in which to exclude voxels
  /// @param cuda_stream             Cuda stream for GPU work.
  /// @return A vector containing the indices of the blocks deallocated.
  virtual std::vector<Index3D> decay(
      TsdfLayer* layer_ptr,
      const std::optional<DecayBlockExclusionOptions>& block_exclusion_options,
      const std::optional<DecayViewExclusionOptions>& view_exclusion_options,
      const CudaStream cuda_stream) override;

  /// A parameter getter for the decay factor used to decay the weights
  /// @returns the occupied region decay probability
  float decay_factor() const;

  /// A parameter setter for the decay factor used to decay the weights
  /// @param value the new occupied region decay probability.
  /// @pre 0.0 < param < 1.0
  void decay_factor(const float value);

  /// A parameter getter for the decayed weight threshold.
  /// @returns The weight at which we declare that a voxel is fully decayed.
  float decayed_weight_threshold() const;

  /// A parameter setter for the decayed weight threshold.
  /// @param decayed_weight_threshold The weight at which we declare that a
  /// voxel is full decayed.
  void decayed_weight_threshold(const float decayed_weight_threshold);

  /// A parameter getter for the set distance on decayed flag.
  /// @return A flag indicating if we will set the tsdf distance of fully
  /// decayed voxels to the distance specified by the free_distance_vox.
  bool set_free_distance_on_decayed() const;

  /// A parameter setter for the set distance on decayed flag.
  /// @param set_free_distance_on_decayed A flag indicating if we will set the
  /// tsdf distance of fully decayed voxels to the distance specified by the
  /// free_distance_vox.
  void set_free_distance_on_decayed(const bool set_free_distance_on_decayed);

  /// A parameter getter for the free distance.
  /// @returns The distance in voxels which we set fully decayed voxels to when
  /// set_free_distance_on_decayed is true.
  float free_distance_vox() const;

  /// A parameter setter for the free distance.
  /// @param free_distance_vox The distance in voxels which we set fully decayed
  /// voxels to when set_free_distance_on_decayed is true.
  void free_distance_vox(const float free_distance_vox);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 private:
  // The decayer which performs the decay
  VoxelDecayer<TsdfLayer> decayer_;

  // Exponential decay factor
  float decay_factor_{kDefaultTsdfDecayFactor};

  // The TSDF weight threshold. When a voxel is decayed to this threshold:
  // 1) weight decay stops, and
  // 2) A voxel is set to the free_distance_vox (if set distance_on_decayed is
  // true). If the deallocate_decayed_blocks is true and all voxels in a block
  // reach this threshold, the block is deallocated.
  float decayed_weight_threshold_{kDefaultDecayedWeightThreshold};

  // The distance a TSDF voxel obtains on fully decayed, if
  // set_free_distance_on_decayed is true.
  bool set_free_distance_on_decayed_{kDefaultSetFreeDistanceOnDecayed};

  // The distance (in voxels) that voxels are set to once they're fully decayed
  // AND set_free_distance_on_decayed is true. Should be equal or greater than
  // the truncation distance.
  float free_distance_vox_{kDefaultFreeDistanceVox};
};

}  // namespace nvblox
