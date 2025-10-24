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

#include <nvblox/integrators/tsdf_decay_integrator.h>

#include "nvblox/integrators/internal/cuda/impl/decayer_impl.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/lidar.h"

namespace nvblox {
struct TsdfDecayFunctor {
  __host__ __device__ TsdfDecayFunctor(float decay_factor,
                                       float decayed_weight_threshold,
                                       bool set_free_distance_on_decayed,
                                       float free_distance_m)
      : decay_factor_(decay_factor),
        decayed_weight_threshold_(decayed_weight_threshold),
        set_free_distance_on_decayed_(set_free_distance_on_decayed),
        free_distance_m_(free_distance_m) {}

  /// Return true if the passed voxel is fully decayed
  /// @param voxel_ptr The voxel to check
  /// @return True if fully decayed
  __device__ bool isFullyDecayed(TsdfVoxel* voxel_ptr) const {
    constexpr float kEps = 1e-6;
    return (voxel_ptr->weight < (decayed_weight_threshold_ + kEps));
  }

  __host__ __device__ ~TsdfDecayFunctor() = default;

  /// Decays a single TSDF voxel.
  /// @param voxel_ptr voxel to decay
  __device__ void operator()(TsdfVoxel* voxel_ptr) const {
    // Load the weight from global memory
    float weight = voxel_ptr->weight;

    // We only touch voxels which are above or equal to the
    // decayed_weight_threshold. We assume voxels below are unobserved.
    constexpr float kEps = 1e-6;
    if (weight < (decayed_weight_threshold_ - kEps)) {
      return;
    }

    // Decay the voxel.
    weight *= decay_factor_;
    weight = fmaxf(weight, decayed_weight_threshold_);

    // Write weight out to global memory
    voxel_ptr->weight = weight;

    // Check for fully decayed, if not return
    // If voxel *is* fully decayed, update distance to free (if requested)
    if (set_free_distance_on_decayed_ && isFullyDecayed(voxel_ptr)) {
      voxel_ptr->distance = free_distance_m_;
    }
  }

 protected:
  // Params
  float decay_factor_;
  float decayed_weight_threshold_;
  bool set_free_distance_on_decayed_;
  float free_distance_m_;
};

template <typename SensorType>
std::vector<Index3D> TsdfDecayIntegrator::decay(
    TsdfLayer* layer_ptr,
    const std::optional<DecayBlockExclusionOptions>& block_exclusion_options,
    const std::optional<DepthObservationSpace<SensorType>>&
        view_exclusion_options,
    const CudaStream& cuda_stream) {
  // Build the functor which decays a single voxel.
  const float free_distance_m = free_distance_vox_ * layer_ptr->voxel_size();
  TsdfDecayFunctor voxel_decayer(decay_factor_, decayed_weight_threshold_,
                                 set_free_distance_on_decayed_,
                                 free_distance_m);

  // Sanity check the parameters
  if (set_free_distance_on_decayed_ && deallocate_decayed_blocks_) {
    LOG(WARNING) << "Both \"set_free_distance_on_decayed\", and "
                    "\"deallocate_decayed_blocks\" are set true. These flags "
                    "have conflicting effects";
  }

  // Run it on all voxels
  return decayer_.decay(layer_ptr, voxel_decayer, deallocate_decayed_blocks_,
                        block_exclusion_options, view_exclusion_options,
                        cuda_stream);
}

}  // namespace nvblox
