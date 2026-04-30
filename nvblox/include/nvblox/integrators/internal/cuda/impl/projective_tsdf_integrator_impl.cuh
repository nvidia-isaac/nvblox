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

#include <nvblox/integrators/projective_tsdf_integrator.h>
#include <nvblox/integrators/internal/cuda/impl/projective_integrator_impl.cuh>

#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/occupancy_integrator_params.h"

namespace nvblox {
struct UpdateTsdfVoxelFunctor {
  __host__ __device__ UpdateTsdfVoxelFunctor() = default;
  __host__ __device__ ~UpdateTsdfVoxelFunctor() = default;

  // Vector3f p_voxel_C, float depth, TsdfVoxel* voxel_ptr
  __device__ bool operator()(const float surface_depth_measured,
                             const float voxel_depth_m, const bool is_active,
                             TsdfVoxel* voxel_ptr) {
    // Ignore invalid (negative) depth measurements.
    if (surface_depth_measured <= 0.F) {
      if (invalid_depth_decay_factor_ >= 0.F) {
        // Invalid depth pixels are decayed aggressively
        voxel_ptr->weight *= invalid_depth_decay_factor_;
      }
      return false;
    }

    // Get the distance between the voxel we're updating the surface.
    // Note that the distance is the projective distance, i.e. the distance
    // along the ray.
    const float voxel_to_surface_distance =
        surface_depth_measured - voxel_depth_m;

    // If we're behind the negative truncation distance, just continue.
    if (voxel_to_surface_distance < -truncation_distance_m_) {
      return false;
    }

    // Dynamic discrepancy check (disabled when threshold < 0).
    // Voxels (with weight >= dynamic_discrepancy_min_weight_) whose
    // clamped projective distance disagrees with the stored TSDF value
    // by more than the threshold are reset.
    // Voxels behind the surface are left untouched.
    if (dynamic_discrepancy_threshold_m_ >= 0.0f &&
        voxel_ptr->weight >= dynamic_discrepancy_min_weight_) {
      const float clamped_distance =
          fmin(truncation_distance_m_,
               fmax(-truncation_distance_m_, voxel_to_surface_distance));
      const float discrepancy = fabs(clamped_distance - voxel_ptr->distance);
      if (discrepancy > dynamic_discrepancy_threshold_m_) {
        voxel_ptr->weight = 0.0f;
        voxel_ptr->distance = 0.0f;
      }
    }

    // Handle inactive depth pixels. We do not want to integrate
    // them, but we still want to clear any voxels in front of the surface. We
    // therefore integrate only up until the positive truncation distance.
    if (!is_active && voxel_to_surface_distance < truncation_distance_m_) {
      return false;
    }

    // Read CURRENT voxel values (from global GPU memory)
    const float voxel_distance_current = voxel_ptr->distance;
    const float voxel_weight_current = voxel_ptr->weight;

    // NOTE(alexmillane): We could try to use CUDA math functions to speed up
    // below
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE

    // Get the weight of this observation from the sensor model.
    const float measurement_weight = weighting_function_(
        surface_depth_measured, voxel_depth_m, truncation_distance_m_);

    // Fuse
    float fused_distance = (voxel_to_surface_distance * measurement_weight +
                            voxel_distance_current * voxel_weight_current) /
                           (measurement_weight + voxel_weight_current);

    // Clip
    if (fused_distance > 0.0f) {
      fused_distance = fmin(truncation_distance_m_, fused_distance);
    } else {
      fused_distance = fmax(-truncation_distance_m_, fused_distance);
    }
    const float weight =
        fmin(measurement_weight + voxel_weight_current, max_weight_);

    // Write NEW voxel values (to global GPU memory)
    voxel_ptr->distance = fused_distance;
    voxel_ptr->weight = weight;
    return true;
  }

  float truncation_distance_m_ = 0.2f;
  float max_weight_ = kProjectiveIntegratorMaxWeightParamDesc.default_value;
  float invalid_depth_decay_factor_ =
      kProjectiveIntegratorMaxWeightParamDesc.default_value;
  float dynamic_discrepancy_threshold_m_ =
      kProjectiveDynamicTsdfIntegratorDiscrepancyThresholdMParamDesc
          .default_value;
  float dynamic_discrepancy_min_weight_ =
      kProjectiveDynamicTsdfIntegratorDynamicDiscrepancyMinWeightParamDesc
          .default_value;

  WeightingFunction weighting_function_ =
      kProjectiveIntegratorWeightingModeParamDesc.default_value;
};

template <typename SensorType>
void ProjectiveTsdfIntegrator::integrateFrame(
    const MaskedDepthImageConstView& depth_frame, const Transform& T_L_C,
    const SensorType& sensor, TsdfLayer* layer,
    std::vector<Index3D>* updated_blocks) {
  // Get the update functor on the device
  unified_ptr<UpdateTsdfVoxelFunctor> update_functor_device_ptr =
      getTsdfUpdateFunctorOnDevice(layer->voxel_size());
  // Integrate
  ProjectiveIntegrator<TsdfVoxel>::integrateFrame(
      depth_frame, T_L_C, sensor, update_functor_device_ptr.get(), layer,
      updated_blocks);
}

}  // namespace nvblox
