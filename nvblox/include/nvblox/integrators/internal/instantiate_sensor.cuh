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

#include "nvblox/dynamics/internal/cuda/impl/dynamics_detection_impl.cuh"
#include "nvblox/integrators/internal/cuda/impl/freespace_integrator_impl.cuh"
#include "nvblox/integrators/internal/cuda/impl/occupancy_decay_integrator_impl.cuh"
#include "nvblox/integrators/internal/cuda/impl/projective_occupancy_integrator_impl.cuh"
#include "nvblox/integrators/internal/cuda/impl/projective_tsdf_integrator_impl.cuh"
#include "nvblox/integrators/internal/cuda/impl/tsdf_decay_integrator_impl.cuh"
#include "nvblox/integrators/internal/cuda/impl/view_calculator_impl.cuh"
#include "nvblox/semantics/internal/cuda/impl/image_projector_impl.cuh"

// Macro to instantiate all template functions for a given SensorType
#define NVBLOX_INSTANTIATE_SENSOR(SensorType)                                \
  namespace nvblox {                                                         \
                                                                             \
  template void ProjectiveOccupancyIntegrator::integrateFrame<SensorType>(   \
      const MaskedDepthImageConstView& depth_frame, const Transform& T_L_C,  \
      const SensorType& camera, OccupancyLayer* layer,                       \
      std::vector<Index3D>* updated_blocks);                                 \
                                                                             \
  template std::vector<Index3D> ViewCalculator::getBlocksInImageViewRaycast< \
      SensorType>(const MaskedDepthImageConstView&, const Transform&,        \
                  const SensorType&, const float, const float, const float); \
                                                                             \
  template void FreespaceIntegrator::updateFreespaceLayer<SensorType>(       \
      const std::vector<Index3D>& block_indices_to_update,                   \
      Time update_time_ms, const TsdfLayer& tsdf_layer,                      \
      const std::optional<DepthObservationSpace<SensorType>>& view,          \
      FreespaceLayer* freespace_layer_ptr);                                  \
                                                                             \
  template std::vector<Index3D> OccupancyDecayIntegrator::decay<SensorType>( \
      OccupancyLayer * layer_ptr,                                            \
      const std::optional<DecayBlockExclusionOptions>&                       \
          block_exclusion_options,                                           \
      const std::optional<DepthObservationSpace<SensorType>>&                \
          view_exclusion_options,                                            \
      const CudaStream& cuda_stream);                                        \
                                                                             \
  template std::vector<Index3D> TsdfDecayIntegrator::decay<SensorType>(      \
      TsdfLayer * layer_ptr,                                                 \
      const std::optional<DecayBlockExclusionOptions>&                       \
          block_exclusion_options,                                           \
      const std::optional<DepthObservationSpace<SensorType>>&                \
          view_exclusion_options,                                            \
      const CudaStream& cuda_stream);                                        \
                                                                             \
  template void ProjectiveTsdfIntegrator::integrateFrame<SensorType>(        \
      const MaskedDepthImageConstView& depth_frame, const Transform& T_L_C,  \
      const SensorType& camera, TsdfLayer* layer,                            \
      std::vector<Index3D>* updated_blocks);                                 \
                                                                             \
  template void DepthImageBackProjector::backProjectOnGPU<SensorType>(       \
      const DepthImage&, const SensorType&, Pointcloud*, const float);       \
                                                                             \
  template void DynamicsDetection::computeDynamics<SensorType>(              \
      const DepthImage& depth_frame_C,                                       \
      const FreespaceLayer& freespace_layer_L, const SensorType& sensor,     \
      const Transform& T_L_C);                                               \
  }  // namespace nvblox
