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

#include "nvblox/core/common_names.h"
#include "nvblox/core/layer.h"
#include "nvblox/integrators/projective_integrator_base.h"
#include "nvblox/ray_tracing/sphere_tracer.h"

namespace nvblox {

class ProjectiveColorIntegrator : public ProjectiveIntegratorBase {
 public:
  ProjectiveColorIntegrator();
  virtual ~ProjectiveColorIntegrator();

  void finish() const override;

  // Main interface - integrate a ColorImage
  void integrateFrame(const ColorImage& color_frame, const Transform& T_L_C,
                      const Camera& camera, const TsdfLayer& tsdf_layer,
                      ColorLayer* color_layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

 protected:
  void updateBlocks(const std::vector<Index3D>& block_indices,
                    const ColorImage& color_frame,
                    const DepthImage& depth_frame, const Transform& T_L_C,
                    const Camera& camera, const float truncation_distance_m,
                    ColorLayer* layer);

  std::vector<Index3D> reduceBlocksToThoseInTruncationBand(
      const std::vector<Index3D>& block_indices, const TsdfLayer& tsdf_layer,
      const float truncation_distance_m);

  // Params
  int depth_render_ray_subsampling_factor_ = 4;

  // Object to do ray tracing to generate occlusions
  // TODO(alexmillane): Expose params.
  SphereTracer sphere_tracer_;

  // Blocks to integrate on the current call and their indices
  // NOTE(alexmillane): We have one pinned host and one device vector and
  // transfer between them.
  device_vector<Index3D> block_indices_device_;
  device_vector<ColorBlock*> block_ptrs_device_;
  host_vector<Index3D> block_indices_host_;
  host_vector<ColorBlock*> block_ptrs_host_;

  // Buffers for getting blocks in truncation band
  device_vector<const TsdfBlock*> truncation_band_block_ptrs_device_;
  host_vector<const TsdfBlock*> truncation_band_block_ptrs_host_;
  device_vector<bool> block_in_truncation_band_device_;
  host_vector<bool> block_in_truncation_band_host_;

  // CUDA stream to process ingration on
  cudaStream_t integration_stream_;
};

}  // namespace nvblox