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

#include "nvblox/core/blox.h"
#include "nvblox/core/camera.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/image.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/types.h"
#include "nvblox/core/voxels.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/integrators/projective_integrator_base.h"
#include "nvblox/integrators/view_calculator.h"

namespace nvblox {

/// A class performing TSDF intregration
///
/// Integrates a depth images into TSDF layers. The "projective" is a describes
/// one type of integration. Namely that voxels in view are projected into the
/// depth image (the alternative being casting rays out from the camera).
class ProjectiveTsdfIntegrator : public ProjectiveIntegratorBase {
 public:
  ProjectiveTsdfIntegrator();
  virtual ~ProjectiveTsdfIntegrator();

  /// Integrates a depth image in to the passed TSDF layer.
  /// @param depth_frame A depth image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param camera A the camera (intrinsics) model.
  /// @param layer A pointer to the layer into which this observation will be
  /// intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  void integrateFrame(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Camera& camera, TsdfLayer* layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// Blocks until GPU operations are complete
  /// Ensure outstanding operations are finished (relevant for integrators
  /// launching asynchronous work)
  void finish() const override;

 protected:
  // Given a set of blocks in view (block_indices) perform TSDF updates on all
  // voxels within these blocks on the GPU.
  virtual void updateBlocks(const std::vector<Index3D>& block_indices,
                            const DepthImage& depth_frame,
                            const Transform& T_L_C, const Camera& camera,
                            const float truncation_distance_m,
                            TsdfLayer* layer);

  // Blocks to integrate on the current call (indices and pointers)
  // NOTE(alexmillane): We have one pinned host and one device vector and
  // transfer between them.
  device_vector<Index3D> block_indices_device_;
  device_vector<TsdfBlock*> block_ptrs_device_;
  host_vector<Index3D> block_indices_host_;
  host_vector<TsdfBlock*> block_ptrs_host_;

  // CUDA stream to process ingration on
  cudaStream_t integration_stream_;
};

}  // namespace nvblox
