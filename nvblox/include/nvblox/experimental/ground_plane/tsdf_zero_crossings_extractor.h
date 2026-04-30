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

#include <vector>

#include "nvblox/core/indexing.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/map/common_names.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

class TsdfZeroCrossingsExtractor {
 public:
  TsdfZeroCrossingsExtractor() = delete;
  TsdfZeroCrossingsExtractor(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~TsdfZeroCrossingsExtractor() = default;

  /// @brief Computes the zero-crossings in the TSDF layer using the GPU.
  /// @param tsdf_layer The input TSDF layer from which zero-crossings are to be
  /// computed.
  /// @return If successfull, the tsdf zero crossings from positive to negative
  /// are returned. Otherwise nullopt if the maximum buffer limit was reached.
  std::optional<std::vector<Vector3f>> computeZeroCrossingsFromAboveOnGPU(
      const TsdfLayer& tsdf_layer);

  /// @brief Set the minimum weight to consider a tsdf voxel for computation.
  void min_tsdf_weight(float min_tsdf_weight);
  /// @brief Get the minimum weight to consider a tsdf voxel for computation.
  float min_tsdf_weight() const { return min_tsdf_weight_; }

  /// @brief Set the maximum number that can be stored internally.
  void max_crossings(int max_crossings);
  /// @brief Get the maximum number that can be stored internally.
  int max_crossings() const { return max_crossings_; }

  /// @brief Set the buffer expansion factor for auto resize.
  /// Default: 2.0 (doubles the buffer size when resizing).
  void set_buffer_expansion_factor(float factor) {
    buffer_expansion_factor_ = factor;
  }

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 private:
  /// @brief Resets and allocates buffers
  void resetAndAllocateCrossingBuffers();

  // Buffer counters for the tsdf crossings.
  unified_ptr<int> p_L_crossings_count_device_;
  unified_ptr<int> p_L_crossings_count_host_;

  // Buffer vectors holding the 3d location of found tsdf crossings.
  device_vector<Vector3f> p_L_crossings_global_device_;
  host_vector<Vector3f> p_L_crossings_global_host_;

  // Buffer pointers
  device_vector<const TsdfBlock*> block_ptrs_device_;
  device_vector<const TsdfBlock*> block_ptrs_above_device_;
  device_vector<Index3D> block_indices_device_;

  // Initial maximum number of crossings to store. The internal buffer holding
  // the zero crossings is initialized to this size. If the limit is reached
  // during computation, the buffer will automatically resize using
  // buffer_expansion_factor.
  int max_crossings_ = 360000;
  // Expansion factor for buffer resizing.
  float buffer_expansion_factor_ = 2.0f;

  // The minimum tsdf weight for a voxel to be considered a candidate.
  float min_tsdf_weight_ = 0.1;

  // The CUDA stream on which processing occurs
  std::shared_ptr<CudaStream> cuda_stream_;
};

}  // namespace nvblox
