/*
Copyright 2026 NVIDIA CORPORATION

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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"

namespace nvblox {

/// @brief Transform mesh vertices and normals on GPU.
/// @param T_out_in Transform from input frame to output frame.
/// @param vertices Unified vector of vertices (will be modified in-place).
/// @param normals Unified vector of normals (will be modified in-place).
/// @param cuda_stream_ptr CUDA stream for asynchronous execution.
void transformMeshOnGPU(const Transform& T_out_in,
                        unified_vector<Vector3f>* vertices,
                        unified_vector<Vector3f>* normals,
                        CudaStream* cuda_stream_ptr);

}  // namespace nvblox
