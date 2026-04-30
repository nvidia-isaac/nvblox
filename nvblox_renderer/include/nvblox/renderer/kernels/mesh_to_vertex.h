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

#include <cstdint>

#include "nvblox/core/cuda_stream.h"

namespace nvblox {
namespace renderer {

/// Mesh vertex structure for rendering (position + color + UV).
struct MeshVertex {
  float x, y, z;       // Position
  uint8_t r, g, b, a;  // Color (RGBA)
  float u, v;          // UV coordinates (-1,-1 = use vertex color)
};

/// Interleave separate position, color, and UV arrays into vertex format.
///
/// nvblox stores mesh vertex positions and colors in separate arrays
/// (struct-of-arrays), but Vulkan expects a single interleaved vertex buffer
/// (array-of-structs). This kernel converts between the two layouts on the GPU,
/// writing directly into the CUDA/Vulkan shared buffer so no additional copy
/// is required before rendering.
///
/// @param positions CUDA device pointer to float positions (x, y, z per
/// vertex).
/// @param colors CUDA device pointer to uint8 colors (r, g, b per vertex).
/// @param uvs CUDA device pointer to float UVs (u, v per vertex), or nullptr
///            if no UVs (all UVs set to -1,-1).
/// @param output CUDA device pointer to output MeshVertex array.
/// @param num_vertices Number of vertices.
/// @param stream CUDA stream.
void interleaveMeshVertexData(const float* positions, const uint8_t* colors,
                              MeshVertex* output, size_t num_vertices,
                              const CudaStream& stream,
                              const float* uvs = nullptr);

}  // namespace renderer
}  // namespace nvblox
