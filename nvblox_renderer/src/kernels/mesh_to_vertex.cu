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

#include "nvblox/renderer/kernels/mesh_to_vertex.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include "nvblox/core/internal/error_check.h"

namespace nvblox {
namespace renderer {

namespace {

// CUDA kernel to interleave position, color, and UV data into Vertex format
// positions: float array (x, y, z per vertex - 3 floats each)
// colors: uint8 array (r, g, b per vertex - 3 bytes each)
// uvs: float array (u, v per vertex - 2 floats each), or nullptr
// output: Vertex array (float3 position + uint8_t rgba + float2 uv)
__global__ void interleaveVertexDataKernel(const float* positions,
                                           const uint8_t* colors,
                                           const float* uvs, MeshVertex* output,
                                           size_t num_vertices) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_vertices) return;

  // Position is stored as 3 floats per vertex
  output[idx].x = positions[idx * 3 + 0];
  output[idx].y = positions[idx * 3 + 1];
  output[idx].z = positions[idx * 3 + 2];

  // Color is stored as 3 bytes per vertex (RGB)
  output[idx].r = colors[idx * 3 + 0];
  output[idx].g = colors[idx * 3 + 1];
  output[idx].b = colors[idx * 3 + 2];
  output[idx].a = 255;  // Full opacity

  // UV coordinates (2 floats per vertex, or -1,-1 if no UVs)
  if (uvs) {
    output[idx].u = uvs[idx * 2 + 0];
    output[idx].v = uvs[idx * 2 + 1];
  } else {
    output[idx].u = -1.0f;
    output[idx].v = -1.0f;
  }
}

}  // namespace

void interleaveMeshVertexData(const float* positions, const uint8_t* colors,
                              MeshVertex* output, size_t num_vertices,
                              const CudaStream& stream, const float* uvs) {
  if (num_vertices == 0) return;

  // Validate required pointers (uvs may be nullptr)
  if (!positions || !colors || !output) {
    LOG(ERROR) << "Null pointer passed to interleaveMeshVertexData";
    return;
  }

  constexpr int kBlockSize = 256;
  int num_blocks = (num_vertices + kBlockSize - 1) / kBlockSize;
  cudaStream_t cuda_stream = stream;
  interleaveVertexDataKernel<<<num_blocks, kBlockSize, 0, cuda_stream>>>(
      positions, colors, uvs, output, num_vertices);

  // Check for kernel launch errors (fatal)
  checkCudaErrors(cudaGetLastError());
}

}  // namespace renderer
}  // namespace nvblox
