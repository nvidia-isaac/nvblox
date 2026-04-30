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
#include "nvblox/mesh/mesh_transform.h"

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/internal/error_check.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/utils/cuda_kernel_utils.h"

namespace nvblox {

__global__ void transformVerticesKernel(const Transform T_out_in,
                                        int num_vertices,
                                        const Vector3f* vertices_in,
                                        Vector3f* vertices_out) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= num_vertices) {
    return;
  }

  vertices_out[index] = T_out_in * vertices_in[index];
}

__global__ void transformNormalsKernel(const Matrix3f rotation, int num_normals,
                                       const Vector3f* normals_in,
                                       Vector3f* normals_out) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= num_normals) {
    return;
  }

  normals_out[index] = rotation * normals_in[index];
}

void transformMeshOnGPU(const Transform& T_out_in,
                        unified_vector<Vector3f>* vertices,
                        unified_vector<Vector3f>* normals,
                        CudaStream* cuda_stream_ptr) {
  CHECK_NOTNULL(cuda_stream_ptr);
  CHECK_NOTNULL(vertices);
  CHECK_NOTNULL(normals);

  if (vertices->empty()) {
    return;
  }

  // Transform vertices (full transform: rotation + translation)
  constexpr int kThreadsPerThreadBlock = 512;
  const int num_blocks_vertices =
      divideRoundUp(static_cast<int>(vertices->size()), kThreadsPerThreadBlock);
  transformVerticesKernel<<<num_blocks_vertices, kThreadsPerThreadBlock, 0,
                            *cuda_stream_ptr>>>(
      T_out_in, static_cast<int>(vertices->size()), vertices->data(),
      vertices->data());
  checkCudaErrors(cudaPeekAtLastError());

  // Transform normals (rotation only, no translation)
  if (!normals->empty()) {
    CHECK_EQ(normals->size(), vertices->size())
        << "Normals count must match vertices count";
    const int num_blocks_normals = divideRoundUp(
        static_cast<int>(normals->size()), kThreadsPerThreadBlock);
    transformNormalsKernel<<<num_blocks_normals, kThreadsPerThreadBlock, 0,
                             *cuda_stream_ptr>>>(
        T_out_in.linear(), static_cast<int>(normals->size()), normals->data(),
        normals->data());
    checkCudaErrors(cudaPeekAtLastError());
  }
}

}  // namespace nvblox
