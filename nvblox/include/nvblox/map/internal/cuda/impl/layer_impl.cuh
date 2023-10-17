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

#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/map/layer.h"

namespace nvblox {

template <typename VoxelType>
__global__ void queryVoxelsKernel(
    int num_queries, Index3DDeviceHashMapType<VoxelBlock<VoxelType>> block_hash,
    float block_size, const Vector3f* query_locations_ptr,
    VoxelType* voxels_ptr, bool* success_flags_ptr) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_queries) {
    return;
  }
  const Vector3f query_location = query_locations_ptr[idx];

  VoxelType* voxel;
  if (!getVoxelAtPosition<VoxelType>(block_hash, query_location, block_size,
                                     &voxel)) {
    success_flags_ptr[idx] = false;
  } else {
    success_flags_ptr[idx] = true;
    voxels_ptr[idx] = *voxel;
  }
}

template <typename VoxelType>
void VoxelBlockLayer<VoxelType>::getVoxelsGPU(
    const device_vector<Vector3f>& positions_L,
    device_vector<VoxelType>* voxels_ptr,
    device_vector<bool>* success_flags_ptr) const {
  // Call the underlying streamed method on a newly created stream.
  CudaStreamOwning cuda_stream;
  getVoxelsGPU(positions_L, voxels_ptr, success_flags_ptr, &cuda_stream);
}

template <typename VoxelType>
void VoxelBlockLayer<VoxelType>::getVoxelsGPU(
    const device_vector<Vector3f>& positions_L,
    device_vector<VoxelType>* voxels_ptr,
    device_vector<bool>* success_flags_ptr, CudaStream* cuda_stream_ptr) const {
  CHECK_NOTNULL(voxels_ptr);
  CHECK_NOTNULL(success_flags_ptr);
  CHECK_NOTNULL(cuda_stream_ptr);

  const int num_queries = positions_L.size();
  voxels_ptr->resizeAsync(num_queries, *cuda_stream_ptr);
  success_flags_ptr->resizeAsync(num_queries, *cuda_stream_ptr);

  constexpr int kNumThreads = 512;
  const int num_blocks = num_queries / kNumThreads + 1;

  queryVoxelsKernel<VoxelType>
      <<<num_blocks, kNumThreads, 0, *cuda_stream_ptr>>>(
          num_queries, this->getGpuLayerView().getHash().impl_,
          this->block_size_, positions_L.data(), voxels_ptr->data(),
          success_flags_ptr->data());

  cuda_stream_ptr->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

}  // namespace nvblox
