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
#include "nvblox/tests/gpu_indexing.h"

#include <cuda_runtime.h>

#include "nvblox/core/indexing.h"
#include "nvblox/core/unified_vector.h"

namespace nvblox {
namespace test_utils {

__global__ void getBlockAndVoxelIndexFromPositionInLayerKernel(
    const Vector3f position, const float block_size, Index3D* block_idx,
    Index3D* voxel_idx) {
  getBlockAndVoxelIndexFromPositionInLayer(block_size, position, block_idx,
                                           voxel_idx);
}

__global__ void getBlockAndVoxelIndexFromPositionInLayerKernel(
    const Vector3f* positions, const float block_size, const int num_positions,
    Index3D* block_indices, Index3D* voxel_indices) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_positions) {
    getBlockAndVoxelIndexFromPositionInLayer(
        block_size, positions[idx], &block_indices[idx], &voxel_indices[idx]);
  }
}

void getBlockAndVoxelIndexFromPositionInLayerOnGPU(const float block_size,
                                                   const Vector3f& position,
                                                   Index3D* block_idx,
                                                   Index3D* voxel_idx) {
  Index3D* block_idx_device;
  Index3D* voxel_idx_device;
  checkCudaErrors(cudaMalloc(&block_idx_device, sizeof(Index3D)));
  checkCudaErrors(cudaMalloc(&voxel_idx_device, sizeof(Index3D)));

  getBlockAndVoxelIndexFromPositionInLayerKernel<<<1, 1>>>(
      position, block_size, block_idx_device, voxel_idx_device);

  checkCudaErrors(cudaMemcpy(block_idx, block_idx_device, sizeof(Index3D),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(voxel_idx, voxel_idx_device, sizeof(Index3D),
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(block_idx_device));
  checkCudaErrors(cudaFree(voxel_idx_device));
}

void getBlockAndVoxelIndexFromPositionInLayerOnGPU(
    const float block_size, const std::vector<Vector3f>& positions,
    std::vector<Index3D>* block_indices, std::vector<Index3D>* voxel_indices) {
  device_vector<Vector3f> positions_device;
  positions_device.copyFrom(positions);

  device_vector<Index3D> block_indices_device(positions.size());
  device_vector<Index3D> voxel_indices_device(positions.size());

  constexpr int kNumThreads = 1024;
  const int kNumBlocks = positions.size() / kNumThreads + 1;

  getBlockAndVoxelIndexFromPositionInLayerKernel<<<kNumBlocks, kNumThreads>>>(
      positions_device.data(), block_size, positions_device.size(),
      block_indices_device.data(), voxel_indices_device.data());

  *block_indices = block_indices_device.toVector();
  *voxel_indices = voxel_indices_device.toVector();
}

}  // namespace test_utils
}  // namespace nvblox