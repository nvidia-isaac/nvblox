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
#include "nvblox/tests/gpu_layer_utils.h"

#include <thrust/device_vector.h>

#include "nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh"
#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"

namespace nvblox {
namespace test_utils {

__global__ void getContainsFlagsKernel(
    Index3DDeviceHashMapType<TsdfBlock> block_hash, Index3D* indices,
    int num_indices, bool* flags) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_indices) {
    flags[idx] = block_hash.contains(indices[idx]);
  }
}

std::vector<bool> getContainsFlags(const GPULayerView<TsdfBlock>& gpu_layer,
                                   const std::vector<Index3D>& indices) {
  // CPU -> GPU
  thrust::device_vector<Index3D> device_indices(indices);

  // Output space
  thrust::device_vector<bool> device_flags(device_indices.size());

  // Kernel
  constexpr int kNumThreadsPerBlock = 32;
  const int num_blocks = device_indices.size() / kNumThreadsPerBlock + 1;
  getContainsFlagsKernel<<<num_blocks, kNumThreadsPerBlock>>>(
      gpu_layer.getHash().impl_, device_indices.data().get(),
      device_indices.size(), device_flags.data().get());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  // GPU -> CPU
  std::vector<bool> host_flags(device_flags.size());
  thrust::copy(device_flags.begin(), device_flags.end(), host_flags.begin());

  return host_flags;
}

__global__ void getVoxelsAtPositionsKernel(
    Index3DDeviceHashMapType<TsdfBlock> block_hash, const Vector3f* p_L_vec,
    float block_size, int num_points, TsdfVoxel* voxels, bool* flags) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_points) {
    const Vector3f p_L = p_L_vec[idx];
    TsdfVoxel* voxel_ptr;
    const bool flag =
        getVoxelAtPosition(block_hash, p_L, block_size, &voxel_ptr);
    flags[idx] = flag;
    if (flag) {
      voxels[idx] = *voxel_ptr;
    }
  }
}

std::pair<std::vector<TsdfVoxel>, std::vector<bool>> getVoxelsAtPositionsOnGPU(
    const GPULayerView<TsdfBlock>& gpu_layer,
    const std::vector<Vector3f>& p_L_vec) {
  // CPU -> GPU
  thrust::device_vector<Vector3f> device_positions(p_L_vec);

  // Output space
  thrust::device_vector<TsdfVoxel> device_voxels(device_positions.size());
  thrust::device_vector<bool> device_flags(device_positions.size());

  // Kernel
  constexpr int kNumThreadsPerBlock = 32;
  const int num_blocks = device_positions.size() / kNumThreadsPerBlock + 1;
  getVoxelsAtPositionsKernel<<<num_blocks, kNumThreadsPerBlock>>>(
      gpu_layer.getHash().impl_, device_positions.data().get(),
      gpu_layer.block_size(), device_positions.size(),
      device_voxels.data().get(), device_flags.data().get());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  // GPU -> CPU
  std::vector<TsdfVoxel> host_voxels(device_voxels.size());
  thrust::copy(device_voxels.begin(), device_voxels.end(), host_voxels.begin());
  std::vector<bool> host_flags(device_flags.size());
  thrust::copy(device_flags.begin(), device_flags.end(), host_flags.begin());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  return {std::move(host_voxels), std::move(host_flags)};
}

}  // namespace test_utils
}  // namespace nvblox
