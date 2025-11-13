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
#include "nvblox/utils/cuda_kernel_utils.h"

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
  device_vector<Index3D> device_indices;
  device_indices.copyFromAsync(indices, CudaStreamOwning());

  // Output space
  device_vector<bool> device_flags(device_indices.size());

  // Kernel
  constexpr int kNumThreadsPerBlock = 32;
  const int num_blocks =
      divideRoundUp(device_indices.size(), kNumThreadsPerBlock);
  getContainsFlagsKernel<<<num_blocks, kNumThreadsPerBlock>>>(
      gpu_layer.getHash().impl_, device_indices.data(), device_indices.size(),
      device_flags.data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  // GPU -> CPU
  std::vector<bool> host_flags = device_flags.toVectorAsync(CudaStreamOwning());
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

void getVoxelsAtPositionsOnGPU(const GPULayerView<TsdfBlock>& gpu_layer,
                               const std::vector<Vector3f>& p_L_vec,
                               host_vector<TsdfVoxel>* host_voxels,
                               host_vector<bool>* host_flags,
                               const float block_size) {
  // CPU -> GPU
  device_vector<Vector3f> device_positions;
  device_positions.copyFromAsync(p_L_vec, CudaStreamOwning());

  // Output space
  device_vector<TsdfVoxel> device_voxels(device_positions.size());
  device_vector<bool> device_flags(device_positions.size());
  device_voxels.setZeroAsync(CudaStreamOwning());
  device_flags.setZeroAsync(CudaStreamOwning());

  // Kernel
  constexpr int kNumThreadsPerBlock = 32;
  const int num_blocks =
      divideRoundUp(device_positions.size(), kNumThreadsPerBlock);
  getVoxelsAtPositionsKernel<<<num_blocks, kNumThreadsPerBlock>>>(
      gpu_layer.getHash().impl_, device_positions.data(), block_size,
      device_positions.size(), device_voxels.data(), device_flags.data());
  checkCudaErrors(cudaPeekAtLastError());

  // GPU -> CPU
  host_voxels->copyFromAsync(device_voxels, CudaStreamOwning());
  host_flags->copyFromAsync(device_flags, CudaStreamOwning());
}

}  // namespace test_utils
}  // namespace nvblox
