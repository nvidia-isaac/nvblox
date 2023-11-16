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
#include "nvblox/tests/blox_utils.h"
#include "nvblox/tests/voxels.h"

namespace nvblox {

// The requirement for this second definition of this variable here is a
// language issue with c++14. Can be removed in c++17.
constexpr uint8_t TestBlockNoAllocation::kCPUInitializationValue;

namespace test_utils {

__global__ void setFloatingBlockVoxelsInSequenceKernel(FloatVoxelBlock* block) {
  const int lin_idx =
      threadIdx.x + TsdfBlock::kVoxelsPerSide *
                        (threadIdx.y + TsdfBlock::kVoxelsPerSide * threadIdx.z);
  FloatVoxel* voxel_ptr = &block->voxels[threadIdx.z][threadIdx.y][threadIdx.x];
  voxel_ptr->voxel_data = static_cast<float>(lin_idx);
}

void setFloatingBlockVoxelsInSequence(FloatVoxelBlock::Ptr block) {
  CHECK(block.memory_type() != MemoryType::kHost);
  constexpr int kNumBlocks = 1;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  setFloatingBlockVoxelsInSequenceKernel<<<kNumBlocks, kThreadsPerBlock>>>(
      block.get());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
}

__global__ void setTsdfBlockVoxelsInSequenceKernel(TsdfBlock* block) {
  const int lin_idx =
      threadIdx.x + TsdfBlock::kVoxelsPerSide *
                        (threadIdx.y + TsdfBlock::kVoxelsPerSide * threadIdx.z);
  TsdfVoxel* voxel_ptr = &block->voxels[threadIdx.z][threadIdx.y][threadIdx.x];
  voxel_ptr->distance = static_cast<float>(lin_idx);
  voxel_ptr->weight = static_cast<float>(lin_idx);
}

void setTsdfBlockVoxelsInSequence(TsdfBlock::Ptr block) {
  CHECK(block.memory_type() != MemoryType::kHost);
  constexpr int kNumBlocks = 1;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  setTsdfBlockVoxelsInSequenceKernel<<<kNumBlocks, kThreadsPerBlock>>>(
      block.get());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
}

__global__ void setTsdfBlockVoxelsConstantKernel(TsdfBlock* block,
                                                 float distance) {
  TsdfVoxel* voxel_ptr = &block->voxels[threadIdx.z][threadIdx.y][threadIdx.x];
  voxel_ptr->distance = distance;
}

void setTsdfBlockVoxelsConstant(const float distance, TsdfBlock::Ptr block) {
  CHECK(block.memory_type() != MemoryType::kHost);
  constexpr int kNumBlocks = 1;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  setTsdfBlockVoxelsConstantKernel<<<kNumBlocks, kThreadsPerBlock>>>(
      block.get(), distance);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
}

__global__ void checkBlockAllConstant(const TsdfBlock* block,
                                      const TsdfVoxel* voxel_constant,
                                      bool* flag) {
  if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
    *flag = true;
  }
  __syncthreads();
  const TsdfVoxel voxel = block->voxels[threadIdx.z][threadIdx.y][threadIdx.x];
  const float distance_diff =
      std::abs(voxel.distance - voxel_constant->distance);
  const float weight_diff = std::abs(voxel.weight - voxel_constant->weight);
  constexpr float eps = 1e-4;
  if ((distance_diff > eps) || (weight_diff > eps)) {
    *flag = false;
  }
}

__global__ void checkBlockAllConstant(
    const InitializationTestVoxelBlock* block,
    const InitializationTestVoxel* voxel_constant, bool* flag) {
  if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
    *flag = true;
  }
  __syncthreads();
  const InitializationTestVoxel voxel =
      block->voxels[threadIdx.z][threadIdx.y][threadIdx.x];
  if (voxel.data != voxel_constant->data) {
    *flag = false;
  }
}

__global__ void checkBlockAllConstant(const ColorBlock* block,
                                      const ColorVoxel* voxel_constant,
                                      bool* flag) {
  if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
    *flag = true;
  }
  __syncthreads();
  const ColorVoxel voxel = block->voxels[threadIdx.z][threadIdx.y][threadIdx.x];
  if ((voxel.color.r != voxel_constant->color.r) ||
      (voxel.color.g != voxel_constant->color.g) ||
      (voxel.color.b != voxel_constant->color.b)) {
    *flag = false;
  }
  const float weight_diff = std::abs(voxel.weight - voxel_constant->weight);
  constexpr float eps = 1e-4;
  if (weight_diff > eps) {
    *flag = false;
  }
}

template <typename VoxelType>
bool checkBlockAllConstantTemplate(
    const typename VoxelBlock<VoxelType>::Ptr block, VoxelType voxel_cpu) {
  // Allocate memory for the flag
  bool* flag_device_ptr;
  checkCudaErrors(cudaMalloc(&flag_device_ptr, sizeof(bool)));

  // Transfer the CPU voxel to GPU
  VoxelType* voxel_device_ptr;
  checkCudaErrors(cudaMalloc(&voxel_device_ptr, sizeof(VoxelType)));
  checkCudaErrors(cudaMemcpy(voxel_device_ptr, &voxel_cpu, sizeof(VoxelType),
                             cudaMemcpyHostToDevice));

  constexpr int kNumBlocks = 1;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  checkBlockAllConstant<<<kNumBlocks, kThreadsPerBlock>>>(
      block.get(), voxel_device_ptr, flag_device_ptr);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  // Copy the flag back
  bool flag;
  checkCudaErrors(
      cudaMemcpy(&flag, flag_device_ptr, sizeof(bool), cudaMemcpyDeviceToHost));

  // Free the flag
  checkCudaErrors(cudaFree(flag_device_ptr));
  checkCudaErrors(cudaFree(voxel_device_ptr));

  return flag;
}

bool checkBlockAllConstant(const TsdfBlock::Ptr block, TsdfVoxel voxel_cpu) {
  return checkBlockAllConstantTemplate<TsdfVoxel>(block, voxel_cpu);
}

bool checkBlockAllConstant(const InitializationTestVoxelBlock::Ptr block,
                           InitializationTestVoxel voxel_cpu) {
  return checkBlockAllConstantTemplate<InitializationTestVoxel>(block,
                                                                voxel_cpu);
}

bool checkBlockAllConstant(const ColorBlock::Ptr block, ColorVoxel voxel_cpu) {
  return checkBlockAllConstantTemplate<ColorVoxel>(block, voxel_cpu);
}

}  // namespace test_utils
}  // namespace nvblox
