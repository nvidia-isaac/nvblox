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
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"

namespace nvblox {

// Must be called with:
// - a single block
// - one thread per voxel
__global__ void setColorBlockGray(ColorBlock* block_device_ptr) {
  ColorVoxel* voxel_ptr =
      &block_device_ptr->voxels[threadIdx.z][threadIdx.y][threadIdx.x];
  // NOTE(alexmillane): We don't use alpha channel for blocks (yet). So we just
  // set 0 here.
  voxel_ptr->color.r = 127;
  voxel_ptr->color.g = 127;
  voxel_ptr->color.b = 127;
  voxel_ptr->color.a = 0;
  voxel_ptr->weight = 0.0f;
}

void setColorBlockGrayOnGPUAsync(ColorBlock* block_device_ptr,
                                 const CudaStream& cuda_stream) {
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  setColorBlockGray<<<1, kThreadsPerBlock, 0, cuda_stream>>>(block_device_ptr);
  checkCudaErrors(cudaPeekAtLastError());
}

}  // namespace nvblox
