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
#include "nvblox/experiments/integrators/cuda/depth_frame_texture.cuh"

#include "nvblox/core/cuda/error_check.cuh"

namespace nvblox {
namespace experiments {

DepthImageTexture::DepthImageTexture(const DepthImage& depth_frame,
                                     cudaStream_t transfer_stream) {
  // Note(alexmillane): Taken from texture memory example
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory

  // Allocate CUDA array in device memory
  // Each channel is a 32bit float in the first (x) dimension
  const cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  checkCudaErrors(cudaMallocArray(&depth_array_, &channelDesc,
                                  depth_frame.width(), depth_frame.height()));

  // Set pitch of the source (the width in memory in bytes of the 2D array
  // pointed to by src, including padding), we dont have any padding
  const size_t spitch = depth_frame.width() * sizeof(float);
  // Copy data located at address h_data in host memory to device memory
  checkCudaErrors(cudaMemcpy2DToArrayAsync(
      depth_array_, 0, 0, depth_frame.dataConstPtr(), spitch,
      depth_frame.width() * sizeof(float), depth_frame.height(),
      cudaMemcpyDefault, transfer_stream));

  // Specify texture
  struct cudaResourceDesc resource_description;
  memset(&resource_description, 0, sizeof(resource_description));
  resource_description.resType = cudaResourceTypeArray;
  resource_description.res.array.array = depth_array_;

  // Specify texture object parameters
  struct cudaTextureDesc texture_description;
  memset(&texture_description, 0, sizeof(texture_description));
  texture_description.addressMode[0] = cudaAddressModeClamp;
  texture_description.addressMode[1] = cudaAddressModeClamp;
  texture_description.filterMode = cudaFilterModeLinear;
  texture_description.readMode = cudaReadModeElementType;
  texture_description.normalizedCoords = 0;

  // Create texture object
  checkCudaErrors(cudaCreateTextureObject(
      &depth_texture_, &resource_description, &texture_description, NULL));
}

DepthImageTexture::~DepthImageTexture() {
  cudaDestroyTextureObject(depth_texture_);
  cudaFreeArray(depth_array_);
}

}  // namespace experiments
}  // namespace nvblox
