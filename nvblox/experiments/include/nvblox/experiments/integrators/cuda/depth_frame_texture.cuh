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

#include <cuda_runtime.h>

#include "nvblox/core/image.h"

namespace nvblox {
namespace experiments {

class DepthImageTexture {
 public:
  DepthImageTexture(const DepthImage& depth_frame, cudaStream_t transfer_stream = 0);
  ~DepthImageTexture();

  cudaTextureObject_t texture_object() const { return depth_texture_; }

 private:
  cudaArray_t depth_array_;
  cudaTextureObject_t depth_texture_;
};

} // namespace experiments
} // namespace nvblox
