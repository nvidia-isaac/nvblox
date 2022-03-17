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
#include "nvblox/core/image.h"

#include <algorithm>

#include "nvblox/core/cuda/image_cuda.h"
#include "nvblox/io/csv.h"

namespace nvblox {
namespace image {

float max(const DepthImage& image) { return maxGPU(image); }

float min(const DepthImage& image) { return minGPU(image); }

std::pair<float, float> minmax(const DepthImage& image) {
  return minmaxGPU(image);
}

float maxGPU(const DepthImage& image) {
  return cuda::max(image.rows(), image.cols(), image.dataConstPtr());
}

float minGPU(const DepthImage& image) {
  return cuda::min(image.rows(), image.cols(), image.dataConstPtr());
}

std::pair<float, float> minmaxGPU(const DepthImage& image) {
  const auto minmax_elem =
      cuda::minmax(image.rows(), image.cols(), image.dataConstPtr());
  return {minmax_elem.first, minmax_elem.second};
}

float maxCPU(const DepthImage& image) {
  CHECK(image.memory_type() == MemoryType::kUnified)
      << "CPU function called on kDevice image.";
  return *std::max_element(image.dataConstPtr(),
                           image.dataConstPtr() + image.numel());
}

float minCPU(const DepthImage& image) {
  CHECK(image.memory_type() == MemoryType::kUnified)
      << "CPU function called on kDevice image.";
  return *std::min_element(image.dataConstPtr(),
                           image.dataConstPtr() + image.numel());
}

std::pair<float, float> minmaxCPU(const DepthImage& image) {
  CHECK(image.memory_type() == MemoryType::kUnified)
      << "CPU function called on kDevice image.";
  const auto res = std::minmax_element(image.dataConstPtr(),
                                       image.dataConstPtr() + image.numel());
  return {*res.first, *res.second};
}

}  // namespace image
}  // namespace nvblox
