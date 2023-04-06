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

#include "nvblox/core/types.h"
#include "nvblox/sensors/image.h"

namespace nvblox {
namespace test_utils {

void linearInterpolateImageGpu(const DepthImage& image,
                               const std::vector<Vector2f>& u_px_vec,
                               std::vector<float>* values_ptr,
                               std::vector<int>* success_flags_ptr);

void linearInterpolateImageGpu(const ColorImage& image,
                               const std::vector<Vector2f>& u_px_vec,
                               std::vector<Color>* values_ptr,
                               std::vector<int>* success_flags_ptr);

}  // namespace test_utils
}  // namespace nvblox
