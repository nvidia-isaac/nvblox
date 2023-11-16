/*
Copyright 2023 NVIDIA CORPORATION

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

#include <bitset>
#include <memory>

#include "nvblox/sensors/image.h"
#include "nvblox/utils/logging.h"

namespace nvblox {
namespace image {

/// Removal of small connected components from mask image
///
/// @attention Resulting mask will be non-zero for active pixels, but not
///            necessarily 255.
///
/// @param mask            Target image. Any non-zero pixels are masked
/// @param size_threshold  Keep only components with num pixels more than this
/// @param mask_out        Output image
/// @param cuda_stream     Stream for GPU work
void removeSmallConnectedComponents(const MonoImage& mask,
                                    const int size_threshold,
                                    MonoImage* mask_out,
                                    const CudaStream cuda_stream);

}  // namespace image
};  // namespace nvblox
