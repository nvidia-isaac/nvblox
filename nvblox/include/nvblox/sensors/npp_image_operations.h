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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/sensors/image.h"

#include <npp.h>

namespace nvblox {
namespace image {

/// @brief Prints NPP version info to the console for debug.
void printNPPVersionInfo();

/// @brief Converts a CUDA stream to an NPP stream context.
/// Note that we use whatever device is currently in use.
/// @param cuda_stream Cuda stream to process NPP calls on
/// @return The NPP stream context.
NppStreamContext getNppStreamContext(const CudaStream& cuda_stream);

/// @brief Generates a mask image which is true where depth values are invalid.
/// Note that we require that the output image is allocated and has the same
/// size as the input.
/// @param depth_image Depth image in which to detect invalid depths
/// @param npp_stream_context The NPP stream context on which to process
/// @param mask_ptr The output mask image
/// @param invalid_threshold The threshold below which we consider a depth pixel
/// invalid
void getInvalidDepthMaskAsync(const DepthImage& depth_image,
                              const NppStreamContext& npp_stream_context,
                              MonoImage* mask_ptr,
                              const float invalid_threshold = 1e-2);

/// @brief Generates a new mask image which is a 3x3 dilation of the input mask.
/// Note that we require that the output image is allocated and has the same
/// size as the input.
/// @param mask_image Input mask
/// @param npp_stream_context The NPP stream context on which to process
/// @param mask_dilated_ptr Output dilated mask
void dilateMask3x3Async(const MonoImage& mask_image,
                        const NppStreamContext& npp_stream_context,
                        MonoImage* mask_dilated_ptr);

/// @brief Set depth image elements to a value where the input mask is >0.
/// Note that we require that the output image is allocated and has the same
/// size as the input.
/// @param mask The mask
/// @param value The value to set float pixels to
/// @param npp_stream_context The NPP stream context on which to process
/// @param depth_image_ptr The depth image to modify
void maskedSetAsync(const MonoImage& mask, const float value,
                    const NppStreamContext& npp_stream_context,
                    DepthImage* depth_image_ptr);

/// @brief Set all pixels strictly above a threshold to a given value
/// Note that we require that the output image is allocated and has the same
/// size as the input.
/// @param image Input image
/// @param threshold The threshold
/// @param value Value which will replace everything above threshold
/// @param npp_stream_context The NPP stream context on which to process
/// @pram image_thresholded Output image
void setGreaterThanThresholdToValue(const MonoImage& image,
                                    const uint8_t threshold,
                                    const uint8_t value,
                                    const NppStreamContext& npp_stream_context,
                                    MonoImage* image_thresholded);

}  // namespace image
}  // namespace nvblox
