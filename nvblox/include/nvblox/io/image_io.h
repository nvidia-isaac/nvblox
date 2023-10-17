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

#include <string>

#include "nvblox/core/types.h"
#include "nvblox/sensors/image.h"

namespace nvblox {
namespace io {

// NOTE(alexmillane): Due to the limitations of stb_image(_write) we load images
// as 16bit unsigned integers, but save images as uint8 images, after scaling by
// the maximum depth. This means if you save a png and then load it back, you'll
// get a different result.

// Default scaling taking uint16 depth to float depth
constexpr float kDefaultUintDepthScaleFactor = 1.0f / 1000.0f;

bool writeToPng(const std::string& filepath, const DepthImage& frame);
bool writeToPng(const std::string& filepath, const MonoImage& frame);
bool writeToPng(const std::string& filepath, const ColorImage& frame);

bool readFromPng(const std::string& filepath, DepthImage* frame_ptr,
                 const float scale_factor = kDefaultUintDepthScaleFactor);
bool readFromPng(const std::string& filepath, MonoImage* frame_ptr);
bool readFromPng(const std::string& filepath, ColorImage* frame_ptr);

}  // namespace io
}  // namespace nvblox
