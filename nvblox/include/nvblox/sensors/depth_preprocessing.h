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
#include "nvblox/sensors/npp_image_operations.h"

namespace nvblox {

/// A class which implements preprocessing steps for an input depth image. At
/// the moment this is only expanding the invalid depth regions.
class DepthPreprocessor {
 public:
  DepthPreprocessor();
  DepthPreprocessor(std::shared_ptr<CudaStream> cuda_stream);
  ~DepthPreprocessor() = default;

  /// @brief Dilates the invalid region in a depth image N times.
  /// @param num_dilations The number of times to apply a 3x3 dilation.
  /// @param depth_image_ptr The image to be dilated.
  void dilateInvalidRegionsAsync(const int num_dilations,
                                 DepthImage* depth_image_ptr);

  /// A parameter getter.
  /// The depth value, below which a depth pixel is considered to be invalid.
  /// @return The threshold.
  float invalid_depth_threshold() const;

  /// A parameter setter
  /// See invalid_depth_threshold()
  /// @param invalid_depth_threshold The threshold
  void invalid_depth_threshold(float invalid_depth_threshold);

  /// A parameter getter.
  /// The value which we set pixels to that are invalid.
  /// @return The invalid value.
  float invalid_depth_value() const;

  /// A parameter setter
  /// See invalid_depth_value()
  /// @param invalid_depth_value The value.
  void invalid_depth_value(float invalid_depth_value);

 private:
  // The value below which we deem pixels to be invalid in a depth image.
  float invalid_depth_threshold_ = 1e-2f;

  // The value we set pixels to which they are invalid.
  float invalid_depth_value_ = 0.f;

  // Scratch space for images to avoid reallocating
  MonoImage mask_{MemoryType::kDevice};

  // This image and mask form a double frame buffer for performing repeated
  // dilations.
  MonoImage mask_dilated_tmp_{MemoryType::kDevice};

  // Streams on which we process work. The npp_stream_context, contains a
  // reference to cuda_stream_ internally.
  std::shared_ptr<CudaStream> cuda_stream_;
  NppStreamContext npp_stream_context_;
};

}  // namespace nvblox
