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
#include "nvblox/sensors/depth_preprocessing.h"

namespace nvblox {

/// Reallocates an output image to be the same size as the input image.
/// Note that the output image has the same memory type before and after
/// re-allocation.
/// @param input The input image from which we take the size.
/// @param output A pointer to image to be potentially reallocated.
template <typename InputImageType, typename OutputImageType>
void reallocateImageToSameSizeIfRequired(const InputImageType& input,
                                         OutputImageType* output) {
  CHECK_NOTNULL(output);
  if (input.rows() != output->rows() || input.cols() != output->cols()) {
    *output =
        OutputImageType(input.rows(), input.cols(), output->memory_type());
  }
}

DepthPreprocessor::DepthPreprocessor()
    : DepthPreprocessor(std::make_shared<CudaStreamOwning>()) {}

DepthPreprocessor::DepthPreprocessor(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {
  npp_stream_context_ = image::getNppStreamContext(*cuda_stream_);
}

float DepthPreprocessor::invalid_depth_threshold() const {
  return invalid_depth_threshold_;
}

void DepthPreprocessor::invalid_depth_threshold(float invalid_depth_threshold) {
  invalid_depth_threshold_ = invalid_depth_threshold;
};

float DepthPreprocessor::invalid_depth_value() const {
  return invalid_depth_value_;
}

void DepthPreprocessor::invalid_depth_value(float invalid_depth_value) {
  invalid_depth_value_ = invalid_depth_value;
};

void DepthPreprocessor::dilateInvalidRegionsAsync(const int num_dilations,
                                                  DepthImage* depth_image_ptr) {
  CHECK_NOTNULL(depth_image_ptr);
  CHECK_GE(num_dilations, 0);
  CHECK_GE(depth_image_ptr->rows(), 3);
  CHECK_GE(depth_image_ptr->cols(), 3);
  // Check for the trivial case
  if (num_dilations == 0) {
    LOG(WARNING) << "Request to dilate 0 times. Doing nothing.";
  }
  // Allocate image space if required
  reallocateImageToSameSizeIfRequired(*depth_image_ptr, &mask_);
  reallocateImageToSameSizeIfRequired(*depth_image_ptr, &mask_dilated_tmp_);

  // Get the invalid region mask
  image::getInvalidDepthMaskAsync(*depth_image_ptr, npp_stream_context_, &mask_,
                                  invalid_depth_threshold_);

  // Dilate the number of times requested
  MonoImage* dilation_in_ptr = &mask_;
  MonoImage* dilation_out_ptr = &mask_dilated_tmp_;
  for (int i = 0; i < num_dilations; i++) {
    image::dilateMask3x3Async(*dilation_in_ptr, npp_stream_context_,
                              dilation_out_ptr);
    std::swap(dilation_in_ptr, dilation_out_ptr);
  }
  // We undo the last swap such that the output ptr points back to the output.
  std::swap(dilation_in_ptr, dilation_out_ptr);

  // Set the masked regions invalid
  image::maskedSetAsync(*dilation_out_ptr, invalid_depth_value_,
                        npp_stream_context_, depth_image_ptr);
}

}  // namespace nvblox
