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

#include <memory>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

/// A class to mask images based on a binary mask.
class ImageMasker {
 public:
  ImageMasker();
  ImageMasker(std::shared_ptr<CudaStream> cuda_stream);
  ~ImageMasker() = default;

  /// Splitting the input color image according to a mask image
  /// assuming color and mask images are coming from the same camera.
  ///@param input Color image to be split according to mask.
  ///@param mask  Mask image.
  ///@param unmasked_output Color image containing the color values of all
  ///                       unmasked input pixels.
  ///                       Masked pixels are set to black.
  ///@param masked_output   Optional. Color image containing the color values
  ///                       of all masked input pixels.
  ///                       Unmasked pixels are set to black.
  void splitImageOnGPU(const ColorImage& input, const MonoImage& mask,
                       ColorImage* unmasked_output, ColorImage* masked_output,
                       ColorImage* masked_color_overlay = nullptr);

  /// Splitting the input depth image according to a mask image taking occlusion
  /// into account.
  ///@param depth_input Depth image to be split according to mask.
  ///@param mask        Mask image.
  ///@param T_CM_CD Transform from depth camera to mask camera frame.
  ///@param depth_camera Intrinsics model of the depth camera.
  ///@param mask_camera  Intrinsics model of the mask camera.
  ///@param unmasked_depth_output Depth image containing the depth values
  ///                             of all unmasked input pixels.
  ///                             Masked pixels are set to -1.
  ///@param masked_depth_output   Optional. Depth image containing the depth
  ///                             values of all masked input pixels.
  ///                             Unmasked pixels are set to -1.
  void splitImageOnGPU(const DepthImage& depth_input, const MonoImage& mask,
                       const Transform& T_CM_CD, const Camera& depth_camera,
                       const Camera& mask_camera,
                       DepthImage* unmasked_depth_output,
                       DepthImage* masked_depth_output,
                       ColorImage* masked_depth_overlay = nullptr);

  /// A parameter getter
  /// The occlusion threshold parameter associated with the image splitter.
  /// A point is considered to be occluded on the mask image only if it lies
  /// more than the occlusion threshold behind the point occluding it.
  /// Occluded points are always assumed to be unmasked.
  /// @returns the occlusion threshold in meters
  float occlusion_threshold_m() const;

  /// A parameter setter
  /// See occlusion_threshold_m().
  /// @param occlusion_threshold_m the occlusion threshold in meters.
  void occlusion_threshold_m(float occlusion_threshold_m);

  /// A parameter getter
  /// The float value that gets written to unmasked pixels on the
  /// masked depth output image.
  /// @returns the invalid pixel value for the masked depth output image
  float depth_masked_image_invalid_pixel() const;

  /// A parameter setter
  /// see depth_masked_image_invalid_pixel()
  /// @param the new parameter value
  void depth_masked_image_invalid_pixel(float value);

  /// A parameter getter
  /// The float value that gets written to masked pixels on the
  /// unmasked depth output image.
  /// @returns the invalid pixel value for the unmasked depth output image
  float depth_unmasked_image_invalid_pixel() const;

  /// A parameter setter
  /// see depth_unmasked_image_invalid_pixel()
  /// @param the new parameter value
  void depth_unmasked_image_invalid_pixel(float value);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 private:
  template <typename ImageType>
  void allocateOutput(const ImageType& input, ImageType* unmasked_output,
                      ImageType* masked_output,
                      ColorImage* overlay_output = nullptr);

  // Params
  float occlusion_threshold_m_ = 0.25;
  float depth_masked_image_invalid_pixel_ = -1.0;
  float depth_unmasked_image_invalid_pixel_ = -1.0;
  Color color_masked_image_invalid_pixel_ = Color(0, 0, 0, 0);
  Color color_unmasked_image_invalid_pixel_ = Color(0, 0, 0, 0);

  std::shared_ptr<CudaStream> cuda_stream_;
};

}  // namespace nvblox
