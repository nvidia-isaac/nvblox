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

#include "nvblox/mapper/mapper.h"

namespace nvblox {

/// The MultiMapper class is composed of two standard Mappers.
/// Depth and color are integrated into one of these Mappers according to a
/// mask.
class MultiMapper {
 public:
  MultiMapper(float voxel_size_m, MemoryType memory_type = MemoryType::kDevice,
              ProjectiveLayerType masked_projective_layer_type =
                  ProjectiveLayerType::kOccupancy,
              ProjectiveLayerType unmasked_projective_layer_type =
                  ProjectiveLayerType::kTsdf);
  ~MultiMapper() = default;

  /// Integrates a depth frame into the reconstruction
  /// using the transformation between depth and mask frame and their
  /// intrinsics.
  ///@param depth_frame Depth frame to integrate. Depth in the image is
  ///                   specified as a float representing meters.
  ///@param mask Mask. Interpreted as 0=non-masked, >0=masked.
  ///@param T_L_CD Pose of the depth camera, specified as a transform from
  ///              camera frame to layer frame transform.
  ///@param T_CM_CD Transform from depth camera to mask camera frame.
  ///@param depth_camera Intrinsics model of the depth camera.
  ///@param mask_camera Intrinsics model of the mask camera.
  void integrateDepth(const DepthImage& depth_frame, const MonoImage& mask,
                      const Transform& T_L_CD, const Transform& T_CM_CD,
                      const Camera& depth_camera, const Camera& mask_camera);

  /// Integrates a color frame into the reconstruction.
  ///@param color_frame Color image to integrate.
  ///@param mask Mask. Interpreted as 0=non-masked, >0=masked.
  ///@param T_L_C Pose of the camera, specified as a transform from camera frame
  ///             to the layer frame.
  ///@param camera Intrinsics model of the camera.
  void integrateColor(const ColorImage& color_frame, const MonoImage& mask,
                      const Transform& T_L_C, const Camera& camera);

  // Access to the internal mappers
  const Mapper& unmasked_mapper() const { return *unmasked_mapper_.get(); }
  const Mapper& masked_mapper() const { return *masked_mapper_.get(); }
  std::shared_ptr<Mapper>& unmasked_mapper() { return unmasked_mapper_; }
  std::shared_ptr<Mapper>& masked_mapper() { return masked_mapper_; }

  // These functions return a reference to the masked images generated during
  // the preceeding calls to integrateColor() and integrateDepth().
  const DepthImage& getLastDepthFrameUnmasked();
  const DepthImage& getLastDepthFrameMasked();
  const ColorImage& getLastColorFrameUnmasked();
  const ColorImage& getLastColorFrameMasked();
  const ColorImage& getLastDepthFrameMaskOverlay();
  const ColorImage& getLastColorFrameMaskOverlay();

  /// Setting the depth value for unmasked depth pixels on the masked output
  /// frame.
  ///@param depth_value The new depth value.
  void setDepthMaskedImageInvalidPixel(float depth_value);

  /// Setting the depth value for masked depth pixels on the unmasked output
  /// frame.
  ///@param depth_value The new depth value.
  void setDepthUnmaskedImageInvalidPixel(float depth_value);

 protected:
  // Split depth images based on a mask.
  // Note that we internally pre-allocate space for the split images on the
  // first call.
  ImageMasker image_masker_;
  DepthImage depth_frame_unmasked_;
  DepthImage depth_frame_masked_;
  ColorImage color_frame_unmasked_;
  ColorImage color_frame_masked_;

  // Mask overlays used as debug outputs
  ColorImage masked_depth_overlay_;
  ColorImage masked_color_overlay_;

  // The two mappers to which the frames are integrated.
  std::shared_ptr<Mapper> unmasked_mapper_;
  std::shared_ptr<Mapper> masked_mapper_;
};

}  // namespace nvblox
