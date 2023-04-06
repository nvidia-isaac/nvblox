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
#include "nvblox/io/layer_cake_io.h"

#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/mapper/multi_mapper.h"

namespace nvblox {

MultiMapper::MultiMapper(float voxel_size_m, MemoryType memory_type,
                         ProjectiveLayerType masked_projective_layer_type,
                         ProjectiveLayerType unmasked_projective_layer_type)
    : masked_mapper_(std::make_shared<Mapper>(voxel_size_m, memory_type,
                                              masked_projective_layer_type)),
      unmasked_mapper_(std::make_shared<Mapper>(
          voxel_size_m, memory_type, unmasked_projective_layer_type)) {}

void MultiMapper::integrateDepth(const DepthImage& depth_frame,
                                 const MonoImage& mask, const Transform& T_L_CD,
                                 const Transform& T_CM_CD,
                                 const Camera& depth_camera,
                                 const Camera& mask_camera) {
  // Split masked and non masked depth frame
  image_masker_.splitImageOnGPU(depth_frame, mask, T_CM_CD, depth_camera,
                                mask_camera, &depth_frame_unmasked_,
                                &depth_frame_masked_, &masked_depth_overlay_);

  // Integrate the frames to the respective layer cake
  unmasked_mapper_->integrateDepth(depth_frame_unmasked_, T_L_CD, depth_camera);
  masked_mapper_->integrateDepth(depth_frame_masked_, T_L_CD, depth_camera);
}

void MultiMapper::integrateColor(const ColorImage& color_frame,
                                 const MonoImage& mask, const Transform& T_L_C,
                                 const Camera& camera) {
  // Split masked and non masked color frame
  image_masker_.splitImageOnGPU(color_frame, mask, &color_frame_unmasked_,
                                &color_frame_masked_, &masked_color_overlay_);

  // Integrate the frames to the respective layer cake
  unmasked_mapper_->integrateColor(color_frame_unmasked_, T_L_C, camera);
  masked_mapper_->integrateColor(color_frame_masked_, T_L_C, camera);
}

const DepthImage& MultiMapper::getLastDepthFrameUnmasked() {
  return depth_frame_unmasked_;
}
const DepthImage& MultiMapper::getLastDepthFrameMasked() {
  return depth_frame_masked_;
}
const ColorImage& MultiMapper::getLastColorFrameUnmasked() {
  return color_frame_unmasked_;
}
const ColorImage& MultiMapper::getLastColorFrameMasked() {
  return color_frame_masked_;
}
const ColorImage& MultiMapper::getLastDepthFrameMaskOverlay() {
  return masked_depth_overlay_;
}
const ColorImage& MultiMapper::getLastColorFrameMaskOverlay() {
  return masked_color_overlay_;
}
void MultiMapper::setDepthMaskedImageInvalidPixel(float depth_value) {
  image_masker_.depth_masked_image_invalid_pixel(depth_value);
}

void MultiMapper::setDepthUnmaskedImageInvalidPixel(float depth_value) {
  image_masker_.depth_unmasked_image_invalid_pixel(depth_value);
}

}  // namespace nvblox
