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
#include "nvblox/mapper/multi_mapper.h"

#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/io/layer_cake_io.h"
#include "nvblox/sensors/connected_components.h"

namespace nvblox {

// The unmasked mapper is handling static objects and freespace (with occupancy
// or tsdf)
ProjectiveLayerType findUnmaskedLayerType(MappingType mapping_type) {
  switch (mapping_type) {
    case MappingType::kStaticTsdf:
      return ProjectiveLayerType::kTsdf;
    case MappingType::kStaticOccupancy:
      return ProjectiveLayerType::kOccupancy;
    case MappingType::kDynamic:
      return ProjectiveLayerType::kTsdf;
    case MappingType::kHumanWithStaticTsdf:
      return ProjectiveLayerType::kTsdf;
    case MappingType::kHumanWithStaticOccupancy:
      return ProjectiveLayerType::kOccupancy;
    default:
      LOG(FATAL) << "Requested mapping type not implemented.";
  };
}

// The masked mapper is handling general dynamics and humans (with occupancy)
ProjectiveLayerType findMaskedLayerType(MappingType mapping_type) {
  switch (mapping_type) {
    case MappingType::kStaticTsdf:
      return ProjectiveLayerType::kNone;
    case MappingType::kStaticOccupancy:
      return ProjectiveLayerType::kNone;
    case MappingType::kDynamic:
      return ProjectiveLayerType::kOccupancy;
    case MappingType::kHumanWithStaticTsdf:
      return ProjectiveLayerType::kOccupancy;
    case MappingType::kHumanWithStaticOccupancy:
      return ProjectiveLayerType::kOccupancy;
    default:
      LOG(FATAL) << "Requested mapping type not implemented.";
  };
}

MultiMapper::MultiMapper(float voxel_size_m, MappingType mapping_type,
                         EsdfMode esdf_mode, MemoryType memory_type,
                         std::shared_ptr<CudaStream> cuda_stream)
    : mapping_type_(mapping_type),
      esdf_mode_(esdf_mode),
      cuda_stream_(cuda_stream) {
  // Initialize the multi mapper. Composed of:
  // - masked occupancy mapper for dynamic objects or humans
  // - unmasked mapper for static objects (with either an occupancy or tsdf
  //   layer)
  const ProjectiveLayerType unmasked_layer_type =
      findUnmaskedLayerType(mapping_type);
  const ProjectiveLayerType masked_layer_type =
      findMaskedLayerType(mapping_type);

  unmasked_mapper_ = std::make_shared<Mapper>(voxel_size_m, memory_type,
                                              unmasked_layer_type, cuda_stream);
  masked_mapper_ = std::make_shared<Mapper>(voxel_size_m, memory_type,
                                            masked_layer_type, cuda_stream);

  if (mapping_type_ == MappingType::kDynamic) {
    // For dynamic mapping we integrate the full depth into the unmasked mapper
    // (see NOTE in the integrateDepth function). Therefore, we need to ignore
    // the esdf sites that fall into freespace because they are actually dynamic
    // and handled by the masked mapper.
    unmasked_mapper_->ignore_esdf_sites_in_freespace(true);
  }

  // Set to an invalid depth to ignore dynamic/human pixels in the unmasked
  // mapper during integration.
  image_masker_.depth_unmasked_image_invalid_pixel(-1.f);

  // Set to a distance bigger than the max. integration distance to not
  // include non dynamic/human pixels on the dynamic mapper, but clear along the
  // projection.
  // TODO(remosteiner): Think of a better way to do this.
  // Currently this leads to blocks being allocated even behind solid
  // obstacles.
  image_masker_.depth_masked_image_invalid_pixel(
      std::numeric_limits<float>::max());
}

void MultiMapper::setMapperParams(
    const MapperParams& unmasked_mapper_params,
    const std::optional<MapperParams>& masked_mapper_params) {
  unmasked_mapper_->setMapperParams(unmasked_mapper_params);
  if (masked_mapper_params) {
    masked_mapper_->setMapperParams(masked_mapper_params.value());
  }
}

void MultiMapper::integrateDepth(const DepthImage& depth_frame,
                                 const Transform& T_L_CD,
                                 const Camera& depth_camera,
                                 const std::optional<Time>& update_time_ms) {
  CHECK(!isHumanMapping(mapping_type_))
      << "Only use this function for static or dynamic mapping. For human "
         "mapping please pass a mask to integrateDepth.";

  if (mapping_type_ == MappingType::kDynamic) {
    CHECK(update_time_ms);
    unmasked_mapper_->updateFreespace(update_time_ms.value());
    dynamic_detector_.computeDynamics(
        depth_frame, unmasked_mapper_->freespace_layer(), depth_camera, T_L_CD);

    // Remove small components (assumed to be noise) from the mask
    const MonoImage& dynamic_mask = dynamic_detector_.getDynamicMaskImage();
    image::removeSmallConnectedComponents(
        dynamic_mask, params_.connected_mask_component_size_threshold,
        &cleaned_dynamic_mask_, *cuda_stream_);

    // Note: the mask is a direct overlay of the depth image (therefore we use
    // identity for T_CM_CD)
    image_masker_.splitImageOnGPU(depth_frame, cleaned_dynamic_mask_,
                                  Transform::Identity(), depth_camera,
                                  depth_camera, &depth_frame_unmasked_,
                                  &depth_frame_masked_, &masked_depth_overlay_);

    // NOTE(remos): We need to integrate the full depth image into the unmasked
    // mapper. Otherwise freespace can not be reset as depth measurements
    // falling into the freespace will always be masked dynamic by the
    // DynamicsDetection module.
    unmasked_mapper_->integrateDepth(depth_frame, T_L_CD, depth_camera);
    masked_mapper_->integrateDepth(depth_frame_masked_, T_L_CD, depth_camera);
  } else {
    // For static mapping only integrate to the unmasked mapper
    unmasked_mapper_->integrateDepth(depth_frame, T_L_CD, depth_camera);
  }
}

void MultiMapper::integrateDepth(const DepthImage& depth_frame,
                                 const MonoImage& mask, const Transform& T_L_CD,
                                 const Transform& T_CM_CD,
                                 const Camera& depth_camera,
                                 const Camera& mask_camera) {
  CHECK(isHumanMapping(mapping_type_))
      << "Passing a mask to integrateDepth is only valid for human mapping.";

  // Split masked and non masked depth frame
  image_masker_.splitImageOnGPU(depth_frame, mask, T_CM_CD, depth_camera,
                                mask_camera, &depth_frame_unmasked_,
                                &depth_frame_masked_, &masked_depth_overlay_);

  // Integrate the frames to the respective layer cake
  unmasked_mapper_->integrateDepth(depth_frame_unmasked_, T_L_CD, depth_camera);
  masked_mapper_->integrateDepth(depth_frame_masked_, T_L_CD, depth_camera);
}

void MultiMapper::integrateColor(const ColorImage& color_frame,
                                 const Transform& T_L_C, const Camera& camera) {
  CHECK(!isHumanMapping(mapping_type_))
      << "Only use this function for static or dynamic mapping. For human "
         "mapping please pass a mask to integrateColor.";
  // TODO(remos): For kDynamic we should split the image and only integrate
  // unmasked pixels. As the dynamic mask is not a direct overlay of the color
  // image, this requires implementing a new splitImageOnGPU for color
  // images.
  unmasked_mapper_->integrateColor(color_frame, T_L_C, camera);
}

void MultiMapper::integrateColor(const ColorImage& color_frame,
                                 const MonoImage& mask, const Transform& T_L_C,
                                 const Camera& camera) {
  CHECK(isHumanMapping(mapping_type_))
      << "Passing a mask to integrateColor is only valid for human mapping.";
  if (mapping_type_ == MappingType::kHumanWithStaticOccupancy) {
    // We do nothing because color integration is only implemented for static
    // tsdf.
    return;
  }

  // Split masked and non masked color frame
  image_masker_.splitImageOnGPU(color_frame, mask, &color_frame_unmasked_,
                                &color_frame_masked_, &masked_color_overlay_);

  // Integrate the frames to the respective layer cake
  masked_mapper_->integrateColor(color_frame_masked_, T_L_C, camera);
  unmasked_mapper_->integrateColor(color_frame_unmasked_, T_L_C, camera);
}

void MultiMapper::updateEsdf() {
  updateEsdfOfMapper(unmasked_mapper_);
  if (masked_mapper_->projective_layer_type() != ProjectiveLayerType::kNone) {
    // Only update the masked mapper in case we run dynamics or human detection
    updateEsdfOfMapper(masked_mapper_);
  }
}

std::vector<Index3D> MultiMapper::updateMesh() {
  // At the moment we never have a mesh for the masked mapper as it alway uses a
  // occupancy layer.
  return unmasked_mapper_->updateMesh();
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
const ColorImage& MultiMapper::getLastDynamicFrameMaskOverlay() {
  return dynamic_detector_.getDynamicOverlayImage();
}
const Pointcloud& MultiMapper::getLastDynamicPointcloud() {
  return dynamic_detector_.getDynamicPointcloudDevice();
}

void MultiMapper::updateEsdfOfMapper(const std::shared_ptr<Mapper>& mapper) {
  switch (esdf_mode_) {
    case EsdfMode::kUnset:
      LOG(WARNING) << "ESDF mode not set. Doing nothing.";
      break;
    case EsdfMode::k3D:
      mapper->updateEsdf();
      break;
    case EsdfMode::k2D:
      mapper->updateEsdfSlice(params_.esdf_2d_min_height,
                              params_.esdf_2d_max_height,
                              params_.esdf_slice_height);
      break;
  }
}

parameters::ParameterTreeNode MultiMapper::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name = (name_remap.empty()) ? "multi_mapper" : name_remap;
  return ParameterTreeNode(
      name,
      {ParameterTreeNode("esdf_2d_min_height", params_.esdf_2d_min_height),
       ParameterTreeNode("esdf_2d_max_height", params_.esdf_2d_max_height),
       ParameterTreeNode("esdf_slice_height", params_.esdf_slice_height),
       ParameterTreeNode("connected_mask_component_size_threshold",
                         params_.connected_mask_component_size_threshold),
       unmasked_mapper_->getParameterTree("unmasked_mapper"),
       masked_mapper_->getParameterTree("masked_mapper"),
       image_masker_.getParameterTree()});
}

std::string MultiMapper::getParametersAsString() const {
  return parameterTreeToString(getParameterTree());
}

}  // namespace nvblox
