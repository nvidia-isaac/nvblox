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

#include "nvblox/experimental/ground_plane/ground_plane_estimator.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/mapper/multi_mapper_params.h"
#include "nvblox/semantics/mask_from_detections.h"
#include "nvblox/sensors/mask_preprocessor.h"

namespace nvblox {

enum class MappingType {
  kStaticTsdf,           /// only static tsdf
  kStaticOccupancy,      /// only static occupancy
  kDynamic,              /// static tsdf (incl. freespace) and dynamic occupancy
  kHumanWithStaticTsdf,  /// static tsdf and human occupancy
  kHumanWithStaticOccupancy  /// static occupancy and human occupancy
};

/// Whether the foreground mapper is used for human mapping
inline bool isHumanMapping(MappingType mapping_type) {
  if (mapping_type == MappingType::kHumanWithStaticTsdf ||
      mapping_type == MappingType::kHumanWithStaticOccupancy) {
    return true;
  }
  return false;
}

/// Whether the foreground mapper is used for dynamic mapping
inline bool isDynamicMapping(MappingType mapping_type) {
  if (mapping_type == MappingType::kDynamic) {
    return true;
  }
  return false;
}

/// Whether both the background and foreground mapper are active,
/// i.e. the foreground mapper is used for dynamic/human mapping
inline bool isUsingHumanOrDynamicMapper(MappingType mapping_type) {
  if (isHumanMapping(mapping_type) || isDynamicMapping(mapping_type)) {
    return true;
  }
  return false;
}

/// Whether the background mapper is doing occupancy
inline bool isStaticOccupancy(MappingType mapping_type) {
  if (mapping_type == MappingType::kStaticOccupancy ||
      mapping_type == MappingType::kHumanWithStaticOccupancy) {
    return true;
  }
  return false;
}

template <>
inline std::string toString(const MappingType& mapping_type) {
  switch (mapping_type) {
    case MappingType::kStaticTsdf:
      return "kStaticTsdf";
      break;
    case MappingType::kStaticOccupancy:
      return "kStaticOccupancy";
      break;
    case MappingType::kDynamic:
      return "kDynamic";
      break;
    case MappingType::kHumanWithStaticTsdf:
      return "kHumanWithStaticTsdf";
      break;
    case MappingType::kHumanWithStaticOccupancy:
      return "kHumanWithStaticOccupancy";
      break;
    default:
      CHECK(false) << "Requested mapping type is not implemented.";

      return "";  // should never be reached
  }
}

/// The MultiMapper class is composed of two standard Mappers.
/// Depth and color are integrated into one of these Mappers according to a
/// mask.
/// Setup:
/// - foreground mapper: Handling general dynamics or humans in an occupancy
///                      layer.
/// - background mapper: Handling static objects with a tsdf or a occupancy
/// layer.
///                      Also updating a freespace layer if mapping type is
///                      kDynamic.
///
/// NOTE(remos): For dynamic mapping the full depth image is integrated into the
/// background mapper (no masking). Otherwise freespace can not be reset as
/// depth measurements falling into the freespace will always be masked dynamic
/// by the DynamicsDetection module. As a consequence, we need to ignore the
/// esdf sites in the background mapper that fall into freespace because they
/// are actually dynamic and handled by the foreground mapper.
class MultiMapper {
 public:
  /// @param voxel_size_m The voxel size in meters for the contained layers.
  /// @param projective_layer_type The layer type to which the projective
  ///        data is integrated (either tsdf or occupancy).
  /// @param memory_type In which type of memory the layers should be stored.

  /// Constructor
  /// @param voxel_size_m The voxel size in meters for the contained layers.
  /// @param mapping_type Static, Dynamic etc. See MappingType.
  /// @param esdf_mode 2D or 3D. See EsdfMode.
  /// @param memory_type In which type of memory the layers should be stored.
  /// @param cuda_stream Optional cuda stream to perform all work on.
  MultiMapper(float voxel_size_m, MappingType mapping_type, EsdfMode esdf_mode,
              MemoryType memory_type = MemoryType::kDevice,
              std::shared_ptr<CudaStream> cuda_stream =
                  std::make_shared<CudaStreamOwning>());
  ~MultiMapper() = default;

  /// @brief Setting the multi mapper param struct
  /// @param multi_mapper_params the param struct
  void setMultiMapperParams(const MultiMapperParams& multi_mapper_params);

  /// @brief Setting the mapper param struct to the two mappers
  /// @param background_mapper_params param struct for background mapper
  /// @param foreground_mapper_params param struct for foreground mapper
  /// (optional)
  void setMapperParams(const MapperParams& background_mapper_params,
                       const std::optional<MapperParams>&
                           foreground_mapper_params = std::nullopt);

  /// @brief Integrates a depth frame into the reconstruction (for mapping type
  /// kStaticTsdf/kStaticOccupancy/kDynamic).
  /// @param depth_frame Depth frame to integrate.
  /// @param T_L_CD Pose of the depth sensor, specified as a transform from
  ///              sensor frame to layer frame transform.
  /// @param depth_sensor Intrinsics model of the depth sensor.
  /// @param update_time_ms Current update time in millisecond. Only needed for
  /// dynamic mapping.
  template <typename SensorType>
  void integrateDepth(const DepthImage& depth_frame, const Transform& T_L_CD,
                      const SensorType& depth_sensor,
                      const std::optional<Time>& update_time_ms = std::nullopt);

  /// @brief Integrates a LiDAR pointcloud into the reconstruction.
  /// The pointcloud will be converted to a depth image and then integrated.
  /// Runs motion compensation if use_lidar_motion_compensation is true and
  //  per-point timestamps, T_L_S_scanEnd and scan_duration_ms are provided.
  ///
  /// @param pointcloud Pointcloud to integrate. Must contain points and
  ///                   optionally per-point timestamps for motion compensation.
  /// @param T_L_S_scanStart Pose of the LiDAR sensor at scan start, specified
  ///                        as a transform from sensor frame to layer frame.
  /// @param lidar_sensor Intrinsics model of the LiDAR sensor.
  /// @param use_lidar_motion_compensation Whether to use motion compensation.
  /// @param T_L_S_scanEnd Pose of the LiDAR sensor at scan end. Only needed if
  /// motion compensation is enabled.
  /// @param scan_duration_ms Duration of the scan in milliseconds. Only needed
  /// if motion compensation is enabled.
  /// @param update_time_ms Current update time in millisecond. Only needed for
  /// dynamic mapping.
  template <typename SensorType>
  void integrateDepth(
      const Pointcloud& pointcloud, const Transform& T_L_S_scanStart,
      const SensorType& lidar_sensor, bool use_lidar_motion_compensation,
      const std::optional<Transform>& T_L_S_scanEnd = std::nullopt,
      const std::optional<Time>& scan_duration_ms = std::nullopt,
      const std::optional<Time>& update_time_ms = std::nullopt);

  /// @brief Integrates a depth camera with detection boxes into the
  /// reconstruction (for mapping type kStaticTsdf/kStaticOccupancy/kDynamic).
  ///
  /// @param depth_frame Depth frame to integrate.
  /// @param detection_boxes Bounding boxes.
  /// @param T_L_CD Pose of the depth camera, specified as a transform from
  ///              camera frame to layer frame transform.
  /// @param camera Intrinsics model of the depth camera.
  void integrateDepth(const DepthImage& depth_frame,
                      const std::vector<ImageBoundingBox>& detection_boxes,
                      const Transform& T_L_CD, const Camera& depth_camera);

  /// @brief Integrates a depth frame into the reconstruction
  /// using the transformation between depth and mask frame and their
  /// intrinsics (for mapping type
  /// kHumanWithStaticTsdf/kHumanWithStaticOccupancy).
  ///@param depth_frame Depth frame to integrate. Depth in the image is
  ///                   specified as a float representing meters.
  ///@param mask Mask. Interpreted as 0=background, >0=foreground. Dimensions
  /// must be divisible by kDownScaleFactor.
  ///@param T_L_CD Pose of the depth sensor, specified as a transform from
  ///              sensor frame to layer frame transform.
  ///@param T_CM_CD Transform from depth sensor to mask sensor frame.
  ///@param depth_sensor Intrinsics model of the depth sensor.
  ///@param mask_sensor Intrinsics model of the mask sensor.
  template <typename SensorType>
  void integrateDepth(const DepthImage& depth_frame, const MonoImage& mask,
                      const Transform& T_L_CD, const Transform& T_CM_CD,
                      const SensorType& depth_sensor,
                      const SensorType& mask_sensor);

  /// @brief Integrates a depth frame into the reconstruction
  /// using the transformation between depth and bounding box frame and their
  /// intrinsics (for mapping type
  /// kHumanWithStaticTsdf/kHumanWithStaticOccupancy).
  ///@param depth_frame Depth frame to integrate. Depth in the image is
  ///                   specified as a float representing meters.
  ///@param detection_boxes Boxes indicating location of objects to be masked.
  /// Given in pixel coordinates.
  ///@param T_L_CD Pose of the depth camera, specified as a transform from
  ///              camera frame to layer frame transform.
  ///@param T_CM_CD Transform from depth camera to the bounding box camera
  /// frame.
  ///@param depth_camera Intrinsics model of the depth camera.
  ///@param detection_boxes_camera Intrinsics model of the camera used for
  /// acquiring the detection boxes
  void integrateDepth(const DepthImage& depth_frame,
                      const std::vector<ImageBoundingBox>& detection_boxes,
                      const Transform& T_L_CD, const Transform& T_CM_CD,
                      const Camera& depth_camera,
                      const Camera& detection_boxes_camera);

  /// @brief Integrates a color frame into the reconstruction (for mapping
  /// type kStaticTsdf/kStaticOccupancy/kDynamic).
  /// @param color_frame Color image to integrate.
  /// @param T_L_C Pose of the sensor, specified as a transform from sensor
  /// frame to the layer frame.
  /// @param sensor Intrinsics model of the sensor.
  template <typename SensorType>
  void integrateColor(const ColorImage& color_frame, const Transform& T_L_C,
                      const SensorType& sensor);

  /// @brief Integrates a color frame into the reconstruction (for mapping
  /// type kHumanWithStaticTsdf/kHumanWithStaticOccupancy).
  ///@param color_frame Color image to integrate.
  ///@param mask Mask. Interpreted as 0=foreground, >0=background. Dimensions
  /// must be divisible by kDownScaleFactor.
  ///@param T_L_C Pose of the sensor, specified as a transform from sensor
  ///             frame to the layer frame.
  ///@param sensor Intrinsics model of the sensor.
  template <typename SensorType>
  void integrateColor(const ColorImage& color_frame, const MonoImage& mask,
                      const Transform& T_L_C, const SensorType& sensor);

  /// Integrated a color frame with a list of detection boxes.
  /// NOTE: Right now the detection boxes are ignored and the whole color frame
  /// is integrated into the static map.
  ///@param color_frame Color image to integrate.
  ///@param detection_boxes Boxes indicating location of objects to be masked.
  /// Given in pixel coordinates.
  ///@param T_L_C Pose of the camera, specified as a transform from camera
  ///             frame to the layer frame.
  ///@param camera Intrinsics model of the camera.
  void integrateColor(const ColorImage& color_frame,
                      const std::vector<ImageBoundingBox>& detection_boxes,
                      const Transform& T_L_C, const Camera& camera);

  /// @brief Updating the esdf layers of the mappers depending on the mapping
  /// type.
  void updateEsdf();

  /// @brief Updating the mesh layers of the mappers depending on the mapping
  /// type.
  void updateColorMesh();

  /// @brief Extract a flat Mesh via single-pass mesh integration.
  /// @tparam AppearanceVoxelType ColorVoxel or FeatureVoxel.
  template <typename AppearanceVoxelType>
  void updateFlatMesh();

  /// @brief Extract a frustum-culled flat Mesh via single-pass mesh
  /// integration. Only blocks visible in the camera frustum are meshed.
  /// @tparam AppearanceVoxelType ColorVoxel or FeatureVoxel.
  /// @param camera Camera intrinsics defining the field of view.
  /// @param T_L_C Camera-to-layer-frame transform (camera pose in map frame).
  /// @param max_depth Maximum depth (meters) for frustum extent.
  template <typename AppearanceVoxelType>
  void updateFlatMesh(const Camera& camera, const Transform& T_L_C,
                      float max_depth);

  /// Convenience: extract a flat ColorMesh from all TSDF blocks.
  void updateFlatColorMesh() { updateFlatMesh<ColorVoxel>(); }

  /// Convenience: extract a frustum-culled flat ColorMesh.
  void updateFlatColorMesh(const Camera& camera, const Transform& T_L_C,
                           float max_depth) {
    updateFlatMesh<ColorVoxel>(camera, T_L_C, max_depth);
  }

  /// Convenience: extract a flat FeatureMesh from all TSDF blocks.
  void updateFlatFeatureMesh() { updateFlatMesh<FeatureVoxel>(); }

  /// Access to one of the mappers
  const Mapper& background_mapper() const { return *background_mapper_.get(); }
  /// Access to one of the mappers
  const Mapper& foreground_mapper() const { return *foreground_mapper_.get(); }
  /// Access to one of the mappers
  std::shared_ptr<Mapper> background_mapper() { return background_mapper_; }
  /// Access to one of the mappers
  std::shared_ptr<Mapper> foreground_mapper() { return foreground_mapper_; }

  /// Getter
  ///@return GroundPlaneEstimator& ground_plane_estimator
  GroundPlaneEstimator& ground_plane_estimator() {
    return ground_plane_estimator_;
  }

  /// These functions return a reference to the images generated during
  /// the preceeding calls to integrateColor() and integrateDepth().
  const DepthImage& getLastDepthFrameBackground();
  const DepthImage& getLastDepthFrameForeground();
  const ColorImage& getLastColorFrameBackground();
  const ColorImage& getLastColorFrameForeground();
  const ColorImage& getLastDepthFrameMaskOverlay();
  const ColorImage& getLastColorFrameMaskOverlay();
  const ColorImage& getLastDynamicFrameMaskOverlay();
  const Pointcloud& getLastDynamicPointcloud();
  const DepthImage& getLastDepthFrameFromPointcloud();

  const MultiMapperParams& getMultiMapperParams() const { return params_; }

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

  /// Return the parameter tree represented as a string
  /// @return the parameter tree string
  virtual std::string getParametersAsString() const;

  /// Save the ground plane as a YAML file.
  /// @param filepath Path to the output YAML file.
  /// @return if write to file was successful
  bool saveGroundPlaneAsYaml(const std::string& filepath);

 protected:
  // Performs the esdf update on the passed mapper
  void updateEsdfOfMapper(const std::shared_ptr<Mapper> mapper,
                          std::optional<Plane> ground_plane = std::nullopt);

  // Mapping type needed on construction
  const MappingType mapping_type_;
  const EsdfMode esdf_mode_;

  // Parameter struct for multi mapper
  MultiMapperParams params_;

  // Helper to detect dynamics from a freespace layer
  DynamicsDetection dynamic_detector_;
  MonoImage cleaned_dynamic_mask_{MemoryType::kDevice};

  // Declared to use for cleaning up of semantic mask
  MonoImage cleaned_semantic_mask_{MemoryType::kDevice};

  // Split depth images based on a mask.
  // Note that we internally pre-allocate space for the split images on the
  // first call.
  ImageMasker image_masker_;
  DepthImage depth_frame_background_{MemoryType::kDevice};
  DepthImage depth_frame_foreground_{MemoryType::kDevice};

  // Mask overlays used as debug outputs
  ColorImage foreground_depth_overlay_{MemoryType::kDevice};

  // The two mappers to which the frames are integrated.
  std::shared_ptr<Mapper> foreground_mapper_;
  std::shared_ptr<Mapper> background_mapper_;

  // Cleanup segmentation masks
  image::MaskPreprocessor mask_preprocessor_;

  // Helper class estimating the ground plane
  GroundPlaneEstimator ground_plane_estimator_;

  // Pre-allocated depth frame for pointcloud integration.
  DepthImage depth_frame_from_pointcloud_{MemoryType::kDevice};

  // The CUDA stream on which to process all work
  std::shared_ptr<CudaStream> cuda_stream_;
};

}  // namespace nvblox

#include "nvblox/mapper/internal/impl/multi_mapper_impl.h"
