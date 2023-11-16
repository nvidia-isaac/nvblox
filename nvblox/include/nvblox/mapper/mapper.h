/*
Copyright 2022-2023 NVIDIA CORPORATION

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

#include <optional>
#include <unordered_set>

#include "nvblox/core/hash.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/dynamics/dynamics_detection.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/freespace_integrator.h"
#include "nvblox/integrators/occupancy_decay_integrator.h"
#include "nvblox/integrators/projective_color_integrator.h"
#include "nvblox/integrators/projective_occupancy_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/tsdf_decay_integrator.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/layer_cake.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mapper/mapper_params.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/mesh/mesh_streamer.h"
#include "nvblox/semantics/image_masker.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/depth_preprocessing.h"
#include "nvblox/sensors/lidar.h"

namespace nvblox {

// Which type of mapping to do.
enum class ProjectiveLayerType { kTsdf, kOccupancy, kNone };
inline std::string toString(ProjectiveLayerType layer_type) {
  switch (layer_type) {
    case ProjectiveLayerType::kTsdf:
      return "kTsdf";
      break;
    case ProjectiveLayerType::kOccupancy:
      return "kOccupancy";
      break;
    case ProjectiveLayerType::kNone:
      return "kNone";
      break;
    default:
      LOG(FATAL) << "Not implemented";
      break;
  }
  return "";
}

/// The ESDF mode. Enum indicates if an Mapper is configured for 3D or 2D
/// Esdf production, or that this has not yet been determined (kUnset).
enum class EsdfMode { k3D, k2D, kUnset };
std::string toString(EsdfMode esdf_mode);

/// The mapper classes wraps layers and integrators together.
/// In the base class we only specify that a mapper should contain map layers
/// and leave it up to sub-classes to add functionality.
class MapperBase {
 public:
  static constexpr bool kDefaultIgnoreEsdfSitesInFreespace = false;
  static constexpr ProjectiveLayerType kDefaultProjectiveLayerType =
      ProjectiveLayerType::kTsdf;

  MapperBase() = default;
  virtual ~MapperBase() = default;

  /// Move
  MapperBase(MapperBase&& other) = default;
  MapperBase& operator=(MapperBase&& other) = default;

 protected:
  /// Map layers
  LayerCake layers_;
};

/// The Mapper class is what we consider the default mapping behaviour in
/// nvblox.
/// Contains:
/// - TsdfLayer, OccupancyLayer, ColorLayer, EsdfLayer, MeshLayer
/// - Integrators associated with these layer types.
///
/// Exposes functions for:
/// - Integrating depth/rgbd images, 3D LiDAR scans, and color images.
/// - Functions for generating Meshes, ESDF, and ESDF-slices.
class Mapper : public MapperBase {
 public:
  // Parameter defaults: See mapper_params.h

  Mapper() = delete;
  /// Constructor
  /// @param voxel_size_m The voxel size in meters for the contained layers.
  /// @param projective_layer_type The layer type to which the projective
  ///        data is integrated (either tsdf or occupancy).
  /// @param memory_type In which type of memory the layers should be stored.
  Mapper(float voxel_size_m, MemoryType memory_type = MemoryType::kDevice,
         ProjectiveLayerType projective_layer_type = ProjectiveLayerType::kTsdf,
         std::shared_ptr<CudaStream> cuda_stream =
             std::make_shared<CudaStreamOwning>());
  virtual ~Mapper() = default;

  /// Constructor which initializes from a saved map.
  /// @param map_filepath Path to the serialized map to be loaded.
  /// @param memory_type In which type of memory the layers should be stored.
  Mapper(const std::string& map_filepath,
         MemoryType memory_type = MemoryType::kDevice,
         std::shared_ptr<CudaStream> cuda_stream =
             std::make_shared<CudaStreamOwning>());

  /// Move
  Mapper(Mapper&& other) = default;
  Mapper& operator=(Mapper&& other) = default;

  void setMapperParams(const MapperParams& params);

  /// Integrates a depth frame into the tsdf reconstruction.
  ///@param depth_frame Depth frame to integrate. Depth in the image is
  ///                   specified as a float representing meters.
  ///@param T_L_C Pose of the camera, specified as a transform from
  ///             Camera-frame to Layer-frame transform.
  ///@param camera Intrinsics model of the camera.
  void integrateDepth(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Camera& camera);

  /// Integrates a color frame into the reconstruction.
  ///@param color_frame Color image to integrate.
  ///@param T_L_C Pose of the camera, specified as a transform from
  ///             Camera-frame to Layer-frame transform.
  ///@param camera Intrinsics model of the camera.
  void integrateColor(const ColorImage& color_frame, const Transform& T_L_C,
                      const Camera& camera);

  /// Integrates a 3D LiDAR scan into the reconstruction.
  ///@param depth_frame Depth image representing the LiDAR scan. To convert a
  ///                   lidar scan to a DepthImage see TODOOO.
  ///@param T_L_C Pose of the LiDAR, specified as a transform from LiDAR-frame
  ///             to Layer-frame transform.
  ///@param lidar Intrinsics model of the LiDAR.
  void integrateLidarDepth(const DepthImage& depth_frame,
                           const Transform& T_L_C, const Lidar& lidar);

  /// Decay the TSDF layer (reduce weights)
  void decayTsdf();

  /// Decay the TSDF layer (reduce weights)
  /// @param block_to_exclude   Blocks to exclude when decaying
  /// @param exclusion_center   Center of radial exclusion of blocks
  /// @param exclusion_radius_m Radius of the radial exclusion
  void decayTsdf(const std::vector<Index3D>& block_to_exclude,
                 const std::optional<Vector3f>& exclusion_center,
                 const std::optional<float>& exclusion_radius_m);

  /// Decay the full occupancy layer.
  void decayOccupancy();

  /// Decay the occupancy layer (approach 0.5 occupancy probability).
  /// @param block_to_exclude  Blocks to exclude when decaying
  /// @param camera_center     Center of radial exclusion of blocks
  /// @param exclusion_radius_m Radius of the radial exclusion
  void decayOccupancy(std::vector<Index3D> const& blocks_to_exclude,
                      const std::optional<Vector3f>& exclusion_center,
                      const std::optional<float>& exclusion_radius_m);

  /// Updates the freespace blocks which require an update
  /// @param The time of the update in miliseconds.
  /// @return The indices of the blocks that were updated in this call.
  std::vector<Index3D> updateFreespace(Time update_time_ms);

  /// Updates the mesh blocks which require an update
  /// @return The indices of the blocks that were updated in this call.
  std::vector<Index3D> updateMesh();

  /// Generate (or re-generate) a mesh for the entire map. Useful if loading
  /// a layer cake without a mesh layer, for example.
  void updateFullMesh();

  /// Updates the ESDF blocks which require an update.
  /// Note that currently we limit the Mapper class to calculating *either*
  /// the 2D or 3D ESDF, not both. Which is to be calculated is determined by
  /// the first call to updateEsdf().
  ///@return std::vector<Index3D> The indices of the blocks that were updated
  ///        in this call.
  std::vector<Index3D> updateEsdf();

  /// Generate an ESDF on *all* allocated blocks. Will replace whatever has
  /// been done before.
  void updateFullEsdf();

  /// Updates the ESDF blocks which require an update.
  /// Note that currently we limit the Mapper class to calculating *either*
  /// the 2D or 3D ESDF, not both. Which is to be calculated is determined by
  /// the first call to updateEsdf(). This function operates by collapsing a
  /// finite thickness slice of the 3D TSDF into a binary obstacle map, and
  /// then generating the 2D ESDF. The input parameters define the limits of
  /// the 3D slice that are considered. Note that the resultant 2D ESDF is
  /// stored in a single voxel thick layer in ESDF layer.
  /// @return The indices of the blocks that were updated in this call.
  ///@param slice_input_z_min The minimum height of the 3D TSDF slice used to
  ///                         generate the 2D binary obstacle map.
  ///@param slice_input_z_max The minimum height of the 3D TSDF slice used to
  ///                         generate the 2D binary obstacle map.
  ///@param slice_output_z The height at which the 2D ESDF is stored.
  ///@return std::vector<Index3D>  The indices of the blocks that were updated
  ///        in this call.
  std::vector<Index3D> updateEsdfSlice(float slice_input_z_min,
                                       float slice_input_z_max,
                                       float slice_output_z);

  /// Clears the reconstruction outside a radius around a center point,
  /// deallocating the memory.
  ///@param center The center of the keep-sphere.
  ///@param radius The radius of the keep-sphere.
  ///@return std::vector<Index3D> The block indices removed.
  std::vector<Index3D> clearOutsideRadius(const Vector3f& center, float radius);

  /// Allocates blocks touched by radius and gives their voxels some small
  /// positive weight.
  /// @param center The center of allocation-sphere
  /// @param radius The radius of allocation-sphere
  void markUnobservedTsdfFreeInsideRadius(const Vector3f& center, float radius);

  /// Return N bytes of MeshBlocks from the stream queue.
  /// @param num_bytes The number of bytes of mesh blocks to stream
  /// @param exclusion_center_m Optional center of radius-based exclusion.
  /// This parameter is required if radius-based exclusion is requested.
  /// @return The list of mesh block indices to stream
  std::vector<Index3D> getNBytesOfMeshBlocksFromStreamQueue(
      const size_t num_bytes,
      const std::optional<Vector3f>& exclusion_center_m = std::nullopt);

  /// Return N bytes of serialized MeshBlocks from the stream queue.
  /// @param num_bytes The number of bytes of mesh blocks to stream
  /// @param exclusion_center_m Optional center of radius-based exclusion.
  /// This parameter is required if radius-based exclusion is requested.
  /// @param cuda_stream Cuda stream.
  /// @param Serialized mesh containing highest priority mesh blocks
  const std::shared_ptr<const SerializedMesh>
  getNBytesOfSerializedMeshBlocksFromStreamQueue(
      const size_t num_bytes, const std::optional<Vector3f>& exclusion_center_m,
      CudaStream cuda_stream);

  /// Gets the preprocessed version of the last depth image passed to
  /// integrateDepth(). Note that we return a shared_ptr to a buffered depth
  /// image inside the mapper to avoid copying the image. Subsequent calls to
  /// integrateDepth will change the contents of this image.
  /// @return The preprocessed DepthImage.
  const std::shared_ptr<const DepthImage> getPreprocessedDepthImage() const {
    return preprocessed_depth_image_;
  }

  /// Getter
  ///@return const LayerCake& The collection of layers mapped.
  const LayerCake& layers() const { return layers_; }
  /// Getter
  ///@return const TsdfLayer& TSDF layer
  const TsdfLayer& tsdf_layer() const { return layers_.get<TsdfLayer>(); }
  /// Getter
  ///@return const OccupancyLayer& occupancy layer
  const OccupancyLayer& occupancy_layer() const {
    return layers_.get<OccupancyLayer>();
  }
  /// Getter
  ///@return const FreespaceLayer& freespace layer
  const FreespaceLayer& freespace_layer() const {
    return layers_.get<FreespaceLayer>();
  }
  /// Getter
  ///@return const ColorLayer& Color layer
  const ColorLayer& color_layer() const { return layers_.get<ColorLayer>(); }
  /// Getter
  ///@return const EsdfLayer& ESDF layer
  const EsdfLayer& esdf_layer() const { return layers_.get<EsdfLayer>(); }
  /// Getter
  ///@return const MeshLayer& Mesh layer
  const MeshLayer& mesh_layer() const { return layers_.get<MeshLayer>(); }

  /// Getter
  ///@return LayerCake& The collection of layers mapped.
  LayerCake& layers() { return layers_; }
  /// Getter
  ///@return TsdfLayer& TSDF layer
  TsdfLayer& tsdf_layer() { return *layers_.getPtr<TsdfLayer>(); }
  /// Getter
  ///@return OccupancyLayer& occupancy layer
  OccupancyLayer& occupancy_layer() {
    return *layers_.getPtr<OccupancyLayer>();
  }
  /// Getter
  ///@return FreespaceLayer& freespace layer
  FreespaceLayer& freespace_layer() {
    return *layers_.getPtr<FreespaceLayer>();
  }
  /// Getter
  ///@return ColorLayer& Color layer
  ColorLayer& color_layer() { return *layers_.getPtr<ColorLayer>(); }
  /// Getter
  ///@return EsdfLayer& ESDF layer
  EsdfLayer& esdf_layer() { return *layers_.getPtr<EsdfLayer>(); }
  /// Getter
  ///@return MeshLayer& Mesh layer
  MeshLayer& mesh_layer() { return *layers_.getPtr<MeshLayer>(); }

  /// Getter
  ///@return const ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        depth/rgbd frame integration.
  const ProjectiveTsdfIntegrator& tsdf_integrator() const {
    return tsdf_integrator_;
  }
  /// Getter
  ///@return const ProjectiveOccupancyIntegrator& occupancy integrator used
  /// for
  ///        depth/rgbd frame integration.
  const ProjectiveOccupancyIntegrator& occupancy_integrator() const {
    return occupancy_integrator_;
  }
  /// Getter
  ///@return const FreespaceIntegrator& freespace integrator used for
  ///        updating the freespace layer according to a tsdf layer.
  const FreespaceIntegrator& freespace_integrator() const {
    return freespace_integrator_;
  }
  /// Getter
  ///@return const ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        3D LiDAR scan integration.
  const ProjectiveTsdfIntegrator& lidar_tsdf_integrator() const {
    return lidar_tsdf_integrator_;
  }
  /// Getter
  ///@return const ProjectiveOccupancyIntegrator& occupancy integrator used
  /// for
  ///        3D LiDAR scan integration.
  const ProjectiveOccupancyIntegrator& lidar_occupancy_integrator() const {
    return lidar_occupancy_integrator_;
  }
  /// Getter
  ///@return const OccupancyDecayIntegrator& occupancy integrator used for
  ///        decaying an occupancy layer towards 0.5 occupancy probability.
  const OccupancyDecayIntegrator& occupancy_decay_integrator() const {
    return occupancy_decay_integrator_;
  }
  /// Getter
  ///@return const TsdfDecayIntegrator& tsdf integrator used for
  ///        decaying an tsdf layer
  const TsdfDecayIntegrator& tsdf_decay_integrator() const {
    return tsdf_decay_integrator_;
  }
  /// Getter
  ///@return const ProjectiveColorIntegrator& Color integrator.
  const ProjectiveColorIntegrator& color_integrator() const {
    return color_integrator_;
  }
  /// Getter
  ///@return const MeshStreamerOldestBlocks& Mesh streamer.
  const MeshStreamerOldestBlocks& mesh_streamer() const {
    return mesh_streamer_;
  }
  /// Getter
  ///@return const MeshIntegrator& Mesh integrator
  const MeshIntegrator& mesh_integrator() const { return mesh_integrator_; }
  /// Getter
  ///@return const EsdfIntegrator& ESDF integrator
  const EsdfIntegrator& esdf_integrator() const { return esdf_integrator_; }

  /// Getter
  ///@return ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        depth/rgbd frame integration.
  ProjectiveTsdfIntegrator& tsdf_integrator() { return tsdf_integrator_; }
  /// Getter
  ///@return ProjectiveOccupancyIntegrator& occupancy integrator used for
  ///        depth/rgbd frame integration.
  ProjectiveOccupancyIntegrator& occupancy_integrator() {
    return occupancy_integrator_;
  }
  /// Getter
  ///@return FreespaceIntegrator& freespace integrator used for
  ///        updating the freespace layer according to a tsdf layer.
  FreespaceIntegrator& freespace_integrator() { return freespace_integrator_; }
  /// Getter
  ///@return ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        3D LiDAR scan integration.
  ProjectiveTsdfIntegrator& lidar_tsdf_integrator() {
    return lidar_tsdf_integrator_;
  }
  /// Getter
  ///@return ProjectiveOccupancyIntegrator& occupancy integrator used for
  ///        3D LiDAR scan integration.
  ProjectiveOccupancyIntegrator& lidar_occupancy_integrator() {
    return lidar_occupancy_integrator_;
  }
  /// Getter
  ///@return OccupancyDecayIntegrator& occupancy decay integrator used for
  ///        decaying an occupancy layer towards 0.5 occupancy probability.
  OccupancyDecayIntegrator& occupancy_decay_integrator() {
    return occupancy_decay_integrator_;
  }
  /// Getter
  ///@return TsdfDecayIntegrator& TSDF decay integrator used for decaying a TSDF
  ///        layer (through reduction of voxel weights).
  TsdfDecayIntegrator& tsdf_decay_integrator() {
    return tsdf_decay_integrator_;
  }
  /// Getter
  ///@return ProjectiveColorIntegrator& Color integrator.
  ProjectiveColorIntegrator& color_integrator() { return color_integrator_; }
  /// Getter
  ///@return MeshIntegrator& Mesh integrator
  MeshIntegrator& mesh_integrator() { return mesh_integrator_; }
  /// Getter
  ///@return EsdfIntegrator& ESDF integrator
  EsdfIntegrator& esdf_integrator() { return esdf_integrator_; }
  /// Getter
  ///@return MeshStreamerOldestBlocks& Mesh streamer.
  MeshStreamerOldestBlocks& mesh_streamer() { return mesh_streamer_; }
  /// Getter
  /// @return The voxel size in meters
  float voxel_size_m() const { return voxel_size_m_; };
  /// Getter
  /// @return The type of projectivelayer we're mapping
  ProjectiveLayerType projective_layer_type() const {
    return projective_layer_type_;
  };

  /// Getter: see description of ignore_esdf_sites_in_freespace_
  /// @return value Boolean whether to ignore esdf sites in freespace
  bool ignore_esdf_sites_in_freespace() const {
    return ignore_esdf_sites_in_freespace_;
  };
  /// Setter: see description of ignore_esdf_sites_in_freespace_
  /// @param value Boolean whether to ignore esdf sites in freespace
  void ignore_esdf_sites_in_freespace(bool value) {
    ignore_esdf_sites_in_freespace_ = value;
  };

  /// Getter
  /// @return Whether we should build a MeshBlock stream queue such that N
  /// bytes of mesh can be requested through
  /// getNBytesOfMeshBlocksFromStreamQueue().
  bool maintain_mesh_block_stream_queue() const {
    return maintain_mesh_block_stream_queue_;
  }
  /// Setter
  /// @param Whether we should build a MeshBlock stream queue.
  void maintain_mesh_block_stream_queue(
      const bool maintain_mesh_block_stream_queue) {
    maintain_mesh_block_stream_queue_ = maintain_mesh_block_stream_queue;
  }

  /// Getter
  /// @return Whether we should perform preprocessing on input DepthImages
  bool do_depth_preprocessing() const { return do_depth_preprocessing_; }
  /// Setter
  /// @param do_depth_preprocessing Whether to perform depth preprocessing.
  void do_depth_preprocessing(const bool do_depth_preprocessing) {
    do_depth_preprocessing_ = do_depth_preprocessing;
  }
  /// Getter How many times to run a 3x3 dilation kernel on the invalid mask
  /// of the depth image.
  /// @return The number of application of the 3x3 dilation kernel.
  int depth_preprocessing_num_dilations() const {
    return depth_preprocessing_num_dilations_;
  }
  /// Setter. See depth_preprocessing_num_dilations()
  /// @param depth_preprocessing_num_dilations How many times to run the
  /// kernel.
  void depth_preprocessing_num_dilations(
      const int depth_preprocessing_num_dilations) {
    CHECK_GE(depth_preprocessing_num_dilations, 0);
    depth_preprocessing_num_dilations_ = depth_preprocessing_num_dilations;
  }

  /// Getter for TSDF decay factor.
  /// @return decay factor
  int tsdf_decay_factor() const {
    return tsdf_decay_integrator_.decay_factor();
  }
  /// Setter for TSDF decay factor
  /// @param tsdf_factor decay factor.
  /// @pre 0.0 < param < 1.0
  void tsdf_decay_factor(const float tsdf_decay_factor) {
    tsdf_decay_integrator_.decay_factor(tsdf_decay_factor);
  }

  /// Saving and loading functions.
  /// Saving a map will serialize the TSDF and ESDF layers to a file.
  ///@param filename
  ///@return true
  ///@return false
  bool saveLayerCake(const std::string& filename) const;
  bool saveLayerCake(const char* filename) const;
  /// Loading the map will load a the TSDF and ESDF layers from a file.
  /// Will clear anything in the map already.
  bool loadMap(const std::string& filename);
  bool loadMap(const char* filename);

  /// Write mesh as a PLY
  /// @param filename Path to output PLY file.
  /// @return bool Flag indicating if write was successful.
  bool saveMeshAsPly(const std::string& filename) const;

  /// Writes the Esdf as a PLY
  /// @param filename Path to the output PLY file.
  /// @return bool Flag indicating if the write was successful.
  bool saveEsdfAsPly(const std::string& filename) const;

  /// Writes the Tsdf as a PLY
  /// @param filename Path to the output PLY file.
  /// @return bool Flag indicating if the write was successful.
  bool saveTsdfAsPly(const std::string& filename) const;

  /// Writes the freespace as a PLY
  /// @param filename Path to the output PLY file.
  /// @return bool Flag indicating if the write was successful.
  bool saveFreespaceAsPly(const std::string& filename) const;

  /// Writes the occupancy as a PLY
  /// @param filename Path to the output PLY file.
  /// @return bool Flag indicating if the write was successful.
  bool saveOccupancyAsPly(const std::string& filename) const;

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

  /// Return the parameter tree represented as a string
  /// @return the parameter tree string
  virtual std::string getParametersAsString() const;

 protected:
  /// Perform preprocessing on a depth image
  const DepthImage& preprocessDepthImageAsync(const DepthImage& depth_image);

  /// The CUDA stream that mapper work is processed on
  std::shared_ptr<CudaStream> cuda_stream_;

  /// The size of the voxels to be used in the TSDF, ESDF, Color layers.
  float voxel_size_m_;
  /// The storage location for the TSDF, ESDF, Color, and Mesh Layers.
  MemoryType memory_type_;
  /// The layer type to which the projective data is integrated (either tsdf
  /// or occupancy).
  ProjectiveLayerType projective_layer_type_ = kDefaultProjectiveLayerType;

  /// This class can be used to generate *either* (not both) the 2D or 3D
  /// ESDF. The mode used is determined by the first call to either
  /// updateEsdf() or updateEsdfSlice(). This member tracks which mode we're
  /// in.
  EsdfMode esdf_mode_ = EsdfMode::kUnset;

  /// Integrators
  ProjectiveTsdfIntegrator tsdf_integrator_;
  ProjectiveTsdfIntegrator lidar_tsdf_integrator_;
  FreespaceIntegrator freespace_integrator_;
  ProjectiveOccupancyIntegrator occupancy_integrator_;
  ProjectiveOccupancyIntegrator lidar_occupancy_integrator_;
  OccupancyDecayIntegrator occupancy_decay_integrator_;
  TsdfDecayIntegrator tsdf_decay_integrator_;
  ProjectiveColorIntegrator color_integrator_;
  MeshIntegrator mesh_integrator_;
  EsdfIntegrator esdf_integrator_;

  /// Preprocessing depth maps prior to integration.
  /// Currently, the only preprocessing step is to dilate the invalid regions
  /// of the input depth image. We have found this useful to reduce the
  /// depth-bleeding effects on the intel realsense.
  bool do_depth_preprocessing_ = mapper::kDefaultDoDepthPreprocessing;
  int depth_preprocessing_num_dilations_ =
      mapper::kDefaultDepthPreprocessingNumDilations;
  DepthPreprocessor depth_preprocessor_;
  std::shared_ptr<DepthImage> preprocessed_depth_image_ =
      std::make_shared<DepthImage>(MemoryType::kDevice);

  /// These collections keep track of the blocks which need to be updated on
  /// the next calls to updateMesh(), updateFreespace() upd updateEsdf()
  /// respectively. They are updated when new frames are integrated into the
  /// reconstruction by calls to integrateDepth() and integrateLidarDepth().
  Index3DSet mesh_blocks_to_update_;
  Index3DSet freespace_blocks_to_update_;
  Index3DSet esdf_blocks_to_update_;

  /// This object manages a queue of mesh blocks to be streamed when bandwidth
  /// limiting is desired. Set maintain_mesh_block_stream_queue_ to true to
  /// build the queue during mapping.
  bool maintain_mesh_block_stream_queue_ =
      mapper::kDefaultMaintainMeshBlockStreamQueue;
  MeshStreamerOldestBlocks mesh_streamer_;

  /// Whether to ignore esdf sites that fall into freespace.
  /// When this flag is set, the freespace layer is passed to the esdf
  /// integrator for checking if candidate esdf sites fall into freespace.
  bool ignore_esdf_sites_in_freespace_ = kDefaultIgnoreEsdfSitesInFreespace;
};

}  // namespace nvblox
