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

#include <unordered_set>

#include "nvblox/core/blox.h"
#include "nvblox/core/camera.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/hash.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/layer_cake.h"
#include "nvblox/core/lidar.h"
#include "nvblox/core/oslidar.h"
#include "nvblox/core/voxels.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/projective_color_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/mesh/mesh_integrator.h"

namespace nvblox {

/// The mapper classes wraps layers and integrators together.
/// In the base class we only specify that a mapper should contain map layers
/// and leave it up to sub-classes to add functionality.
class MapperBase {
 public:
  MapperBase() = default;

  /// Move
  MapperBase(MapperBase&& other) = default;
  MapperBase& operator=(MapperBase&& other) = default;

 protected:
  /// Map layers
  LayerCake layers_;
};

/// The RgbdMapper class is what we consider the default mapping behaviour in
/// nvblox.
/// Contains:
/// - TsdfLayer, ColorLayer, EsdfLayer, MeshLayer
/// - Integrators associated with these layer types.
///
/// Exposes functions for:
/// - Integrating depth/rgbd images, 3D LiDAR scans, and color images.
/// - Functions for generating Meshes, ESDF, and ESDF-slices.
class RgbdMapper : public MapperBase {
 public:
  /// The ESDF mode. Enum indicates if an RgbdMapper is configured for 3D or 2D
  /// Esdf production, or that this has not yet been determined (kUnset).
  enum class EsdfMode { k3D, k2D, kUnset };

  RgbdMapper() = delete;
  /// Constructor
  /// @param voxel_size_m The voxel size in meters for the contained layers.
  /// @param memory_type In which type of memory the layers should be stored.
  RgbdMapper(float voxel_size_m, MemoryType memory_type = MemoryType::kDevice);
  virtual ~RgbdMapper() {}

  /// Constructor which initializes from a saved map.
  /// @param map_filepath Path to the serialized map to be loaded.
  /// @param memory_type In which type of memory the layers should be stored.
  RgbdMapper(const std::string& map_filepath,
             MemoryType memory_type = MemoryType::kDevice);

  /// Move
  RgbdMapper(RgbdMapper&& other) = default;
  RgbdMapper& operator=(RgbdMapper&& other) = default;

  /// Integrates a depth frame into the tsdf reconstruction.
  ///@param depth_frame Depth frame to integrate. Depth in the image is
  ///                   specified as a float representing meters.
  ///@param T_L_C Pose of the camera, specified as a transform from Camera-frame
  ///             to Layer-frame transform.
  ///@param camera Intrinsics model of the camera.
  void integrateDepth(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Camera& camera);

  /// Integrates a color frame into the reconstruction.
  ///@param color_frame Color image to integrate.
  ///@param T_L_C Pose of the camera, specified as a transform from Camera-frame
  ///             to Layer-frame transform.
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

  /// Integrates a 3D LiDAR scan into the reconstruction.
  ///@param depth_frame Depth image representing the LiDAR scan. To convert a
  ///                   lidar scan to a DepthImage see TODOOO.
  ///@param T_L_C Pose of the LiDAR, specified as a transform from LiDAR-frame
  ///             to Layer-frame transform.
  ///@param lidar Intrinsics model of the Ouster LiDAR.
  void integrateOSLidarDepth(DepthImage& depth_frame, DepthImage& height_frame,
                             const Transform& T_L_C, OSLidar& oslidar);

  /// Updates the mesh blocks which require an update
  /// @return The indices of the blocks that were updated in this call.
  std::vector<Index3D> updateMesh();

  /// Generate (or re-generate) a mesh for the entire map. Useful if loading
  /// a layer cake without a mesh layer, for example.
  void generateMesh();

  /// Updates the ESDF blocks which require an update.
  /// Note that currently we limit the Mapper class to calculating *either* the
  /// 2D or 3D ESDF, not both. Which is to be calculated is determined by the
  /// first call to updateEsdf().
  ///@return std::vector<Index3D> The indices of the blocks that were updated
  ///        in this call.
  std::vector<Index3D> updateEsdf();

  /// Generate an ESDF on *all* allocated blocks. Will replace whatever has been
  /// done before.
  void generateEsdf();

  /// Updates the ESDF blocks which require an update.
  /// Note that currently we limit the Mapper class to calculating *either* the
  /// 2D or 3D ESDF, not both. Which is to be calculated is determined by the
  /// first call to updateEsdf().
  /// This function operates by collapsing a finite thickness slice of the 3D
  /// TSDF into a binary obstacle map, and then generating the 2D ESDF. The
  /// input parameters define the limits of the 3D slice that are considered.
  /// Note that the resultant 2D ESDF is stored in a single voxel thick layer in
  /// ESDF layer.
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

  /// Getter
  ///@return const LayerCake& The collection of layers mapped.
  const LayerCake& layers() const { return layers_; }
  /// Getter
  ///@return const TsdfLayer& TSDF layer
  const TsdfLayer& tsdf_layer() const { return layers_.get<TsdfLayer>(); }
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
  ///@return const LayerCake& The collection of layers mapped.
  LayerCake& layers() { return layers_; }
  /// Getter
  ///@return const TsdfLayer& TSDF layer
  TsdfLayer& tsdf_layer() { return *layers_.getPtr<TsdfLayer>(); }
  /// Getter
  ///@return const ColorLayer& Color layer
  ColorLayer& color_layer() { return *layers_.getPtr<ColorLayer>(); }
  /// Getter
  ///@return const EsdfLayer& ESDF layer
  EsdfLayer& esdf_layer() { return *layers_.getPtr<EsdfLayer>(); }
  /// Getter
  ///@return const MeshLayer& Mesh layer
  MeshLayer& mesh_layer() { return *layers_.getPtr<MeshLayer>(); }

  /// Getter
  ///@return const ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        depth/rgbd frame integration.
  const ProjectiveTsdfIntegrator& tsdf_integrator() const {
    return tsdf_integrator_;
  }
  /// Getter
  ///@return const ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        3D LiDAR scan integration.
  const ProjectiveTsdfIntegrator& lidar_tsdf_integrator() const {
    return lidar_tsdf_integrator_;
  }
  /// Getter
  ///@return const ProjectiveColorIntegrator& Color integrator.
  const ProjectiveColorIntegrator& color_integrator() const {
    return color_integrator_;
  }
  /// Getter
  ///@return const MeshIntegrator& Mesh integrator
  const MeshIntegrator& mesh_integrator() const { return mesh_integrator_; }
  /// Getter
  ///@return const EsdfIntegrator& ESDF integrator
  const EsdfIntegrator& esdf_integrator() const { return esdf_integrator_; }

  /// Getter
  ///@return const ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        depth/rgbd frame integration.
  ProjectiveTsdfIntegrator& tsdf_integrator() { return tsdf_integrator_; }
  /// Getter
  ///@return const ProjectiveTsdfIntegrator& TSDF integrator used for
  ///        3D LiDAR scan integration.
  ProjectiveTsdfIntegrator& lidar_tsdf_integrator() {
    return lidar_tsdf_integrator_;
  }
  /// Getter
  ///@return const ProjectiveColorIntegrator& Color integrator.
  ProjectiveColorIntegrator& color_integrator() { return color_integrator_; }
  /// Getter
  ///@return const MeshIntegrator& Mesh integrator
  MeshIntegrator& mesh_integrator() { return mesh_integrator_; }
  /// Getter
  ///@return const EsdfIntegrator& ESDF integrator
  EsdfIntegrator& esdf_integrator() { return esdf_integrator_; }

  /// Saving and loading functions.
  /// Saving a map will serialize the TSDF and ESDF layers to a file.
  ///@param filename
  ///@return true
  ///@return false
  bool saveMap(const std::string& filename);
  /// Loading the map will load a the TSDF and ESDF layers from a file.
  /// Will clear anything in the map already.
  bool loadMap(const std::string& filename);

 protected:
  /// The size of the voxels to be used in the TSDF, ESDF, Color layers.
  float voxel_size_m_;
  /// The storage location for the TSDF, ESDF, Color, and Mesh Layers.
  MemoryType memory_type_;

  /// This class can be used to generate *either* (not both) the 2D or 3D ESDF.
  /// The mode used is determined by the first call to either updateEsdf() or
  /// updateEsdfSlice(). This member tracks which mode we're in.
  EsdfMode esdf_mode_ = EsdfMode::kUnset;

  /// Integrators
  ProjectiveTsdfIntegrator tsdf_integrator_;
  ProjectiveTsdfIntegrator lidar_tsdf_integrator_;
  ProjectiveColorIntegrator color_integrator_;
  MeshIntegrator mesh_integrator_;
  EsdfIntegrator esdf_integrator_;

  /// These collections keep track of the blocks which need to be updated on the
  /// next calls to updateMeshLayer() and updateEsdfLayer() respectively. They
  /// are updated when new frames are integrated into the reconstruction by
  /// calls to integrateDepth() and integrateLidarDepth().
  Index3DSet mesh_blocks_to_update_;
  Index3DSet esdf_blocks_to_update_;
};

}  // namespace nvblox