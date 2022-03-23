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
#include "nvblox/core/common_names.h"
#include "nvblox/core/hash.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/layer_cake.h"
#include "nvblox/core/voxels.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/projective_color_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/mesh/mesh_integrator.h"

namespace nvblox {

// The mapper classes wraps layers and integrators together.
// In the base class we only specify that a mapper should contain map layers and
// leave it up to sub-classes to add functionality.
class MapperBase {
 public:
  MapperBase() = default;

  // Move
  MapperBase(MapperBase&& other) = default;
  MapperBase& operator=(MapperBase&& other) = default;

 protected:
  // Map layers
  LayerCake layers_;
};

// The RgbdMapper class is what we consider the default (but extensible) mapping
// behaviour in nvblox.
// Contains:
// - TsdfLayer, ColorLayer, EsdfLayer, MeshLayer
// - Functions for integrating depth and color frames
// - Function for generating Meshes, ESDF, and ESDF-slices
class RgbdMapper : public MapperBase {
 public:
  RgbdMapper() = delete;
  RgbdMapper(float voxel_size_m, MemoryType memory_type = MemoryType::kDevice);
  virtual ~RgbdMapper() {}

  // Move
  RgbdMapper(RgbdMapper&& other) = default;
  RgbdMapper& operator=(RgbdMapper&& other) = default;

  void integrateDepth(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Camera& camera);
  void integrateColor(const ColorImage& color_frame, const Transform& T_L_C,
                      const Camera& camera);

  /// Updates the mesh blocks which require an update
  /// @return The indices of the blocks that were updated in this call.
  std::vector<Index3D> updateMesh();

  /// Updates the Esdf blocks which require update
  /// @return The indices of the blocks that were updated in this call.
  std::vector<Index3D> updateEsdf();
  std::vector<Index3D> updateEsdfSlice(float slice_input_z_min,
                                       float slice_input_z_max,
                                       float slice_output_z);

  // Const access to layers
  const LayerCake& layers() const { return layers_; }
  const TsdfLayer& tsdf_layer() const { return layers_.get<TsdfLayer>(); }
  const ColorLayer& color_layer() const { return layers_.get<ColorLayer>(); }
  const EsdfLayer& esdf_layer() const { return layers_.get<EsdfLayer>(); }
  const MeshLayer& mesh_layer() const { return layers_.get<MeshLayer>(); }

  // Non-const access to layers
  LayerCake& layers() { return layers_; }
  TsdfLayer& tsdf_layer() { return *layers_.getPtr<TsdfLayer>(); }
  ColorLayer& color_layer() { return *layers_.getPtr<ColorLayer>(); }
  EsdfLayer& esdf_layer() { return *layers_.getPtr<EsdfLayer>(); }
  MeshLayer& mesh_layer() { return *layers_.getPtr<MeshLayer>(); }

  // Const access to integrators
  const ProjectiveTsdfIntegrator& tsdf_integrator() const {
    return tsdf_integrator_;
  }
  const ProjectiveColorIntegrator& color_integrator() const {
    return color_integrator_;
  }
  const MeshIntegrator& mesh_integrator() const { return mesh_integrator_; }
  const EsdfIntegrator& esdf_integrator() const { return esdf_integrator_; }

  // Non-const access to integrators
  ProjectiveTsdfIntegrator& tsdf_integrator() { return tsdf_integrator_; }
  ProjectiveColorIntegrator& color_integrator() { return color_integrator_; }
  MeshIntegrator& mesh_integrator() { return mesh_integrator_; }
  EsdfIntegrator& esdf_integrator() { return esdf_integrator_; }

 protected:
  // Params
  float voxel_size_m_;

  // Behaviour params
  enum class EsdfMode { k3D, k2D, kUnset };
  EsdfMode esdf_mode_ = EsdfMode::kUnset;

  // Map layers defined in base)
  // For RgbdMapper class: TSDF, Color, ESDF, Mesh
  // LayerCake layers_;

  // Integrators
  ProjectiveTsdfIntegrator tsdf_integrator_;
  ProjectiveColorIntegrator color_integrator_;
  MeshIntegrator mesh_integrator_;
  EsdfIntegrator esdf_integrator_;

  // These queue keep track of the blocks which need to be updated on the next
  // calls to updateMeshLayer() and updateEsdfLayer() respectively.
  Index3DSet mesh_blocks_to_update_;
  Index3DSet esdf_blocks_to_update_;
};

}  // namespace nvblox