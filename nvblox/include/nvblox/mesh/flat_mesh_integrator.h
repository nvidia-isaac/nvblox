/*
Copyright 2026 NVIDIA CORPORATION

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

#include <algorithm>
#include <memory>
#include <vector>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/map/common_names.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/mesh/mesh_integrator_params.h"

namespace nvblox {

/// Single-pass flat mesh integrator using marching cubes.
///
/// Extracts a triangle mesh from a TSDF layer (with optional appearance) into
/// a flat Mesh<AppearanceType> in a single GPU kernel launch with atomic
/// triangle output. Significantly faster than MeshIntegrator for use cases
/// that require a single GPU-friendly mesh buffer (e.g. real-time rendering).
///
/// Note: this is distinct from MeshSerializerGPU, which serializes an
/// already-extracted MeshLayer into flat host buffers for streaming.
/// FlatMeshIntegrator extracts geometry directly from the TSDF.
///
/// Key differences from MeshIntegrator:
/// - Single kernel: table lookup + vertex interpolation (+ appearance) in one
///   pass
/// - No intermediate CPU synchronization or per-block memory allocation
/// - Triangle list output (no index buffer, no vertex welding)
/// - Writes directly into the output Mesh with auto-growing on overflow
/// - Full re-extraction each call (not incremental)
///
/// Follows the same template convention as MeshIntegrator: templated on
/// AppearanceVoxelType, with geometry as the base operation and appearance
/// as optional.
///
/// Usage (standalone integrator):
/// @code
/// ColorFlatMeshIntegrator flat_mesh(cuda_stream);
/// auto block_indices = tsdf_layer.getAllBlockIndices();
///
/// // Geometry only:
/// ColorMesh mesh;
/// flat_mesh.integrateBlocks(tsdf_layer, block_indices, &mesh);
///
/// // Geometry + color in single pass:
/// flat_mesh.integrateBlocks(tsdf_layer, color_layer, block_indices, &mesh);
///
/// // Features instead of color:
/// FeatureFlatMeshIntegrator feat_flat_mesh(cuda_stream);
/// FeatureMesh feat_mesh;
/// feat_flat_mesh.integrateBlocks(tsdf_layer, feature_layer, block_indices,
///                                &feat_mesh);
/// @endcode
///
/// Usage (via Mapper convenience methods):
/// @code
/// mapper.updateFlatColorMesh();                          // all blocks + color
/// mapper.updateFlatColorMesh(camera, T_L_C, max_depth);  // frustum-culled
/// mapper.updateFlatColorMeshGeometryOnly();               // no color, faster
/// mapper.updateFlatFeatureMesh();                         // features
/// const ColorMesh& mesh = mapper.flat_color_mesh();
///
/// // Or equivalently, using the template API:
/// mapper.updateFlatMesh<ColorVoxel>();
/// mapper.updateFlatMesh<ColorVoxel>(camera, T_L_C, max_depth);
/// mapper.updateFlatMesh<FeatureVoxel>();
/// mapper.updateFlatMesh<FeatureVoxel>(camera, T_L_C, max_depth);
/// @endcode
template <typename AppearanceVoxelType>
class FlatMeshIntegrator {
  using AppearanceType = typename AppearanceVoxelType::ArrayType;
  using MeshType = Mesh<AppearanceType>;
  using AppearanceLayerType = VoxelBlockLayer<AppearanceVoxelType>;

  static constexpr int kMinTriangleBufferSize = 1024;

 public:
  FlatMeshIntegrator();
  explicit FlatMeshIntegrator(std::shared_ptr<CudaStream> cuda_stream);
  ~FlatMeshIntegrator();

  FlatMeshIntegrator(FlatMeshIntegrator&&);
  FlatMeshIntegrator& operator=(FlatMeshIntegrator&&);
  FlatMeshIntegrator(const FlatMeshIntegrator&) = delete;
  FlatMeshIntegrator& operator=(const FlatMeshIntegrator&) = delete;

  /// @brief Extract geometry-only mesh from TSDF layer.
  /// Fills vertices, normals, and triangles. vertex_appearances are cleared.
  /// @param distance_layer The TSDF layer (geometry source).
  /// @param block_indices Which blocks to mesh.
  /// @param mesh Output mesh.
  /// @param fill_triangle_indices If true (default), populates the trivial
  ///   triangle index buffer (triangles[i] = i). Set to false to skip this
  ///   step when the caller can assume identity indexing.
  /// @return Actual triangle count (may exceed max_num_triangles before
  ///   auto-grow; after auto-grow, equals the output triangle count).
  int integrateBlocks(const TsdfLayer& distance_layer,
                      const std::vector<Index3D>& block_indices, MeshType* mesh,
                      bool fill_triangle_indices = true);

  /// @brief Extract mesh with appearance from TSDF + appearance layers.
  /// Geometry and appearance are computed together in a single GPU pass.
  /// @param distance_layer The TSDF layer (geometry source).
  /// @param appearance_layer The appearance layer (color/feature source).
  /// @param block_indices Which blocks to mesh.
  /// @param mesh Output mesh.
  /// @param fill_triangle_indices If true (default), populates the trivial
  ///   triangle index buffer (triangles[i] = i). Set to false to skip.
  /// @return Actual triangle count.
  int integrateBlocks(const TsdfLayer& distance_layer,
                      const AppearanceLayerType& appearance_layer,
                      const std::vector<Index3D>& block_indices, MeshType* mesh,
                      bool fill_triangle_indices = true);

  /// Parameter getters/setters (matching MeshIntegrator convention).
  float min_weight() const { return min_weight_; }
  void min_weight(float min_weight) { min_weight_ = min_weight; }

  int max_num_triangles() const { return max_num_triangles_; }
  void max_num_triangles(int max_num_triangles) {
    max_num_triangles_ = std::max(max_num_triangles, kMinTriangleBufferSize);
  }

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 private:
  /// Shared implementation for both integrateBlocks overloads.
  /// When appearance_layer is nullptr, only geometry is extracted.
  /// @return Actual triangle count.
  int integrateBlocksImpl(const TsdfLayer& tsdf_layer,
                          const AppearanceLayerType* appearance_layer,
                          const std::vector<Index3D>& block_indices,
                          MeshType* mesh, bool fill_triangle_indices);

  /// Shared setup: transfer block pointers and indices to device.
  void prepareBlockData(const TsdfLayer& tsdf_layer,
                        const std::vector<Index3D>& block_indices);

  // Parameters
  float min_weight_ = kMeshIntegratorMinWeightParamDesc.default_value;
  int max_num_triangles_ =
      kMeshIntegratorMaxFlatMeshTrianglesParamDesc.default_value;

  // CUDA stream
  std::shared_ptr<CudaStream> cuda_stream_;

  // Atomic triangle counter (kernel writes via atomicAdd, host reads after
  // sync)
  unified_ptr<int> triangle_counter_;

  // Block pointer transfer buffers
  host_vector<const TsdfBlock*> tsdf_block_ptrs_host_;
  device_vector<const TsdfBlock*> tsdf_block_ptrs_device_;
  host_vector<Index3D> block_indices_host_;
  device_vector<Index3D> block_indices_device_;
};

using ColorFlatMeshIntegrator = FlatMeshIntegrator<ColorVoxel>;
using FeatureFlatMeshIntegrator = FlatMeshIntegrator<FeatureVoxel>;

}  // namespace nvblox
