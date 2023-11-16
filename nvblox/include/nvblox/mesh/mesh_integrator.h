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

#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/mesh/internal/marching_cubes.h"
#include "nvblox/mesh/mesh_block.h"
#include "nvblox/core/parameter_tree.h"

namespace nvblox {

/// Class to integrate TSDF data into a mesh using marching cubes.
class MeshIntegrator {
 public:
  static constexpr float kDefaultMinWeight = 1e-4;
  static constexpr bool kDefaultWeldVertices = true;

  MeshIntegrator();
  MeshIntegrator(std::shared_ptr<CudaStream> cuda_stream);
  ~MeshIntegrator() = default;

  /// Chooses the default mesher between CPU and GPU.
  bool integrateMeshFromDistanceField(
      const TsdfLayer& distance_layer, BlockLayer<MeshBlock>* mesh_layer,
      const DeviceType device_type = DeviceType::kGPU);

  /// Integrates only the selected blocks from the distance layer on the CPU.
  /// Prefer to use the GPU version.
  bool integrateBlocksCPU(const TsdfLayer& distance_layer,
                          const std::vector<Index3D>& block_indices,
                          BlockLayer<MeshBlock>* mesh_layer);

  /// @brief Integrate new TSDF blocks into a mesh on the GPU.
  /// @param distance_layer The TSDF layer to integrate.
  /// @param block_indices Which block indices to integrate, can either be
  /// updated ones or all blocks in the TSDF layer.
  /// @param mesh_layer The mesh layer for output.
  /// @return Whether the integration succeeded.
  bool integrateBlocksGPU(const TsdfLayer& distance_layer,
                          const std::vector<Index3D>& block_indices,
                          BlockLayer<MeshBlock>* mesh_layer);

  /// Color mesh layer.
  /// TODO(alexmillane): Currently these functions color vertices by taking the
  /// CLOSEST color. Would be good to have an option at least for interpolation.
  void colorMesh(const ColorLayer& color_layer, MeshLayer* mesh_layer);
  void colorMesh(const ColorLayer& color_layer,
                 const std::vector<Index3D>& block_indices,
                 MeshLayer* mesh_layer);
  void colorMeshGPU(const ColorLayer& color_layer, MeshLayer* mesh_layer);
  void colorMeshGPU(const ColorLayer& color_layer,
                    const std::vector<Index3D>& block_indices,
                    MeshLayer* mesh_layer);
  void colorMeshCPU(const ColorLayer& color_layer, MeshLayer* mesh_layer);
  void colorMeshCPU(const ColorLayer& color_layer,
                    const std::vector<Index3D>& block_indices,
                    MeshLayer* mesh_layer);

  float min_weight() const { return min_weight_; }
  void min_weight(float min_weight) { min_weight_ = min_weight; }

  bool weld_vertices() const { return weld_vertices_; }
  void weld_vertices(bool weld_vertices) { weld_vertices_ = weld_vertices; }

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(const std::string& name_remap = std::string()) const;

 private:
  bool isBlockMeshable(const VoxelBlock<TsdfVoxel>::ConstPtr block,
                       float cutoff) const;

  bool getTriangleCandidatesAroundVoxel(
      const VoxelBlock<TsdfVoxel>::ConstPtr block,
      const std::vector<VoxelBlock<TsdfVoxel>::ConstPtr>& neighbor_blocks,
      const Index3D& index, const Vector3f& voxel_position,
      const float voxel_size,
      marching_cubes::PerVoxelMarchingCubesResults* neighbors);

  void getTriangleCandidatesInBlock(
      const VoxelBlock<TsdfVoxel>::ConstPtr block,
      const std::vector<VoxelBlock<TsdfVoxel>::ConstPtr>& neighbor_blocks,
      const Index3D& block_index, const float block_size,
      std::vector<marching_cubes::PerVoxelMarchingCubesResults>*
          triangle_candidates);

  void getMeshableBlocksGPU(const TsdfLayer& distance_layer,
                            const std::vector<Index3D>& block_indices,
                            float cutoff_distance,
                            std::vector<Index3D>* meshable_blocks);

  void meshBlocksGPU(const TsdfLayer& distance_layer,
                     const std::vector<Index3D>& block_indices,
                     BlockLayer<MeshBlock>* mesh_layer);

  // Weld overlapping vertices together, updaing the normals & indices of the
  // reduced vertex count.
  void weldVertices(device_vector<CudaMeshBlock>* cuda_mesh_blocks);

  // Minimum weight to actually mesh.
  float min_weight_ = kDefaultMinWeight;

  // The TSDF distance below which we consider a voxel for meshing.
  float cutoff_distance_vox_ = 5.0f;

  // Whether to perform vertex welding or not. It cuts down number
  // of vertices by 5x.
  bool weld_vertices_ = kDefaultWeldVertices;

  // Offsets for cube indices.
  Eigen::Matrix<int, 3, 8> cube_index_offsets_;

  // The color that the mesh takes if no coloring is available.
  Color default_mesh_color_ = Color::Gray();

  // State.
  std::shared_ptr<CudaStream> cuda_stream_;

  // These are temporary variables so we don't have to allocate every single
  // frame.
  host_vector<const VoxelBlock<TsdfVoxel>*> block_ptrs_host_;
  device_vector<const VoxelBlock<TsdfVoxel>*> block_ptrs_device_;
  host_vector<bool> meshable_host_;
  device_vector<bool> meshable_device_;
  host_vector<Vector3f> block_positions_host_;
  device_vector<Vector3f> block_positions_device_;
  host_vector<CudaMeshBlock> mesh_blocks_host_;
  device_vector<CudaMeshBlock> mesh_blocks_device_;
  // Caches for welding.
  device_vector<Vector3f> input_vertices_;
  device_vector<Vector3f> input_normals_;

  // Intermediate marching cube results.
  device_vector<marching_cubes::PerVoxelMarchingCubesResults>
      marching_cubes_results_device_;
  device_vector<int> mesh_block_sizes_device_;
  host_vector<int> mesh_block_sizes_host_;
};

}  // namespace nvblox
