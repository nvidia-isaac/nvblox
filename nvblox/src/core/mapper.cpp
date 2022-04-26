#include "nvblox/core/mapper.h"

namespace nvblox {

RgbdMapper::RgbdMapper(float voxel_size_m, MemoryType memory_type)
    : voxel_size_m_(voxel_size_m) {
  layers_ = LayerCake::create<TsdfLayer, ColorLayer, EsdfLayer, MeshLayer>(
      voxel_size_m_, memory_type);
}

void RgbdMapper::integrateDepth(const DepthImage& depth_frame,
                                const Transform& T_L_C, const Camera& camera) {
  // Call the integrator.
  std::vector<Index3D> updated_blocks;
  tsdf_integrator_.integrateFrame(depth_frame, T_L_C, camera,
                                  layers_.getPtr<TsdfLayer>(), &updated_blocks);

  // Update all the relevant queues.
  mesh_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
  esdf_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
}

void RgbdMapper::integrateColor(const ColorImage& color_frame,
                                const Transform& T_L_C, const Camera& camera) {
  color_integrator_.integrateFrame(color_frame, T_L_C, camera,
                                   layers_.get<TsdfLayer>(),
                                   layers_.getPtr<ColorLayer>());
}

std::vector<Index3D> RgbdMapper::updateMesh() {
  // Convert the set of MeshBlocks needing an update to a vector
  std::vector<Index3D> mesh_blocks_to_update_vector(
      mesh_blocks_to_update_.begin(), mesh_blocks_to_update_.end());

  // Call the integrator.
  mesh_integrator_.integrateBlocksGPU(layers_.get<TsdfLayer>(),
                                      mesh_blocks_to_update_vector,
                                      layers_.getPtr<MeshLayer>());

  mesh_integrator_.colorMesh(layers_.get<ColorLayer>(),
                             mesh_blocks_to_update_vector,
                             layers_.getPtr<MeshLayer>());

  // Mark blocks as updated
  mesh_blocks_to_update_.clear();

  return mesh_blocks_to_update_vector;
}

std::vector<Index3D> RgbdMapper::updateEsdf() {
  CHECK(esdf_mode_ != EsdfMode::k2D)
      << "Currently, we limit computation of the ESDF to 2d *or* 3d. Not both.";
  esdf_mode_ = EsdfMode::k3D;

  // Convert the set of EsdfBlocks needing an update to a vector
  std::vector<Index3D> esdf_blocks_to_update_vector(
      esdf_blocks_to_update_.begin(), esdf_blocks_to_update_.end());

  esdf_integrator_.integrateBlocksOnGPU(layers_.get<TsdfLayer>(),
                                        esdf_blocks_to_update_vector,
                                        layers_.getPtr<EsdfLayer>());

  // Mark blocks as updated
  esdf_blocks_to_update_.clear();

  return esdf_blocks_to_update_vector;
}

std::vector<Index3D> RgbdMapper::updateEsdfSlice(float slice_input_z_min,
                                                 float slice_input_z_max,
                                                 float slice_output_z) {
  CHECK(esdf_mode_ != EsdfMode::k3D)
      << "Currently, we limit computation of the ESDF to 2d *or* 3d. Not both.";
  esdf_mode_ = EsdfMode::k2D;

  // Convert the set of MeshBlocks needing an update to a vector
  std::vector<Index3D> esdf_blocks_to_update_vector(
      esdf_blocks_to_update_.begin(), esdf_blocks_to_update_.end());

  esdf_integrator_.integrateSliceOnGPU(
      layers_.get<TsdfLayer>(), esdf_blocks_to_update_vector, slice_input_z_min,
      slice_input_z_max, slice_output_z, layers_.getPtr<EsdfLayer>());

  // Mark blocks as updated
  esdf_blocks_to_update_.clear();

  return esdf_blocks_to_update_vector;
}

}  // namespace nvblox
