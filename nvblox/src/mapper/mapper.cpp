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
#include "nvblox/mapper/mapper.h"

#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/io/layer_cake_io.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/pointcloud_io.h"

namespace nvblox {

Mapper::Mapper(float voxel_size_m, MemoryType memory_type,
               ProjectiveLayerType projective_layer_type)
    : voxel_size_m_(voxel_size_m),
      memory_type_(memory_type),
      projective_layer_type_(projective_layer_type) {
  layers_ = LayerCake::create<TsdfLayer, ColorLayer, OccupancyLayer, EsdfLayer,
                              MeshLayer>(voxel_size_m_, memory_type);
}

Mapper::Mapper(const std::string& map_filepath, MemoryType memory_type)
    : memory_type_(memory_type) {
  loadMap(map_filepath);
}

void Mapper::integrateDepth(const DepthImage& depth_frame,
                            const Transform& T_L_C, const Camera& camera) {
  // Call the integrator.
  std::vector<Index3D> updated_blocks;
  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    tsdf_integrator_.integrateFrame(depth_frame, T_L_C, camera,
                                    layers_.getPtr<TsdfLayer>(),
                                    &updated_blocks);
    // The mesh is only updated for the tsdf projective layer
    mesh_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    occupancy_integrator_.integrateFrame(depth_frame, T_L_C, camera,
                                         layers_.getPtr<OccupancyLayer>(),
                                         &updated_blocks);
  }
  // Update all the relevant queues.
  esdf_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
}

void Mapper::integrateLidarDepth(const DepthImage& depth_frame,
                                 const Transform& T_L_C, const Lidar& lidar) {
  // Call the integrator.
  std::vector<Index3D> updated_blocks;
  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    lidar_tsdf_integrator_.integrateFrame(depth_frame, T_L_C, lidar,
                                          layers_.getPtr<TsdfLayer>(),
                                          &updated_blocks);
    // The mesh is only updated for the tsdf projective layer
    mesh_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    lidar_occupancy_integrator_.integrateFrame(depth_frame, T_L_C, lidar,
                                               layers_.getPtr<OccupancyLayer>(),
                                               &updated_blocks);
  }
  // Update all the relevant queues.
  esdf_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
}

void Mapper::integrateColor(const ColorImage& color_frame,
                            const Transform& T_L_C, const Camera& camera) {
  color_integrator_.integrateFrame(color_frame, T_L_C, camera,
                                   layers_.get<TsdfLayer>(),
                                   layers_.getPtr<ColorLayer>());
}

void Mapper::decayOccupancy() {
  // The esdf of all blocks has to be updated after decay
  std::vector<Index3D> all_blocks =
      layers_.get<OccupancyLayer>().getAllBlockIndices();
  esdf_blocks_to_update_.insert(all_blocks.begin(), all_blocks.end());

  occupancy_decay_integrator_.decay(layers_.getPtr<OccupancyLayer>());
}

std::vector<Index3D> Mapper::updateMesh() {
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

void Mapper::generateMesh() {
  mesh_integrator_.integrateBlocksGPU(
      layers_.get<TsdfLayer>(), layers_.get<TsdfLayer>().getAllBlockIndices(),
      layers_.getPtr<MeshLayer>());
}

std::vector<Index3D> Mapper::updateEsdf() {
  CHECK(esdf_mode_ != EsdfMode::k2D)
      << "Currently, we limit computation of the ESDF to 2d *or* 3d. Not both.";
  esdf_mode_ = EsdfMode::k3D;

  // Convert the set of EsdfBlocks needing an update to a vector
  std::vector<Index3D> esdf_blocks_to_update_vector(
      esdf_blocks_to_update_.begin(), esdf_blocks_to_update_.end());

  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    esdf_integrator_.integrateBlocks(layers_.get<TsdfLayer>(),
                                     esdf_blocks_to_update_vector,
                                     layers_.getPtr<EsdfLayer>());
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    esdf_integrator_.integrateBlocks(layers_.get<OccupancyLayer>(),
                                     esdf_blocks_to_update_vector,
                                     layers_.getPtr<EsdfLayer>());
  }

  // Mark blocks as updated
  esdf_blocks_to_update_.clear();

  return esdf_blocks_to_update_vector;
}

void Mapper::generateEsdf() {
  CHECK(esdf_mode_ != EsdfMode::k2D)
      << "Currently, we limit computation of the ESDF to 2d *or* 3d. Not both.";
  esdf_mode_ = EsdfMode::k3D;

  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    esdf_integrator_.integrateBlocks(
        layers_.get<TsdfLayer>(), layers_.get<TsdfLayer>().getAllBlockIndices(),
        layers_.getPtr<EsdfLayer>());
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    esdf_integrator_.integrateBlocks(
        layers_.get<OccupancyLayer>(),
        layers_.get<OccupancyLayer>().getAllBlockIndices(),
        layers_.getPtr<EsdfLayer>());
  }
}

std::vector<Index3D> Mapper::updateEsdfSlice(float slice_input_z_min,
                                             float slice_input_z_max,
                                             float slice_output_z) {
  CHECK(esdf_mode_ != EsdfMode::k3D)
      << "Currently, we limit computation of the ESDF to 2d *or* 3d. Not both.";
  esdf_mode_ = EsdfMode::k2D;

  // Convert the set of MeshBlocks needing an update to a vector
  std::vector<Index3D> esdf_blocks_to_update_vector(
      esdf_blocks_to_update_.begin(), esdf_blocks_to_update_.end());

  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    esdf_integrator_.integrateSlice(
        layers_.get<TsdfLayer>(), esdf_blocks_to_update_vector,
        slice_input_z_min, slice_input_z_max, slice_output_z,
        layers_.getPtr<EsdfLayer>());
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    esdf_integrator_.integrateSlice(
        layers_.get<OccupancyLayer>(), esdf_blocks_to_update_vector,
        slice_input_z_min, slice_input_z_max, slice_output_z,
        layers_.getPtr<EsdfLayer>());
  }

  // Mark blocks as updated
  esdf_blocks_to_update_.clear();

  return esdf_blocks_to_update_vector;
}

std::vector<Index3D> Mapper::clearOutsideRadius(const Vector3f& center,
                                                float radius) {
  std::vector<Index3D> block_indices_for_deletion;
  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    block_indices_for_deletion = getBlocksOutsideRadius(layers_.get<TsdfLayer>().getAllBlockIndices(),
                          layers_.get<TsdfLayer>().block_size(), center,
                          radius);
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    block_indices_for_deletion = getBlocksOutsideRadius(layers_.get<OccupancyLayer>().getAllBlockIndices(),
                      layers_.get<OccupancyLayer>().block_size(), center,
                      radius);
  }
  for (const Index3D& idx : block_indices_for_deletion) {
    mesh_blocks_to_update_.erase(idx);
    esdf_blocks_to_update_.erase(idx);
  }
  layers_.getPtr<TsdfLayer>()->clearBlocks(block_indices_for_deletion);
  layers_.getPtr<ColorLayer>()->clearBlocks(block_indices_for_deletion);
  layers_.getPtr<EsdfLayer>()->clearBlocks(block_indices_for_deletion);
  layers_.getPtr<MeshLayer>()->clearBlocks(block_indices_for_deletion);
  layers_.getPtr<OccupancyLayer>()->clearBlocks(block_indices_for_deletion);
  return block_indices_for_deletion;
}

void Mapper::markUnobservedTsdfFreeInsideRadius(const Vector3f& center,
                                                float radius) {
  CHECK_GT(radius, 0.0f);
  std::vector<Index3D> updated_blocks;
  tsdf_integrator_.markUnobservedFreeInsideRadius(
      center, radius, layers_.getPtr<TsdfLayer>(), &updated_blocks);

  mesh_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
  esdf_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
}

bool Mapper::saveMap(const std::string& filename) {
  return io::writeLayerCakeToFile(filename, layers_);
}

bool Mapper::saveMap(const char* filename) {
  return saveMap(std::string(filename));
}

bool Mapper::loadMap(const std::string& filename) {
  LayerCake new_cake = io::loadLayerCakeFromFile(filename, memory_type_);
  // Will return an empty cake if anything went wrong.
  if (new_cake.empty()) {
    LOG(ERROR) << "Failed to load map from file: " << filename;
    return false;
  }

  TsdfLayer* tsdf_layer = new_cake.getPtr<TsdfLayer>();

  if (tsdf_layer == nullptr) {
    LOG(ERROR) << "No TSDF layer could be loaded from file: " << filename
               << ". Aborting loading.";
    return false;
  }
  // Double check what's going on with the voxel sizes.
  if (tsdf_layer->voxel_size() != voxel_size_m_) {
    LOG(INFO) << "Setting the voxel size from the loaded map as: "
              << tsdf_layer->voxel_size();
    voxel_size_m_ = tsdf_layer->voxel_size();
  }

  // Now we're happy, let's swap the cakes.
  layers_ = std::move(new_cake);

  // We can't serialize mesh layers yet so we have to add a new mesh layer.
  std::unique_ptr<MeshLayer> mesh(
      new MeshLayer(layers_.getPtr<TsdfLayer>()->block_size(), memory_type_));
  layers_.insert(typeid(MeshLayer), std::move(mesh));

  // Clear the to update vectors.
  esdf_blocks_to_update_.clear();
  // Force the mesh to update everything.
  mesh_blocks_to_update_.clear();
  const std::vector<Index3D> all_tsdf_blocks =
      layers_.getPtr<TsdfLayer>()->getAllBlockIndices();
  mesh_blocks_to_update_.insert(all_tsdf_blocks.begin(), all_tsdf_blocks.end());

  updateMesh();
  return true;
}

bool Mapper::loadMap(const char* filename) {
  return loadMap(std::string(filename));
}

bool Mapper::saveMeshAsPly(const std::string& filepath) {
  updateMesh();
  return io::outputMeshLayerToPly(mesh_layer(), filepath);
}

bool Mapper::saveEsdfAsPly(const std::string& filename) {
  updateEsdf();
  return io::outputVoxelLayerToPly(esdf_layer(), filename);
}

}  // namespace nvblox
