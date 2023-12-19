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
#include "nvblox/utils/rates.h"

namespace nvblox {

std::string toString(EsdfMode esdf_mode) {
  switch (esdf_mode) {
    case EsdfMode::k3D:
      return "k3D";
      break;
    case EsdfMode::k2D:
      return "k2D";
      break;
    case EsdfMode::kUnset:
      return "kUnset";
      break;
    default:
      LOG(FATAL) << "Not implemented";
      break;
  }
}

Mapper::Mapper(float voxel_size_m, MemoryType memory_type,
               ProjectiveLayerType projective_layer_type,
               std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream),
      voxel_size_m_(voxel_size_m),
      memory_type_(memory_type),
      projective_layer_type_(projective_layer_type),
      tsdf_integrator_(cuda_stream),
      lidar_tsdf_integrator_(cuda_stream),
      freespace_integrator_(cuda_stream),
      occupancy_integrator_(cuda_stream),
      lidar_occupancy_integrator_(cuda_stream),
      color_integrator_(cuda_stream),
      mesh_integrator_(cuda_stream),
      esdf_integrator_(cuda_stream),
      depth_preprocessor_(cuda_stream) {
  layers_ =
      LayerCake::create<TsdfLayer, ColorLayer, FreespaceLayer, OccupancyLayer,
                        EsdfLayer, MeshLayer>(voxel_size_m_, memory_type);
}

Mapper::Mapper(const std::string& map_filepath, MemoryType memory_type,
               std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream),
      memory_type_(memory_type),
      tsdf_integrator_(cuda_stream),
      lidar_tsdf_integrator_(cuda_stream),
      freespace_integrator_(cuda_stream),
      occupancy_integrator_(cuda_stream),
      lidar_occupancy_integrator_(cuda_stream),
      color_integrator_(cuda_stream),
      mesh_integrator_(cuda_stream),
      esdf_integrator_(cuda_stream),
      depth_preprocessor_(cuda_stream) {
  loadMap(map_filepath);
}

void Mapper::setMapperParams(const MapperParams& params) {
  // ======= MAPPER =======
  // depth preprocessing
  do_depth_preprocessing(params.do_depth_preprocessing);
  depth_preprocessing_num_dilations(params.depth_preprocessing_num_dilations);
  // mesh streaming
  mesh_bandwidth_limit_mbps(params.mesh_bandwidth_limit_mbps);
  // 2d esdf slice
  esdf_slice_min_height(params.esdf_slice_min_height);
  esdf_slice_max_height(params.esdf_slice_max_height);
  esdf_slice_height(params.esdf_slice_height);
  // Decay
  exclude_last_view_from_decay(params.exclude_last_view_from_decay);

  // ======= PROJECTIVE INTEGRATOR (TSDF/COLOR/OCCUPANCY)
  // max integration distance
  tsdf_integrator().max_integration_distance_m(
      params.projective_integrator_max_integration_distance_m);
  occupancy_integrator().max_integration_distance_m(
      params.projective_integrator_max_integration_distance_m);
  color_integrator().max_integration_distance_m(
      params.projective_integrator_max_integration_distance_m);
  lidar_tsdf_integrator().max_integration_distance_m(
      params.lidar_projective_integrator_max_integration_distance_m);
  lidar_occupancy_integrator().max_integration_distance_m(
      params.lidar_projective_integrator_max_integration_distance_m);
  // truncation distance
  tsdf_integrator().truncation_distance_vox(
      params.projective_integrator_truncation_distance_vox);
  occupancy_integrator().truncation_distance_vox(
      params.projective_integrator_truncation_distance_vox);
  lidar_tsdf_integrator().truncation_distance_vox(
      params.projective_integrator_truncation_distance_vox);
  lidar_occupancy_integrator().truncation_distance_vox(
      params.projective_integrator_truncation_distance_vox);
  // weighting
  tsdf_integrator().weighting_function_type(
      params.projective_integrator_weighting_mode);
  color_integrator().weighting_function_type(
      params.projective_integrator_weighting_mode);
  // max weight
  tsdf_integrator().max_weight(params.projective_integrator_max_weight);
  lidar_tsdf_integrator().max_weight(params.projective_integrator_max_weight);
  color_integrator().max_weight(params.projective_integrator_max_weight);

  // ======= OCCUPANCY INTEGRATOR =======
  occupancy_integrator().free_region_occupancy_probability(
      params.free_region_occupancy_probability);
  lidar_occupancy_integrator().free_region_occupancy_probability(
      params.free_region_occupancy_probability);
  occupancy_integrator().occupied_region_occupancy_probability(
      params.occupied_region_occupancy_probability);
  lidar_occupancy_integrator().occupied_region_occupancy_probability(
      params.occupied_region_occupancy_probability);
  occupancy_integrator().unobserved_region_occupancy_probability(
      params.unobserved_region_occupancy_probability);
  lidar_occupancy_integrator().unobserved_region_occupancy_probability(
      params.unobserved_region_occupancy_probability);
  occupancy_integrator().occupied_region_half_width_m(
      params.occupied_region_half_width_m);
  lidar_occupancy_integrator().occupied_region_half_width_m(
      params.occupied_region_half_width_m);

  // ======= ESDF INTEGRATOR =======
  esdf_integrator().max_esdf_distance_m(params.esdf_integrator_max_distance_m);
  esdf_integrator().min_weight(params.esdf_integrator_min_weight);
  esdf_integrator().max_site_distance_vox(
      params.esdf_integrator_max_site_distance_vox);

  // ======= MESH INTEGRATOR =======
  mesh_integrator().min_weight(params.mesh_integrator_min_weight);
  mesh_integrator().weld_vertices(params.mesh_integrator_weld_vertices);

  // ======= TSDF DECAY INTEGRATOR =======
  tsdf_decay_integrator().decay_factor(params.tsdf_decay_factor);

  // ======= OCCUPANCY DECAY INTEGRATOR =======
  occupancy_decay_integrator().free_region_decay_probability(
      params.free_region_decay_probability);
  occupancy_decay_integrator().occupied_region_decay_probability(
      params.occupied_region_decay_probability);

  // ======= FREESPACE INTEGRATOR =======
  freespace_integrator().max_tsdf_distance_for_occupancy_m(
      params.max_tsdf_distance_for_occupancy_m);
  freespace_integrator().max_unobserved_to_keep_consecutive_occupancy_ms(
      params.max_unobserved_to_keep_consecutive_occupancy_ms);
  freespace_integrator().min_duration_since_occupied_for_freespace_ms(
      params.min_duration_since_occupied_for_freespace_ms);
  freespace_integrator().min_consecutive_occupancy_duration_for_reset_ms(
      params.min_consecutive_occupancy_duration_for_reset_ms);
  freespace_integrator().check_neighborhood(params.check_neighborhood);

  // ======= MESH STREAMER =======
  mesh_streamer().exclusion_height_m(params.mesh_streamer_exclusion_height_m);
  mesh_streamer().exclusion_radius_m(params.mesh_streamer_exclusion_radius_m);
}

const DepthImage& Mapper::preprocessDepthImageAsync(
    const DepthImage& depth_image) {
  // NOTE(alexmillane): We return a const reference to an image, to
  // avoid reallocating.
  // Copy in the depth image
  preprocessed_depth_image_->copyFromAsync(depth_image, *cuda_stream_);
  // Dilate the invalid regions
  if (depth_preprocessing_num_dilations_ > 0) {
    depth_preprocessor_.dilateInvalidRegionsAsync(
        depth_preprocessing_num_dilations_, preprocessed_depth_image_.get());
  } else {
    LOG(WARNING) << "You requested preprocessing, but requested "
                 << depth_preprocessing_num_dilations_
                 << "invalid region dilations. Currenly dilation is the only "
                    "preprocessing step, so doing nothing.";
  }
  return *preprocessed_depth_image_;
}

void Mapper::integrateDepth(const DepthImage& depth_frame,
                            const Transform& T_L_C, const Camera& camera) {
  CHECK(projective_layer_type_ != ProjectiveLayerType::kNone)
      << "You are trying to update on an inexistent projective layer.";
  // If requested, we perform preprocessing of the depth image. At the moment
  // this is just (optional) dilation of the invalid regions.
  const DepthImage& depth_image_for_integration =
      (do_depth_preprocessing_) ? preprocessDepthImageAsync(depth_frame)
                                : depth_frame;

  // Call the integrator.
  std::vector<Index3D> updated_blocks;
  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    tsdf_integrator_.integrateFrame(depth_image_for_integration, T_L_C, camera,
                                    layers_.getPtr<TsdfLayer>(),
                                    &updated_blocks);
    // The mesh and freespace is only updated for the tsdf projective layer
    mesh_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
    freespace_blocks_to_update_.insert(updated_blocks.begin(),
                                       updated_blocks.end());
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    occupancy_integrator_.integrateFrame(
        depth_image_for_integration, T_L_C, camera,
        layers_.getPtr<OccupancyLayer>(), &updated_blocks);
  }

  // Save the viewpoint for use in viewpoint exclusion.
  if (exclude_last_view_from_decay_) {
    if (!last_depth_image_.has_value()) {
      last_depth_image_ =
          DepthImage(depth_image_for_integration.rows(),
                     depth_image_for_integration.cols(), MemoryType::kDevice);
    }
    // NOTE(alexmillane): We could get rid this copy by using a double buffer.
    last_depth_image_.value().copyFrom(depth_image_for_integration);
    last_depth_camera_ = camera;
    last_depth_T_L_C_ = T_L_C;
  }

  // Update all the relevant queues.
  esdf_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
}

void Mapper::integrateLidarDepth(const DepthImage& depth_frame,
                                 const Transform& T_L_C, const Lidar& lidar) {
  CHECK(projective_layer_type_ != ProjectiveLayerType::kNone)
      << "You are trying to update on an inexistent projective layer.";
  // Call the integrator.
  std::vector<Index3D> updated_blocks;
  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    lidar_tsdf_integrator_.integrateFrame(depth_frame, T_L_C, lidar,
                                          layers_.getPtr<TsdfLayer>(),
                                          &updated_blocks);
    // The mesh and freespace is only updated for the tsdf projective layer
    mesh_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
    freespace_blocks_to_update_.insert(updated_blocks.begin(),
                                       updated_blocks.end());
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
  // Color is only integrated for Tsdf layers (not for occupancy)
  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    color_integrator_.integrateFrame(color_frame, T_L_C, camera,
                                     layers_.get<TsdfLayer>(),
                                     layers_.getPtr<ColorLayer>());
  }
}

std::vector<Index3D> Mapper::decayTsdf() {
  // The esdf of all blocks has to be updated after decay
  const std::vector<Index3D> all_blocks =
      layers_.get<TsdfLayer>().getAllBlockIndices();
  esdf_blocks_to_update_.insert(all_blocks.begin(), all_blocks.end());

  // Decay - either all blocks or exclude a view
  if (exclude_last_view_from_decay_) {
    if (last_depth_image_.has_value() && last_depth_camera_.has_value() &&
        last_depth_T_L_C_.has_value()) {
      return tsdf_decay_integrator_.decay(
          layers_.getPtr<TsdfLayer>(),
          nvblox::DecayViewExclusionOptions(
              &last_depth_image_.value(), last_depth_T_L_C_.value(),
              last_depth_camera_.value(),
              tsdf_integrator_.max_integration_distance_m(),
              tsdf_integrator_.get_truncation_distance_m(voxel_size_m_)),
          *cuda_stream_);
    } else {
      return {};
    }
  } else {
    return tsdf_decay_integrator_.decay(layers_.getPtr<TsdfLayer>(),
                                        *cuda_stream_);
  }
}

std::vector<Index3D> Mapper::decayOccupancy() {
  // The esdf of all blocks has to be updated after decay
  const std::vector<Index3D> all_blocks =
      layers_.get<OccupancyLayer>().getAllBlockIndices();
  esdf_blocks_to_update_.insert(all_blocks.begin(), all_blocks.end());

  // Decay - either all blocks or exclude a view
  if (exclude_last_view_from_decay_) {
    if (last_depth_image_ && last_depth_camera_ && last_depth_T_L_C_) {
      return occupancy_decay_integrator_.decay(
          layers_.getPtr<OccupancyLayer>(), std::nullopt,
          nvblox::DecayViewExclusionOptions(
              &last_depth_image_.value(), last_depth_T_L_C_.value(),
              last_depth_camera_.value(),
              occupancy_integrator_.max_integration_distance_m(),
              occupancy_integrator_.get_truncation_distance_m(voxel_size_m_)),
          *cuda_stream_);
    } else {
      return {};
    }
  } else {
    return occupancy_decay_integrator_.decay(layers_.getPtr<OccupancyLayer>(),
                                             *cuda_stream_);
  }
}

std::vector<Index3D> Mapper::updateFreespace(Time update_time_ms) {
  CHECK(projective_layer_type_ == ProjectiveLayerType::kTsdf)
      << "Currently, the freespace can only be updated using a Tsdf layer.";

  // Convert the set of Freespace indices needing an update to a vector
  std::vector<Index3D> freespace_blocks_to_update_vector(
      freespace_blocks_to_update_.begin(), freespace_blocks_to_update_.end());

  // Call the integrator.
  freespace_integrator_.updateFreespaceLayer(
      freespace_blocks_to_update_vector, update_time_ms,
      layers_.get<TsdfLayer>(), layers_.getPtr<FreespaceLayer>());

  // Mark blocks as updated
  freespace_blocks_to_update_.clear();

  return freespace_blocks_to_update_vector;
}

std::shared_ptr<const SerializedMesh> Mapper::createSerializedMesh(
    const std::vector<Index3D>& mesh_blocks_to_serialize,
    const std::optional<Transform>& maybe_T_L_C) {
  mesh_streamer_.markIndicesCandidates(mesh_blocks_to_serialize);

  // Measure tick rate of requests to determine how many bytes of mesh we should
  // stream. clamp measurement to avoid instabilities when there are few rate
  // measurements.
  timing::Rates::tick("mapper/stream_mesh");
  constexpr float kMinRateHz = 1.F;
  constexpr float kMaxRateHz = 100.F;
  const float measured_mesh_rate_hz = std::max(
      kMinRateHz,
      std::min(kMaxRateHz, timing::Rates::getMeanRateHz("mapper/stream_mesh")));

  const float measured_mesh_update_period_s = 1.0f / measured_mesh_rate_hz;
  const float megabits_per_update =
      mesh_bandwidth_limit_mbps_ * measured_mesh_update_period_s;
  constexpr float kMegabitsToBytes = 1e6f / 8.0f;

  // If requested bandwidth is negative, we stream as much as we can
  const size_t num_bytes_to_stream =
      (mesh_bandwidth_limit_mbps_ < 0.F)
          ? std::numeric_limits<size_t>::max()
          : static_cast<size_t>(megabits_per_update * kMegabitsToBytes);

  std::optional<Vector3f> exclusion_center;
  if (maybe_T_L_C.has_value()) {
    exclusion_center = maybe_T_L_C.value().translation();
  }

  return mesh_streamer_.getNBytesOfSerializedMeshBlocks(
      num_bytes_to_stream, layers_.get<MeshLayer>(), exclusion_center,
      *cuda_stream_);
}

std::shared_ptr<const SerializedMesh> Mapper::updateMesh(
    const std::optional<Transform>& maybe_T_L_C) {
  // Mesh is only updated for Tsdf layers (not for occupancy)
  if (projective_layer_type_ != ProjectiveLayerType::kTsdf) {
    return std::make_shared<const SerializedMesh>();
  } else {
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

    return createSerializedMesh(mesh_blocks_to_update_vector, maybe_T_L_C);
  }
}

void Mapper::updateFullMesh() {
  // Mesh is only updated for Tsdf layers (not for occupancy)
  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    mesh_integrator_.integrateBlocksGPU(
        layers_.get<TsdfLayer>(), layers_.get<TsdfLayer>().getAllBlockIndices(),
        layers_.getPtr<MeshLayer>());
  }
}

std::vector<Index3D> Mapper::updateEsdf() {
  CHECK(esdf_mode_ != EsdfMode::k2D) << "Currently, we limit computation of "
                                        "the ESDF to 2d *or* 3d. Not both.";
  esdf_mode_ = EsdfMode::k3D;

  // Convert the set of EsdfBlocks needing an update to a vector
  std::vector<Index3D> esdf_blocks_to_update_vector(
      esdf_blocks_to_update_.begin(), esdf_blocks_to_update_.end());

  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    if (ignore_esdf_sites_in_freespace_) {
      // Passing a freespace layer to the integrator for checking if
      // candidate esdf sites fall into freespace
      esdf_integrator_.integrateBlocks(
          layers_.get<TsdfLayer>(), layers_.get<FreespaceLayer>(),
          esdf_blocks_to_update_vector, layers_.getPtr<EsdfLayer>());
    } else {
      esdf_integrator_.integrateBlocks(layers_.get<TsdfLayer>(),
                                       esdf_blocks_to_update_vector,
                                       layers_.getPtr<EsdfLayer>());
    }
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    esdf_integrator_.integrateBlocks(layers_.get<OccupancyLayer>(),
                                     esdf_blocks_to_update_vector,
                                     layers_.getPtr<EsdfLayer>());
  }

  // Mark blocks as updated
  esdf_blocks_to_update_.clear();

  return esdf_blocks_to_update_vector;
}

void Mapper::updateFullEsdf() {
  CHECK(esdf_mode_ != EsdfMode::k2D) << "Currently, we limit computation of "
                                        "the ESDF to 2d *or* 3d. Not both.";
  esdf_mode_ = EsdfMode::k3D;

  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    if (ignore_esdf_sites_in_freespace_) {
      // Passing a freespace layer to the integrator for checking if
      // candidate esdf sites fall into freespace
      esdf_integrator_.integrateLayer(layers_.get<TsdfLayer>(),
                                      layers_.get<FreespaceLayer>(),
                                      layers_.getPtr<EsdfLayer>());
    } else {
      esdf_integrator_.integrateLayer(layers_.get<TsdfLayer>(),
                                      layers_.getPtr<EsdfLayer>());
    }
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    esdf_integrator_.integrateLayer(layers_.get<OccupancyLayer>(),
                                    layers_.getPtr<EsdfLayer>());
  }
}

std::vector<Index3D> Mapper::updateEsdfSlice() {
  CHECK(esdf_mode_ != EsdfMode::k3D) << "Currently, we limit computation of "
                                        "the ESDF to 2d *or* 3d. Not both.";
  esdf_mode_ = EsdfMode::k2D;

  // Convert the set of EsdfBlocks needing an update to a vector
  std::vector<Index3D> esdf_blocks_to_update_vector(
      esdf_blocks_to_update_.begin(), esdf_blocks_to_update_.end());

  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    if (ignore_esdf_sites_in_freespace_) {
      // Passing a freespace layer to the integrator for checking if
      // candidate esdf sites fall into freespace
      esdf_integrator_.integrateSlice(
          layers_.get<TsdfLayer>(), layers_.get<FreespaceLayer>(),
          esdf_blocks_to_update_vector, esdf_slice_min_height_,
          esdf_slice_max_height_, esdf_slice_height_,
          layers_.getPtr<EsdfLayer>());
    } else {
      esdf_integrator_.integrateSlice(
          layers_.get<TsdfLayer>(), esdf_blocks_to_update_vector,
          esdf_slice_min_height_, esdf_slice_max_height_, esdf_slice_height_,
          layers_.getPtr<EsdfLayer>());
    }
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    esdf_integrator_.integrateSlice(
        layers_.get<OccupancyLayer>(), esdf_blocks_to_update_vector,
        esdf_slice_min_height_, esdf_slice_max_height_, esdf_slice_height_,
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
    block_indices_for_deletion = getBlocksOutsideRadius(
        layers_.get<TsdfLayer>().getAllBlockIndices(),
        layers_.get<TsdfLayer>().block_size(), center, radius);
    layers_.getPtr<TsdfLayer>()->clearBlocks(block_indices_for_deletion);
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    block_indices_for_deletion = getBlocksOutsideRadius(
        layers_.get<OccupancyLayer>().getAllBlockIndices(),
        layers_.get<OccupancyLayer>().block_size(), center, radius);
    layers_.getPtr<OccupancyLayer>()->clearBlocks(block_indices_for_deletion);
  }

  if (esdf_mode_ == EsdfMode::k2D) {
    // In 2D mode, all esdf blocks have a z-index of esdf_slice_height().
    // Therefore, we have to find the esdf block indices outside of the radius
    // separately from the other 3D indices.
    // NOTE(remos): The esdf_blocks_to_update set holds full 3D indices which
    // are converted to 2D inside the esdf integrator.
    std::vector<Index3D> esdf_block_indices_for_deletion =
        getBlocksOutsideRadius(layers_.get<EsdfLayer>().getAllBlockIndices(),
                               layers_.get<EsdfLayer>().block_size(), center,
                               radius);
    layers_.getPtr<EsdfLayer>()->clearBlocks(esdf_block_indices_for_deletion);
  } else {
    layers_.getPtr<EsdfLayer>()->clearBlocks(block_indices_for_deletion);
  }

  for (const Index3D& idx : block_indices_for_deletion) {
    mesh_blocks_to_update_.erase(idx);
    freespace_blocks_to_update_.erase(idx);
    esdf_blocks_to_update_.erase(idx);
  }
  layers_.getPtr<MeshLayer>()->clearBlocks(block_indices_for_deletion);
  layers_.getPtr<FreespaceLayer>()->clearBlocks(block_indices_for_deletion);
  layers_.getPtr<ColorLayer>()->clearBlocks(block_indices_for_deletion);
  return block_indices_for_deletion;
}

void Mapper::markUnobservedTsdfFreeInsideRadius(const Vector3f& center,
                                                float radius) {
  CHECK_GT(radius, 0.0f);
  std::vector<Index3D> updated_blocks;
  if (projective_layer_type_ == ProjectiveLayerType::kTsdf) {
    tsdf_integrator_.markUnobservedFreeInsideRadius(
        center, radius, layers_.getPtr<TsdfLayer>(), &updated_blocks);
  } else if (projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    occupancy_integrator_.markUnobservedFreeInsideRadius(
        center, radius, layers_.getPtr<OccupancyLayer>(), &updated_blocks);
  }
  mesh_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
  freespace_blocks_to_update_.insert(updated_blocks.begin(),
                                     updated_blocks.end());
  esdf_blocks_to_update_.insert(updated_blocks.begin(), updated_blocks.end());
}

bool Mapper::saveLayerCake(const std::string& filename) const {
  return io::writeLayerCakeToFile(filename, layers_, *cuda_stream_);
}

bool Mapper::saveLayerCake(const char* filename) const {
  return saveLayerCake(std::string(filename));
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

bool Mapper::saveMeshAsPly(const std::string& filepath) const {
  return io::outputMeshLayerToPly(mesh_layer(), filepath);
}

bool Mapper::saveEsdfAsPly(const std::string& filename) const {
  return io::outputVoxelLayerToPly(esdf_layer(), filename);
}

bool Mapper::saveTsdfAsPly(const std::string& filename) const {
  return io::outputVoxelLayerToPly(tsdf_layer(), filename);
}

bool Mapper::saveOccupancyAsPly(const std::string& filename) const {
  return io::outputVoxelLayerToPly(occupancy_layer(), filename);
}

bool Mapper::saveFreespaceAsPly(const std::string& filename) const {
  return io::outputVoxelLayerToPly(freespace_layer(), filename);
}

parameters::ParameterTreeNode Mapper::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name = (name_remap.empty()) ? "mapper" : name_remap;
  // Enum conversion functions.
  std::function<std::string(const MemoryType&)> memory_type_to_string =
      [](const MemoryType& m) { return toString(m); };
  std::function<std::string(const ProjectiveLayerType&)>
      projective_layer_type_to_string =
          [](const ProjectiveLayerType& l) { return toString(l); };
  std::function<std::string(const EsdfMode&)> esdf_mode_to_string =
      [](const EsdfMode& m) { return toString(m); };
  return ParameterTreeNode(
      name,
      {ParameterTreeNode("voxel_size_m", voxel_size_m_),
       ParameterTreeNode("memory_type", memory_type_, memory_type_to_string),
       ParameterTreeNode("projective_layer_type", projective_layer_type_,
                         projective_layer_type_to_string),
       ParameterTreeNode("esdf_mode", esdf_mode_, esdf_mode_to_string),
       ParameterTreeNode("do_depth_preprocessing", do_depth_preprocessing_),
       ParameterTreeNode("depth_preprocessing_num_dilations",
                         depth_preprocessing_num_dilations_),
       ParameterTreeNode("mesh_bandwidth_limit_mbps",
                         mesh_bandwidth_limit_mbps_),
       ParameterTreeNode("esdf_slice_min_height", esdf_slice_min_height_),
       ParameterTreeNode("esdf_slice_max_height", esdf_slice_max_height_),
       ParameterTreeNode("esdf_slice_height", esdf_slice_height_),
       ParameterTreeNode("exclude_last_view_from_decay",
                         exclude_last_view_from_decay_),
       ParameterTreeNode("ignore_esdf_sites_in_freespace",
                         ignore_esdf_sites_in_freespace_),
       tsdf_integrator_.getParameterTree("camera_tsdf_integrator"),
       lidar_tsdf_integrator_.getParameterTree("lidar_tsdf_integrator"),
       color_integrator_.getParameterTree(),
       occupancy_integrator_.getParameterTree("camera_occupancy_integrator"),
       lidar_occupancy_integrator_.getParameterTree(
           "lidar_occupancy_integrator"),
       esdf_integrator_.getParameterTree(),
       mesh_integrator_.getParameterTree(),
       mesh_streamer_.getParameterTree(),
       occupancy_decay_integrator_.getParameterTree(),
       tsdf_decay_integrator_.getParameterTree(),
       freespace_integrator_.getParameterTree()});
}

std::string Mapper::getParametersAsString() const {
  return parameterTreeToString(getParameterTree());
}

}  // namespace nvblox
