/*
Copyright 2025 NVIDIA CORPORATION

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
#include "nvblox/fuser/fuser.h"

#include <gflags/gflags.h>
#include "nvblox/fuser/fuser_visualizer.h"
#include "nvblox/gflags_param_loading/fuser_params_from_gflags.h"
#include "nvblox/gflags_param_loading/mapper_params_from_gflags.h"

#include "nvblox/core/parameter_tree.h"
#include "nvblox/experimental/ground_plane/ground_plane_estimator.h"
#include "nvblox/geometry/plane.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/utils/rates.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

DEFINE_bool(visualization, true,
            "Show a real-time visualization window during data integration. "
            "Has no effect when nvblox is built without BUILD_RENDERER.");

constexpr float kDefaultDynamicIntegrationDistanceM = 4.0f;
DEFINE_double(dynamic_integrator_max_integration_distance_m,
              kDefaultDynamicIntegrationDistanceM,
              "Maximum distance (in meters) from the camera at which to "
              "integrate data into the dynamic occupancy grid.");

template <typename SensorType, typename SensorDataType>
Fuser<SensorType, SensorDataType>::Fuser(
    std::unique_ptr<datasets::DataLoaderInterface<SensorType, SensorDataType>>&&
        data_loader,
    bool init_from_gflags)
    : data_loader_(std::move(data_loader)) {
  if (init_from_gflags) {
    initFromGflags();
  }
};

template <typename SensorType, typename SensorDataType>
void Fuser<SensorType, SensorDataType>::initFromGflags() {
  // Get the params needed to create the multi_mapper
  get_global_params_from_gflags(&voxel_size_m_, &mapping_type_, &esdf_mode_);

  // Create the multi mapper
  // NOTE(remos): Mesh integration is not implemented for occupancy layers.
  multi_mapper_ =
      std::make_shared<MultiMapper>(voxel_size_m_, mapping_type_, esdf_mode_,
                                    MemoryType::kDevice, cuda_stream_);

  // Init fuser params
  set_fuser_params_from_gflags(this);

  // Init mapper params (for the two mapper held by the multi mapper)
  MapperParams mapper_params = get_mapper_params_from_gflags();
  multi_mapper_->setMultiMapperParams(get_multi_mapper_params_from_gflags());

  // Set the same params for both mappers for now
  multi_mapper_->setMapperParams(mapper_params, mapper_params);

  // NOTE(remos): We default the integration distance for the dynamic mapper to
  //              4 m to limit the computation time of the depth integration.
  LOG(INFO) << "Setting dynamic occupancy max integration distance to "
            << FLAGS_dynamic_integrator_max_integration_distance_m << " m.";
  multi_mapper_.get()
      ->foreground_mapper()
      ->occupancy_integrator()
      .max_integration_distance_m(
          FLAGS_dynamic_integrator_max_integration_distance_m);

  // Dump parameters to the console before mapping starts
  LOG(INFO) << "\n\n-------------\nnvblox mapper parameters\n-------------\n"
            << multi_mapper_->getParametersAsString() << "-------------\n\n";
};

template <typename SensorType, typename SensorDataType>
int Fuser<SensorType, SensorDataType>::run() {
  // Just check that data loader we got is valid.
  if (!data_loader_ || !data_loader_->setup_success()) {
    LOG(FATAL) << "DataLoader was not set up successfully.";
  }

  // Integrate all the data
  integrateFrames();

  if (!occupancy_output_path_.empty()) {
    if (mapping_type_ == MappingType::kStaticOccupancy) {
      LOG(INFO) << "Outputting occupancy pointcloud ply file to "
                << occupancy_output_path_;
      outputOccupancyPointcloudPly();
    } else {
      LOG(ERROR) << "Occupancy pointcloud can not be stored to "
                 << occupancy_output_path_
                 << " because occupancy wasn't selected for static mapping.";
    }
  }

  if (!tsdf_output_path_.empty()) {
    if (!isStaticOccupancy(mapping_type_)) {
      LOG(INFO) << "Outputting tsdf pointcloud ply file to "
                << tsdf_output_path_;
      outputTsdfPointcloudPly();
    } else {
      LOG(ERROR) << "TSDF pointcloud can not be stored to " << tsdf_output_path_
                 << " because tsdf wasn't selected for static mapping.";
    }
  }

  if (!mesh_output_path_.empty() || !ground_aligned_mesh_output_path_.empty()) {
    LOG(INFO) << "Generating the mesh.";
    multi_mapper_->updateColorMesh();
    if (!mesh_output_path_.empty()) {
      LOG(INFO) << "Outputting mesh ply file to " << mesh_output_path_;
      outputColorMeshPly();
    }
    if (!ground_aligned_mesh_output_path_.empty()) {
      outputGroundAlignedMeshPly();
    }
  }

  if (!esdf_output_path_.empty()) {
    LOG(INFO) << "Generating the ESDF.";
    multi_mapper_->updateEsdf();
    LOG(INFO) << "Outputting ESDF pointcloud ply file to " << esdf_output_path_;
    outputESDFPointcloudPly();
  }

  if (!freespace_output_path_.empty()) {
    if (isDynamicMapping(mapping_type_)) {
      LOG(INFO) << "Outputting freespace pointcloud ply file to "
                << freespace_output_path_;
      outputFreespacePointcloudPly();
    } else {
      LOG(ERROR) << "Freespace pointcloud can not be stored to "
                 << freespace_output_path_
                 << " because kDynamic was not selected as mapping type.";
    }
  }

  if (!map_output_path_.empty()) {
    LOG(INFO) << "Outputting the serialized map to " << map_output_path_;
    outputMapToFile();
  }

  if (!ground_plane_output_path_.empty()) {
    LOG(INFO) << "Outputting the ground plane data to "
              << ground_plane_output_path_;
    outputGroundPlaneToFile();
  }

  LOG(INFO) << nvblox::timing::Timing::Print() << "\n";
  LOG(INFO) << nvblox::timing::Rates::Print() << "\n";

  if (!timing_output_path_.empty()) {
    LOG(INFO) << "Writing timings to file.";
    outputTimingsToFile();
  }

  return 0;
}

template <typename SensorType, typename SensorDataType>
void Fuser<SensorType, SensorDataType>::integrateFrames() {
  FuserVisualizer visualizer;
  if (FLAGS_visualization) {
    visualizer.init("nvblox", cuda_stream_);
  }

  int frame_number = 0;
  LOG(INFO) << "Integrating " << num_frames_to_integrate_ << " frames...";
  while (frame_number < num_frames_to_integrate_) {
    if (!visualizer.isPaused()) {
      if (integrateFrame(frame_number++) ==
          datasets::DataLoadResult::kNoMoreData) {
        break;
      }
      timing::mark("Frame " + std::to_string(frame_number - 1), Color::Red());
      std::cout << "." << std::flush;

      if constexpr (std::is_same_v<SensorDataType, DepthImage>) {
        // Update and render the msh
        multi_mapper_->background_mapper()->updateFlatColorMesh();
        visualizer.updateMesh(
            multi_mapper_->background_mapper()->flat_color_mesh(),
            *color_camera_, T_L_C_->inverse(), *color_frame_, *sensor_data_);
      }
    }
    if (!visualizer.renderAndPoll()) {
      LOG(INFO) << "Visualization window closed — stopping integration.";
      break;
    }
  }
  LOG(INFO) << "Ran out of data at frame: " << frame_number - 1;
}

template <typename SensorType, typename SensorDataType>
datasets::DataLoadResult Fuser<SensorType, SensorDataType>::integrateFrame(
    const int frame_number) {
  timing::Rates::tick("fuser/integrate_frame");
  timing::Timer timer_file("fuser/file_loading");

  // Load data - with optional color frame, timestamp, and motion compensation
  // data
  datasets::DataLoadResult load_result = data_loader_->loadNext(
      sensor_data_.get(), T_L_S_.get(), sensor_.get(),
      data_loader_->provides_color() ? color_frame_.get() : nullptr,
      data_loader_->provides_color() ? T_L_C_.get() : nullptr,
      data_loader_->provides_color() ? color_camera_.get() : nullptr,
      data_loader_->provides_frame_timestamps()
          ? frame_timestamp_ms_from_dataset_.get()
          : nullptr,
      data_loader_->provides_lidar_scan_data() ? T_L_S_scanEnd_.get() : nullptr,
      data_loader_->provides_lidar_scan_data() ? scan_duration_ms_.get()
                                               : nullptr);
  timer_file.Stop();

  // We couldn't load this data frame.
  if ((load_result == datasets::DataLoadResult::kBadFrame) ||
      (load_result == datasets::DataLoadResult::kNoMoreData)) {
    return load_result;
  }

  // Depth integration
  timing::Timer per_frame_timer("fuser/time_per_frame");
  if ((frame_number + 1) % projective_frame_subsampling_ == 0) {
    timing::Timer timer_integrate("fuser/integrate_depth");
    timing::Rates::tick("fuser/integrate_depth");

    // Use frame timestamp from the dataset if available, otherwise use the
    // frame period parameter.
    Time frame_timestamp_ms = data_loader_->provides_frame_timestamps()
                                  ? *frame_timestamp_ms_from_dataset_
                                  : Time(frame_number * frame_period_ms_);

    // Do the actual depth integration
    if constexpr (std::is_same_v<SensorDataType, Pointcloud>) {
      if (data_loader_->provides_lidar_scan_data()) {
        // Integrate pointcloud including lidar scan data (for
        // optional lidar motion compensation).
        multi_mapper_->integrateDepth(
            *sensor_data_, *T_L_S_, *sensor_, use_lidar_motion_compensation_,
            *T_L_S_scanEnd_, *scan_duration_ms_, frame_timestamp_ms);
      } else {
        // Integrate pointcloud without lidar scan data.
        // Without lidar scan data, we cannot use lidar motion compensation.
        CHECK(!use_lidar_motion_compensation_);
        multi_mapper_->integrateDepth(
            *sensor_data_, *T_L_S_, *sensor_, use_lidar_motion_compensation_,
            std::nullopt, std::nullopt, frame_timestamp_ms);
      }
    } else {
      // Integrate depth image.
      multi_mapper_->integrateDepth(*sensor_data_, *T_L_S_, *sensor_,
                                    frame_timestamp_ms);
    }
    timer_integrate.Stop();

    // Store the dynamic mask if required
    if (isDynamicMapping(mapping_type_) && !dynamic_overlay_path_.empty()) {
      outputDynamicOverlayImage(frame_number);
    }
  }

  // Color integration (only if data loader provides color)
  if (data_loader_->provides_color() &&
      (frame_number + 1) % color_frame_subsampling_ == 0) {
    CHECK_NOTNULL(color_frame_);
    CHECK_NOTNULL(T_L_C_);
    CHECK_NOTNULL(color_camera_);
    timing::Timer timer_integrate_color("fuser/integrate_color");
    timing::Rates::tick("fuser/integrate_color");
    multi_mapper_->integrateColor(*color_frame_, *T_L_C_, *color_camera_);
    timer_integrate_color.Stop();
  }

  // Mesh update
  if (mesh_frame_subsampling_ > 0) {
    if ((frame_number + 1) % mesh_frame_subsampling_ == 0) {
      timing::Timer timer_mesh("fuser/mesh");
      timing::Rates::tick("fuser/mesh");
      multi_mapper_->updateColorMesh();
    }
  }

  // Esdf update
  if (esdf_frame_subsampling_ > 0) {
    if ((frame_number + 1) % esdf_frame_subsampling_ == 0) {
      timing::Timer timer_integrate_esdf("fuser/integrate_esdf");
      timing::Rates::tick("fuser/integrate_esdf");
      multi_mapper_->updateEsdf();
      timer_integrate_esdf.Stop();
    }
  }

  per_frame_timer.Stop();

  return load_result;
}

template <typename SensorType, typename SensorDataType>
std::shared_ptr<Mapper> Fuser<SensorType, SensorDataType>::static_mapper() {
  return multi_mapper_.get()->background_mapper();
}

template <typename SensorType, typename SensorDataType>
std::shared_ptr<MultiMapper> Fuser<SensorType, SensorDataType>::multi_mapper() {
  return multi_mapper_;
}

template <typename SensorType, typename SensorDataType>
void Fuser<SensorType, SensorDataType>::setMultiMapper(
    const std::shared_ptr<MultiMapper>& multi_mapper) {
  multi_mapper_ = multi_mapper;
}

template <typename SensorType, typename SensorDataType>
bool Fuser<SensorType, SensorDataType>::outputDynamicOverlayImage(
    int frame_number) {
  timing::Timer timer_write("fuser/dynamic_mask/write");
  std::string full_path = dynamic_overlay_path_ + "/overlay_" +
                          std::to_string(frame_number) + ".png";
  return io::writeToPng(full_path,
                        multi_mapper_->getLastDynamicFrameMaskOverlay());
}

template <typename SensorType, typename SensorDataType>
bool Fuser<SensorType, SensorDataType>::outputTsdfPointcloudPly() {
  timing::Timer timer_write("fuser/tsdf/write");
  return static_mapper()->saveTsdfAsPly(tsdf_output_path_);
}

template <typename SensorType, typename SensorDataType>
bool Fuser<SensorType, SensorDataType>::outputOccupancyPointcloudPly() {
  timing::Timer timer_write("fuser/occupancy/write");
  return static_mapper()->saveOccupancyAsPly(occupancy_output_path_);
}

template <typename SensorType, typename SensorDataType>
bool Fuser<SensorType, SensorDataType>::outputFreespacePointcloudPly() {
  timing::Timer timer_write("fuser/freespace/write");
  return static_mapper()->saveFreespaceAsPly(freespace_output_path_);
}

template <typename SensorType, typename SensorDataType>
bool Fuser<SensorType, SensorDataType>::outputESDFPointcloudPly() {
  timing::Timer timer_write("fuser/esdf/write");
  return static_mapper()->saveEsdfAsPly(esdf_output_path_);
}

template <typename SensorType, typename SensorDataType>
bool Fuser<SensorType, SensorDataType>::outputColorMeshPly() {
  timing::Timer timer_write("fuser/mesh/write");
  return static_mapper()->saveColorMeshAsPly(mesh_output_path_);
}

template <typename SensorType, typename SensorDataType>
bool Fuser<SensorType, SensorDataType>::outputGroundAlignedMeshPly() {
  timing::Timer timer_write("fuser/mesh/write_ground_aligned");

  if (!multi_mapper_->getMultiMapperParams()
           .experimental_use_ground_plane_estimation) {
    LOG(WARNING) << "Ground plane estimation is not enabled. "
                 << "Cannot output ground-aligned mesh.";
    return false;
  }

  const TsdfLayer& tsdf_layer =
      multi_mapper_->background_mapper()->tsdf_layer();
  std::optional<Plane> ground_plane =
      multi_mapper_->ground_plane_estimator().computeGroundPlane(tsdf_layer);

  if (!ground_plane.has_value()) {
    LOG(WARNING)
        << "Ground plane estimation enabled but failed to compute plane. "
        << "Skipping ground-aligned mesh output.";
    return false;
  }

  LOG(INFO) << "Outputting ground-aligned mesh to "
            << ground_aligned_mesh_output_path_;
  return io::outputColorMeshLayerToPly(static_mapper()->color_mesh_layer(),
                                       ground_aligned_mesh_output_path_,
                                       ground_plane.value());
}

template <typename SensorType, typename SensorDataType>
bool Fuser<SensorType, SensorDataType>::outputTimingsToFile() {
  LOG(INFO) << "Writing timing to: " << timing_output_path_;
  std::ofstream timing_file(timing_output_path_);
  timing_file << nvblox::timing::Timing::Print();
  timing_file.close();
  return true;
}

template <typename SensorType, typename SensorDataType>
bool Fuser<SensorType, SensorDataType>::outputMapToFile() {
  timing::Timer timer_serialize("fuser/map/write");
  return static_mapper()->saveLayerCake(map_output_path_);
}

template <typename SensorType, typename SensorDataType>
bool Fuser<SensorType, SensorDataType>::outputGroundPlaneToFile() {
  timing::Timer timer_write("fuser/ground_plane/write");
  return multi_mapper_->saveGroundPlaneAsYaml(ground_plane_output_path_);
}

template <typename SensorType, typename SensorDataType>
std::shared_ptr<const ColorImage>
Fuser<SensorType, SensorDataType>::getColorFrame() const {
  return color_frame_;
}

template <typename SensorType, typename SensorDataType>
std::shared_ptr<const SensorDataType>
Fuser<SensorType, SensorDataType>::getSensorData() const {
  return sensor_data_;
}

template <typename SensorType, typename SensorDataType>
std::shared_ptr<const SensorType> Fuser<SensorType, SensorDataType>::getSensor()
    const {
  return sensor_;
}

template <typename SensorType, typename SensorDataType>
std::shared_ptr<const Transform>
Fuser<SensorType, SensorDataType>::getSensorPose() const {
  return T_L_S_;
}

template <typename SensorType, typename SensorDataType>
std::shared_ptr<const Camera>
Fuser<SensorType, SensorDataType>::getColorCamera() const {
  return color_camera_;
}

template <typename SensorType, typename SensorDataType>
std::shared_ptr<const Transform>
Fuser<SensorType, SensorDataType>::getColorCameraPose() const {
  return T_L_C_;
}

template <typename SensorType, typename SensorDataType>
std::shared_ptr<SerializedColorMeshLayer>
Fuser<SensorType, SensorDataType>::getSerializedColorMesh() const {
  multi_mapper_->background_mapper()->serializeSelectedLayers(
      LayerType::kColorMesh, kLayerStreamerUnlimitedBandwidth);
  return multi_mapper_->background_mapper()->serializedColorMeshLayer();
}

// Explicit template instantiations
template class Fuser<Camera, DepthImage>;
template class Fuser<Lidar, Pointcloud>;

// Dataset-specific Fuser factory functions
namespace datasets {
namespace threedmatch {
std::unique_ptr<CameraFuser> createFuser(const std::string base_path,
                                         const int seq_id,
                                         bool init_from_gflags) {
  auto data_loader = DataLoader::create(base_path, seq_id, false);
  if (!data_loader) {
    return std::unique_ptr<CameraFuser>();
  }
  return std::make_unique<CameraFuser>(std::move(data_loader),
                                       init_from_gflags);
}
}  // namespace threedmatch

namespace redwood {
std::unique_ptr<CameraFuser> createFuser(const std::string base_path,
                                         bool init_from_gflags) {
  auto data_loader = DataLoader::create(base_path, false);
  if (!data_loader) {
    return std::unique_ptr<CameraFuser>();
  }
  return std::make_unique<CameraFuser>(std::move(data_loader),
                                       init_from_gflags);
}
}  // namespace redwood

namespace replica {
std::unique_ptr<CameraFuser> createFuser(const std::string base_path,
                                         bool init_from_gflags) {
  auto data_loader = DataLoader::create(base_path, false);
  if (!data_loader) {
    return std::unique_ptr<CameraFuser>();
  }
  return std::make_unique<CameraFuser>(std::move(data_loader),
                                       init_from_gflags);
}
}  // namespace replica

namespace lidarply {
std::unique_ptr<LidarFuser> createFuser(const std::string base_path,
                                        const int seq_id,
                                        bool init_from_gflags) {
  auto data_loader = DataLoader::create(base_path, seq_id);
  if (!data_loader) {
    return std::unique_ptr<LidarFuser>();
  }
  return std::make_unique<LidarFuser>(std::move(data_loader), init_from_gflags);
}
}  // namespace lidarply

namespace cusfm_data {
std::unique_ptr<CameraFuser> createFuser(const std::string& color_image_dir,
                                         const std::string& depth_image_dir,
                                         const std::string& frames_meta_file,
                                         bool init_from_gflags,
                                         bool fit_to_z_plane,
                                         const std::string& output_dir) {
  auto data_loader = DataLoader::create(
      color_image_dir, depth_image_dir, frames_meta_file,
      false /* if use multithread*/, fit_to_z_plane, output_dir);
  if (!data_loader) {
    return std::unique_ptr<CameraFuser>();
  }
  return std::make_unique<CameraFuser>(std::move(data_loader),
                                       init_from_gflags);
}
}  // namespace cusfm_data
}  // namespace datasets

}  //  namespace nvblox
