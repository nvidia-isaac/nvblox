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
#pragma once

#include <glog/logging.h>

#include <memory>
#include <string>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/datasets/3dmatch.h"
#include "nvblox/datasets/cusfm_data.h"
#include "nvblox/datasets/data_loader_interface.h"
#include "nvblox/datasets/lidarply_loader.h"
#include "nvblox/datasets/redwood.h"
#include "nvblox/datasets/replica.h"
#include "nvblox/dynamics/dynamics_detection.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/projective_appearance_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/io/image_io.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/layer_cake.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/mapper/multi_mapper.h"
#include "nvblox/mesh/mesh_block.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/rays/sphere_tracer.h"
#include "nvblox/sensors/lidar.h"

namespace nvblox {

template <typename SensorType, typename SensorDataType>
class Fuser {
 public:
  Fuser() = default;
  Fuser(std::unique_ptr<datasets::DataLoaderInterface<
            SensorType, SensorDataType>>&& data_loader,
        bool init_from_gflags);

  void initFromGflags();
  // Runs an experiment
  int run();

  // Integrate a frame from the dataset
  datasets::DataLoadResult integrateFrame(const int frame_number);
  void integrateFrames();

  // Write a dynamic overlay image to disk.
  bool outputDynamicOverlayImage(int frame_number);
  // Output a pointcloud tsdf as PLY file.
  bool outputTsdfPointcloudPly();
  // Output a pointcloud occupancy as PLY file.
  bool outputOccupancyPointcloudPly();
  // Output a pointcloud freespace as PLY file.
  bool outputFreespacePointcloudPly();
  // Output a pointcloud ESDF as PLY file.
  bool outputESDFPointcloudPly();
  // Output a file with the mesh.
  bool outputColorMeshPly();
  // Output a file with the ground-aligned mesh.
  bool outputGroundAlignedMeshPly();
  // Output timings to a file
  bool outputTimingsToFile();
  // Output the serialized map to a file
  bool outputMapToFile();
  // Output the ground plane data (normal and height) to a file
  bool outputGroundPlaneToFile();

  // Get access to the underlying mappers.
  std::shared_ptr<Mapper> static_mapper();
  std::shared_ptr<MultiMapper> multi_mapper();

  // Set the multi mapper.
  void setMultiMapper(const std::shared_ptr<MultiMapper>& multi_mapper);

  // Getters for the loaded data and generated mesh.
  std::shared_ptr<const ColorImage> getColorFrame() const;
  std::shared_ptr<const SensorDataType> getSensorData() const;
  std::shared_ptr<const SensorType> getSensor() const;
  std::shared_ptr<const Transform> getSensorPose() const;
  std::shared_ptr<const Camera> getColorCamera() const;
  std::shared_ptr<const Transform> getColorCameraPose() const;
  std::shared_ptr<SerializedColorMeshLayer> getSerializedColorMesh() const;

  // CUDA stream shared between the mapper and the renderer.
  std::shared_ptr<CudaStreamOwning> cuda_stream_ =
      std::make_shared<CudaStreamOwning>();

  // MultiMapper - Contains two mappers
  std::shared_ptr<MultiMapper> multi_mapper_;

  // MultiMapper params
  float voxel_size_m_ = 0.05;
  MappingType mapping_type_ = MappingType::kStaticTsdf;
  EsdfMode esdf_mode_ = EsdfMode::k3D;

  // Dataset settings.
  int num_frames_to_integrate_ = std::numeric_limits<int>::max();
  std::unique_ptr<datasets::DataLoaderInterface<SensorType, SensorDataType>>
      data_loader_;

  // Temporal subsampling params
  int projective_frame_subsampling_ = 1;
  int color_frame_subsampling_ = 1;
  int mesh_frame_subsampling_ = 1;
  int esdf_frame_subsampling_ = 1;

  // Param for dynamics
  nvblox::Time frame_period_ms_{33};  // 30 Hz

  // LiDAR motion compensation
  bool use_lidar_motion_compensation_ = true;

  // Output paths
  std::string timing_output_path_;
  std::string tsdf_output_path_;
  std::string esdf_output_path_;
  std::string occupancy_output_path_;
  std::string freespace_output_path_;
  std::string mesh_output_path_;
  std::string ground_aligned_mesh_output_path_;
  std::string map_output_path_;
  std::string dynamic_overlay_path_;
  std::string ground_plane_output_path_;

  // Buffers for the loaded data and generated mesh.
  std::shared_ptr<Transform> T_L_S_ = std::make_shared<Transform>();
  std::shared_ptr<SensorType> sensor_ = std::make_shared<SensorType>();
  std::shared_ptr<Transform> T_L_C_ = std::make_shared<Transform>();
  std::shared_ptr<Camera> color_camera_ = std::make_shared<Camera>();
  std::shared_ptr<SensorDataType> sensor_data_ =
      std::make_shared<SensorDataType>(MemoryType::kDevice);
  std::shared_ptr<ColorImage> color_frame_ =
      std::make_shared<ColorImage>(MemoryType::kDevice);
  std::shared_ptr<Time> frame_timestamp_ms_from_dataset_ =
      std::make_shared<Time>();
  std::shared_ptr<Transform> T_L_S_scanEnd_ = std::make_shared<Transform>();
  std::shared_ptr<Time> scan_duration_ms_ = std::make_shared<Time>();
  std::shared_ptr<SerializedColorMeshLayer> serialized_color_mesh_;
};

// Type aliases for common Fuser instantiations
using CameraFuser = Fuser<Camera, DepthImage>;
using LidarFuser = Fuser<Lidar, Pointcloud>;

// Factory functions that create a Fuser configured for a specific dataset.
// These are declared here (in the executables layer) because they depend on
// both the dataset loaders and the Fuser class.
namespace datasets {
namespace threedmatch {
std::unique_ptr<CameraFuser> createFuser(const std::string base_path,
                                         const int seq_id,
                                         bool init_from_gflags = true);
}  // namespace threedmatch
namespace redwood {
std::unique_ptr<CameraFuser> createFuser(const std::string base_path,
                                         bool init_from_gflags = true);
}  // namespace redwood
namespace replica {
std::unique_ptr<CameraFuser> createFuser(const std::string base_path,
                                         bool init_from_gflags = true);
}  // namespace replica
namespace lidarply {
std::unique_ptr<LidarFuser> createFuser(const std::string base_path,
                                        const int seq_id,
                                        bool init_from_gflags = true);
}  // namespace lidarply
namespace cusfm_data {
std::unique_ptr<CameraFuser> createFuser(const std::string& color_image_dir,
                                         const std::string& depth_image_dir,
                                         const std::string& frames_meta_file,
                                         bool init_from_gflags,
                                         bool fit_to_z_plane = false,
                                         const std::string& output_dir = "");
}  // namespace cusfm_data
}  // namespace datasets

}  //  namespace nvblox
