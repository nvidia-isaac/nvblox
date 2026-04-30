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
#include "nvblox/datasets/lidarply_loader.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "nvblox/io/ply_parser.h"
#include "nvblox/sensors/pointcloud.h"

namespace nvblox {
namespace datasets {
namespace lidarply {
namespace internal {

// Reads the next non-empty, non-comment line from a file stream and returns as
// istringstream. Returns default-constructed istringstream if EOF is reached.
std::istringstream readNextNonCommentedLine(std::ifstream& fin) {
  std::string line;
  while (std::getline(fin, line)) {
    if (!line.empty() && line[0] != '#') {
      return std::istringstream(line);
    }
  }
  return std::istringstream{};
}

bool parseLidarFromFile(const std::string& filename, Lidar* lidar) {
  CHECK_NOTNULL(lidar);

  std::ifstream fin(filename);
  if (!fin.is_open()) {
    return false;
  }

  int num_azimuth_divisions = 0;
  int num_elevation_divisions = 0;
  float min_valid_range_m = 0.0f;
  float min_angle_below_zero_elevation_rad = 0.0f;
  float max_angle_above_zero_elevation_rad = 0.0f;

  // Skip comment lines (starting with #) and read parameters.
  // We are expecting the following format:
  // - line 1: num_azimuth_divisions num_elevation_divisions
  // - line 2: min_valid_range_m
  // - line 3: min_angle_below_zero_elevation_rad
  // max_angle_above_zero_elevation_rad

  // Parse line 1
  if (!(readNextNonCommentedLine(fin) >> num_azimuth_divisions >>
        num_elevation_divisions)) {
    LOG(ERROR) << "Failed to parse lidar dimensions from: " << filename;
    return false;
  }

  // Parse line 2
  if (!(readNextNonCommentedLine(fin) >> min_valid_range_m)) {
    LOG(ERROR) << "Failed to parse lidar min range from: " << filename;
    return false;
  }

  // Parse line 3
  if (!(readNextNonCommentedLine(fin) >> min_angle_below_zero_elevation_rad >>
        max_angle_above_zero_elevation_rad)) {
    LOG(ERROR) << "Failed to parse lidar elevation angles from: " << filename;
    return false;
  }

  fin.close();

  // Create the Lidar object
  *lidar = Lidar(num_azimuth_divisions, num_elevation_divisions,
                 min_valid_range_m, min_angle_below_zero_elevation_rad,
                 max_angle_above_zero_elevation_rad);

  LOG(INFO) << "Read lidar params: " << num_azimuth_divisions << "x"
            << num_elevation_divisions << ", range: " << min_valid_range_m
            << ", min_angle_below: " << min_angle_below_zero_elevation_rad
            << ", max_angle_above: " << max_angle_above_zero_elevation_rad;

  return true;
}

std::string getPathForLidarIntrinsics(const std::string& base_path) {
  return base_path + "/lidar-intrinsics.txt";
}

std::string getPathForPointcloud(const std::string& base_path, const int seq_id,
                                 const int frame_id) {
  std::stringstream ss;
  ss << base_path << "/seq-" << std::setfill('0') << std::setw(2) << seq_id
     << "/frame-" << std::setw(6) << frame_id << ".pointcloud.ply";
  return ss.str();
}

std::string getPathToFrameTimestampFile(const std::string& base_path,
                                        const int seq_id, const int frame_id) {
  std::stringstream ss;
  ss << base_path << "/seq-" << std::setfill('0') << std::setw(2) << seq_id
     << "/frame-" << std::setw(6) << frame_id << ".timestamp.txt";
  return ss.str();
}

bool parseTimestampFromFile(const std::string& filename, Time* timestamp_ms) {
  CHECK_NOTNULL(timestamp_ms);

  std::ifstream fin(filename);
  if (!fin.is_open()) {
    return false;
  }

  // Read the timestamp (int64 value in milliseconds with "ms" suffix)
  int64_t timestamp_ms_value = 0;
  std::string suffix;
  if (!(fin >> timestamp_ms_value >> suffix)) {
    LOG(ERROR) << "Failed to parse timestamp from: " << filename;
    return false;
  }

  // Verify the suffix
  if (suffix != "ms") {
    LOG(ERROR) << "Unexpected timestamp suffix: " << suffix
               << " (expected 'ms') in file: " << filename;
    return false;
  }

  *timestamp_ms = Time(timestamp_ms_value);

  return true;
}

}  // namespace internal

std::unique_ptr<DataLoader> DataLoader::create(const std::string& base_path,
                                               const int seq_id) {
  // Construct a dataset loader but only return it if everything worked
  auto dataset_loader = std::make_unique<DataLoader>(base_path, seq_id);
  if (dataset_loader->setup_success_) {
    return dataset_loader;
  } else {
    return std::unique_ptr<DataLoader>();
  }
}

DataLoader::DataLoader(const std::string& base_path, const int seq_id)
    : base_path_(base_path), seq_id_(seq_id) {
  // Check if the base path exists
  if (!std::filesystem::exists(base_path)) {
    LOG(WARNING) << "Tried to create a dataloader with a non-existent path: "
                 << base_path;
    setup_success_ = false;
    return;
  }

  // Try to load the lidar intrinsics from the base path
  std::string intrinsics_path = internal::getPathForLidarIntrinsics(base_path_);

  // If not found in base path, try parent directory
  if (!std::filesystem::exists(intrinsics_path)) {
    LOG(INFO) << "Lidar intrinsics file does not exist: " << intrinsics_path;
    setup_success_ = false;
    return;
  }

  if (!internal::parseLidarFromFile(intrinsics_path, &lidar_)) {
    LOG(WARNING) << "Failed to load lidar intrinsics from: " << intrinsics_path;
    setup_success_ = false;
    return;
  }
  lidar_loaded_ = true;

  LOG(INFO) << "Successfully initialized lidar pointcloud dataset loader for: "
            << base_path;
}

bool DataLoader::loadPoseAndTimestamp(int frame_num, Transform* pose_ptr,
                                      Time* time_ptr) {
  CHECK_NOTNULL(pose_ptr);
  CHECK_NOTNULL(time_ptr);

  // Find the pose file
  const std::string pose_path =
      internal::getPathForFramePose(base_path_, seq_id_, frame_num);
  if (!std::filesystem::exists(pose_path)) {
    LOG(INFO) << "Pose file does not exist: " << pose_path;
    return false;
  }

  // Parse the pose file
  if (!internal::parsePoseFromFile(pose_path, pose_ptr)) {
    LOG(ERROR) << "Failed to load parse pose from: " << pose_path;
    return false;
  }

  // Check that the loaded data doesn't contain NaNs or a faulty rotation
  // matrix
  constexpr float kRotationMatrixDetEpsilon = 1e-4;
  if (!pose_ptr->matrix().allFinite() ||
      std::abs(pose_ptr->matrix().block<3, 3>(0, 0).determinant() - 1.0f) >
          kRotationMatrixDetEpsilon) {
    LOG(WARNING) << "Bad pose data in frame " << frame_num;
    return false;
  }

  // Find the timestamp file
  const std::string timestamp_path =
      internal::getPathToFrameTimestampFile(base_path_, seq_id_, frame_num);
  if (!std::filesystem::exists(timestamp_path)) {
    LOG(INFO) << "Timestamp file does not exist: " << timestamp_path;
    return false;
  }

  // Parse the timestamp file
  if (!internal::parseTimestampFromFile(timestamp_path, time_ptr)) {
    LOG(ERROR) << "Failed to parse timestamp from: " << timestamp_path;
    return false;
  }
  return true;
}

DataLoadResult DataLoader::loadNext(Pointcloud* pointcloud_ptr,
                                    Transform* T_L_S_scanStart_ptr,
                                    Lidar* lidar_ptr, ColorImage*, Transform*,
                                    Camera*, Time* frame_timestamp_ms_ptr,
                                    Transform* T_L_S_scanEnd_ptr,
                                    Time* scan_duration_ms_ptr) {
  CHECK(setup_success_);
  CHECK_NOTNULL(pointcloud_ptr);
  CHECK_NOTNULL(T_L_S_scanStart_ptr);
  CHECK_NOTNULL(lidar_ptr);
  CHECK_NOTNULL(frame_timestamp_ms_ptr);
  CHECK_NOTNULL(T_L_S_scanEnd_ptr);
  CHECK_NOTNULL(scan_duration_ms_ptr);

  const int frame_number = frame_number_;
  ++frame_number_;

  // Load the pointcloud from PLY file
  const std::string ply_path =
      internal::getPathForPointcloud(base_path_, seq_id_, frame_number);

  if (!std::filesystem::exists(ply_path)) {
    LOG(INFO) << "Pointcloud file does not exist: " << ply_path;
    return DataLoadResult::kNoMoreData;
  }

  io::PlyParser parser(ply_path);
  if (!parser.isValid()) {
    LOG(ERROR) << "Failed to parse PLY file: " << ply_path;
    return DataLoadResult::kBadFrame;
  }

  if (!parser.toPointcloud(pointcloud_ptr, cuda_stream_)) {
    LOG(ERROR) << "Failed to convert PLY to pointcloud: " << ply_path;
    return DataLoadResult::kBadFrame;
  }

  // Load current frame's pose and timestamp
  if (!loadPoseAndTimestamp(frame_number, T_L_S_scanStart_ptr,
                            frame_timestamp_ms_ptr)) {
    return DataLoadResult::kBadFrame;
  }

  // Copy the lidar intrinsics
  *lidar_ptr = lidar_;

  // Load next frame's pose and timestamp
  // for extracting lidar scan data (sensor pose at scan end and scan duration).
  Time frame_timestamp_ms_next;
  if (!loadPoseAndTimestamp(frame_number + 1, T_L_S_scanEnd_ptr,
                            &frame_timestamp_ms_next)) {
    LOG(INFO) << "Failed to load next frame data for motion compensation";
    return DataLoadResult::kNoMoreData;
  }
  *scan_duration_ms_ptr = frame_timestamp_ms_next - *frame_timestamp_ms_ptr;

  LOG(INFO) << "Loaded frame " << frame_number << " with "
            << pointcloud_ptr->size() << " points";
  cuda_stream_.synchronize();

  return DataLoadResult::kSuccess;
}

}  // namespace lidarply
}  // namespace datasets
}  // namespace nvblox
