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

#include "nvblox/datasets/cusfm_data.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <type_traits>

#include "nvblox/utils/timing.h"

// Simple JSON parsing functionality for basic needs
// Simple JSON value class for basic parsing
class SimpleJson {
 public:
  enum Type { OBJECT, ARRAY, STRING, NUMBER, BOOLEAN, NULL_TYPE };

  Type type = NULL_TYPE;
  std::string string_value;
  double number_value = 0.0;
  bool bool_value = false;
  std::map<std::string, SimpleJson> object_value;
  std::vector<SimpleJson> array_value;

  static std::optional<SimpleJson> parse(const std::string& json_str) {
    size_t pos = 0;
    auto result = parseValue(json_str, pos, 0);
    if (!result.has_value()) {
      return std::nullopt;
    }
    // Check if we consumed the entire string (allowing trailing whitespace)
    skipWhitespace(json_str, pos);
    if (pos < json_str.length()) {
      LOG(WARNING)
          << "JSON parsing did not consume entire string, stopped at position "
          << pos;
    }
    return result;
  }

  bool contains(const std::string& key) const {
    return type == OBJECT && object_value.find(key) != object_value.end();
  }

  const SimpleJson& operator[](const std::string& key) const {
    static SimpleJson null_json;
    if (type != OBJECT) return null_json;
    auto it = object_value.find(key);
    return it != object_value.end() ? it->second : null_json;
  }

  const SimpleJson& operator[](size_t index) const {
    static SimpleJson null_json;
    if (type != ARRAY || index >= array_value.size()) return null_json;
    return array_value[index];
  }

  size_t size() const {
    if (type == ARRAY) return array_value.size();
    if (type == OBJECT) return object_value.size();
    return 0;
  }

  bool is_null() const { return type == NULL_TYPE; }

  template <typename T>
  T get() const {
    if constexpr (std::is_same_v<T, std::string>) {
      return string_value;
    } else if constexpr (std::is_same_v<T, double>) {
      return number_value;
    } else if constexpr (std::is_same_v<T, float>) {
      return static_cast<float>(number_value);
    } else if constexpr (std::is_same_v<T, int>) {
      return static_cast<int>(number_value);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return static_cast<uint32_t>(number_value);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return static_cast<uint64_t>(number_value);
    } else if constexpr (std::is_same_v<T, bool>) {
      return bool_value;
    }
    return T{};
  }

  auto begin() const { return object_value.begin(); }
  auto end() const { return object_value.end(); }

 private:
  static constexpr int32_t kMaxJsonDepth = 100;

  static std::optional<SimpleJson> parseValue(const std::string& json_str,
                                              size_t& pos, int32_t depth = 0) {
    skipWhitespace(json_str, pos);
    if (pos >= json_str.length()) {
      LOG(WARNING) << "Unexpected end of JSON string";
      return std::nullopt;
    }

    // Check recursion depth to prevent stack overflow
    if (depth >= kMaxJsonDepth) {
      LOG(ERROR) << "JSON parsing exceeded maximum depth of " << kMaxJsonDepth;
      return std::nullopt;
    }

    char c = json_str[pos];
    if (c == '{') {
      return parseObject(json_str, pos, depth + 1);
    } else if (c == '[') {
      return parseArray(json_str, pos, depth + 1);
    } else if (c == '"') {
      return parseString(json_str, pos);
    } else if (c == 't' || c == 'f') {
      return parseBoolean(json_str, pos);
    } else if (c == 'n') {
      return parseNull(json_str, pos);
    } else if (c == '-' || std::isdigit(c)) {
      return parseNumber(json_str, pos);
    }
    LOG(ERROR) << "Unexpected character '" << c << "' at position " << pos;
    return std::nullopt;
  }

  static std::optional<SimpleJson> parseObject(const std::string& json_str,
                                               size_t& pos, int32_t depth) {
    SimpleJson obj;
    obj.type = OBJECT;
    pos++;  // skip '{'
    skipWhitespace(json_str, pos);

    if (pos < json_str.length() && json_str[pos] == '}') {
      pos++;
      return obj;
    }

    while (pos < json_str.length()) {
      skipWhitespace(json_str, pos);
      if (pos >= json_str.length() || json_str[pos] != '"') {
        LOG(ERROR) << "Expected key string in object at position " << pos;
        return std::nullopt;
      }

      auto key_opt = parseString(json_str, pos);
      if (!key_opt.has_value()) return std::nullopt;

      skipWhitespace(json_str, pos);
      if (pos >= json_str.length() || json_str[pos] != ':') {
        LOG(ERROR) << "Expected ':' after object key at position " << pos;
        return std::nullopt;
      }
      pos++;  // skip ':'

      auto value_opt = parseValue(json_str, pos, depth);
      if (!value_opt.has_value()) return std::nullopt;

      obj.object_value[key_opt->string_value] = std::move(*value_opt);

      skipWhitespace(json_str, pos);
      if (pos >= json_str.length()) {
        LOG(ERROR) << "Unexpected end of JSON in object";
        return std::nullopt;
      }
      if (json_str[pos] == '}') {
        pos++;
        break;
      }
      if (json_str[pos] == ',') {
        pos++;
        continue;
      }
      LOG(ERROR) << "Expected ',' or '}' in object at position " << pos;
      return std::nullopt;
    }
    return obj;
  }

  static std::optional<SimpleJson> parseArray(const std::string& json_str,
                                              size_t& pos, int32_t depth) {
    SimpleJson arr;
    arr.type = ARRAY;
    pos++;  // skip '['
    skipWhitespace(json_str, pos);

    if (pos < json_str.length() && json_str[pos] == ']') {
      pos++;
      return arr;
    }

    while (pos < json_str.length()) {
      auto value_opt = parseValue(json_str, pos, depth);
      if (!value_opt.has_value()) return std::nullopt;

      arr.array_value.push_back(std::move(*value_opt));

      skipWhitespace(json_str, pos);
      if (pos >= json_str.length()) {
        LOG(ERROR) << "Unexpected end of JSON in array";
        return std::nullopt;
      }
      if (json_str[pos] == ']') {
        pos++;
        break;
      }
      if (json_str[pos] == ',') {
        pos++;
        continue;
      }
      LOG(ERROR) << "Expected ',' or ']' in array at position " << pos;
      return std::nullopt;
    }
    return arr;
  }

  static std::optional<SimpleJson> parseString(const std::string& json_str,
                                               size_t& pos) {
    SimpleJson str;
    str.type = STRING;
    pos++;  // skip '"'

    while (pos < json_str.length() && json_str[pos] != '"') {
      if (json_str[pos] == '\\' && pos + 1 < json_str.length()) {
        pos++;
        char escaped = json_str[pos];
        switch (escaped) {
          case 'n':
            str.string_value += '\n';
            break;
          case 't':
            str.string_value += '\t';
            break;
          case 'r':
            str.string_value += '\r';
            break;
          case '\\':
            str.string_value += '\\';
            break;
          case '"':
            str.string_value += '"';
            break;
          default:
            str.string_value += escaped;
            break;
        }
      } else {
        str.string_value += json_str[pos];
      }
      pos++;
    }
    if (pos >= json_str.length()) {
      LOG(ERROR) << "Unterminated string in JSON";
      return std::nullopt;
    }
    pos++;  // skip closing '"'
    return str;
  }

  static std::optional<SimpleJson> parseNumber(const std::string& json_str,
                                               size_t& pos) {
    SimpleJson num;
    num.type = NUMBER;

    size_t start = pos;
    // Handle optional minus sign
    if (json_str[pos] == '-') pos++;
    // Parse digits and decimal point
    while (pos < json_str.length() &&
           (std::isdigit(json_str[pos]) || json_str[pos] == '.')) {
      pos++;
    }
    // Handle scientific notation (e or E)
    if (pos < json_str.length() &&
        (json_str[pos] == 'e' || json_str[pos] == 'E')) {
      pos++;
      // Handle optional +/- after e/E
      if (pos < json_str.length() &&
          (json_str[pos] == '+' || json_str[pos] == '-')) {
        pos++;
      }
      // Parse exponent digits
      while (pos < json_str.length() && std::isdigit(json_str[pos])) {
        pos++;
      }
    }

    std::string num_str = json_str.substr(start, pos - start);
    try {
      num.number_value = std::stod(num_str);
    } catch (const std::exception&) {
      num.number_value = 0.0;
    }
    return num;
  }

  static std::optional<SimpleJson> parseBoolean(const std::string& json_str,
                                                size_t& pos) {
    SimpleJson boolean;
    boolean.type = BOOLEAN;

    if (pos + 4 <= json_str.length() && json_str.substr(pos, 4) == "true") {
      boolean.bool_value = true;
      pos += 4;
      return boolean;
    } else if (pos + 5 <= json_str.length() &&
               json_str.substr(pos, 5) == "false") {
      boolean.bool_value = false;
      pos += 5;
      return boolean;
    }
    LOG(ERROR) << "Invalid boolean value at position " << pos;
    return std::nullopt;
  }

  static std::optional<SimpleJson> parseNull(const std::string& json_str,
                                             size_t& pos) {
    SimpleJson null_val;
    null_val.type = NULL_TYPE;
    if (pos + 4 <= json_str.length() && json_str.substr(pos, 4) == "null") {
      pos += 4;
      return null_val;
    }
    LOG(ERROR) << "Invalid null value at position " << pos;
    return std::nullopt;
  }

  static void skipWhitespace(const std::string& json_str, size_t& pos) {
    while (pos < json_str.length() && std::isspace(json_str[pos])) {
      pos++;
    }
  }
};

template <typename T>
T getJsonValue(const SimpleJson& json, const std::string& key,
               const T& default_value = T{}) {
  if (json.contains(key) && !json[key].is_null()) {
    return json[key].get<T>();
  }
  return default_value;
}

template <typename T>
T getJsonValueFromPath(const SimpleJson& json,
                       const std::vector<std::string>& path,
                       const T& default_value = T{}) {
  const SimpleJson* current = &json;
  for (const auto& key : path) {
    if (!current->contains(key) || (*current)[key].is_null()) {
      return default_value;
    }
    current = &(*current)[key];
  }
  return current->get<T>();
}

std::string getPathForColorImage(
    const std::string& base_path,
    const std::vector<nvblox::datasets::cusfm_data::KeyframeMetadata>&
        keyframe_metadatas,
    const int frame_id) {
  std::string path =
      base_path + "/" + keyframe_metadatas[frame_id].color_image_path;
  VLOG(1) << "Load color image from " << path;
  return path;
}

std::string getPathForDepthImage(
    const std::string& base_path,
    const std::vector<nvblox::datasets::cusfm_data::KeyframeMetadata>&
        keyframe_metadatas,
    const int frame_id) {
  std::string path =
      base_path + "/" + keyframe_metadatas[frame_id].depth_image_path;
  VLOG(1) << "Load depth image from " << path;
  return path;
}

std::string ChangeFileExtension(const std::string& filename,
                                const std::string& new_extension) {
  // Find the last dot in the filename to identify the current extension
  size_t dot_position = filename.rfind('.');

  // If there's no dot, assume the filename has no extension
  if (dot_position == std::string::npos) {
    return filename + '.' + new_extension;
  }

  // Extract the base filename (without the current extension) and add the new
  // extension
  return filename.substr(0, dot_position + 1) + new_extension;
}

template <typename T>
T stringToNumber(const std::string& str) {
  if constexpr (std::is_same_v<T, int>) {
    return std::stoi(str);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return std::stoul(str);
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return std::stoull(str);
  } else {
    throw std::invalid_argument("Unsupported type for conversion");
  }
}

nvblox::Transform transformFromRigidTransform3dJson(const SimpleJson& json) {
  // Use double precision for parsing and intermediate calculations
  double angle_deg =
      getJsonValueFromPath<double>(json, {"axis_angle", "angle_degrees"}, 0.0);
  double qx = getJsonValueFromPath<double>(json, {"axis_angle", "x"}, 0.0);
  double qy = getJsonValueFromPath<double>(json, {"axis_angle", "y"}, 0.0);
  double qz = getJsonValueFromPath<double>(json, {"axis_angle", "z"},
                                           1.0);  // Default to unit z-axis

  double x = getJsonValueFromPath<double>(json, {"translation", "x"}, 0.0);
  double y = getJsonValueFromPath<double>(json, {"translation", "y"}, 0.0);
  double z = getJsonValueFromPath<double>(json, {"translation", "z"}, 0.0);

  // Ensure we have a valid axis for rotation
  Eigen::Vector3d axis(qx, qy, qz);
  if (axis.norm() < 1e-10) {
    axis = Eigen::Vector3d(0, 0, 1);  // Default to z-axis if no valid axis
  } else {
    axis.normalize();
  }

  // Build transform with double precision
  Eigen::AngleAxisd axis_angle(angle_deg * M_PI / 180.0, axis);
  Eigen::Matrix3d rotation_d = axis_angle.matrix();
  Eigen::Vector3d translation_d(x, y, z);

  // Convert to float-based Transform that nvblox uses
  nvblox::Transform transform;
  transform.linear() = rotation_d.cast<float>();
  transform.translation() = translation_d.cast<float>();
  return transform;
}

nvblox::Camera cameraFromMonoCalibrationParametersJson(const SimpleJson& json) {
  // projection_matrix is a row-major stored 3x4 matrix:
  // fu, 0, cu, 0
  // 0, fv, cv, 0
  // 0,  0,  1, 0

  // Safe access to projection matrix data with default values
  auto getProjectionMatrixElement = [&](int index,
                                        double default_val) -> double {
    if (json.contains("projection_matrix") &&
        json["projection_matrix"].contains("data") &&
        json["projection_matrix"]["data"].size() > static_cast<size_t>(index)) {
      return json["projection_matrix"]["data"][index].get<double>();
    }
    return default_val;
  };

  double fu = getProjectionMatrixElement(0, 1.0);  // Default focal length
  double fv = getProjectionMatrixElement(5, 1.0);  // Default focal length
  double cu = getProjectionMatrixElement(2, 0.0);  // Default principal point
  double cv = getProjectionMatrixElement(6, 0.0);  // Default principal point

  int width = getJsonValue<int>(json, "image_width", 0);
  int height = getJsonValue<int>(json, "image_height", 0);

  return nvblox::Camera(fu, fv, cu, cv, width, height);
}

std::unique_ptr<nvblox::datasets::ImageLoader<nvblox::DepthImage>>
createDepthImageLoader(
    const std::string& image_dir,
    const std::vector<nvblox::datasets::cusfm_data::KeyframeMetadata>&
        keyframe_metadatas,
    const bool multithreaded) {
  return nvblox::datasets::createImageLoader<nvblox::DepthImage>(
      std::bind(getPathForDepthImage, image_dir, keyframe_metadatas,
                std::placeholders::_1),
      multithreaded);
}

std::unique_ptr<nvblox::datasets::ImageLoader<nvblox::ColorImage>>
createColorImageLoader(
    const std::string& image_dir,
    const std::vector<nvblox::datasets::cusfm_data::KeyframeMetadata>&
        keyframe_metadatas,
    const bool multithreaded) {
  return nvblox::datasets::createImageLoader<nvblox::ColorImage>(
      std::bind(getPathForColorImage, image_dir, keyframe_metadatas,
                std::placeholders::_1),
      multithreaded);
}

bool loadKeyframeMetadataCollection(
    const std::string& color_image_dir, const std::string& depth_image_dir,
    const std::string& frames_meta_file,
    std::vector<nvblox::datasets::cusfm_data::KeyframeMetadata>*
        keyframe_metadatas,
    std::unordered_map<uint32_t, nvblox::Camera>* cameras) {
  std::ifstream ifs(frames_meta_file);
  if (!ifs.is_open()) {
    LOG(ERROR) << "Failed to open file: " << frames_meta_file;
    return false;
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  std::string json_str = buffer.str();

  auto json_opt = SimpleJson::parse(json_str);
  if (!json_opt.has_value()) {
    LOG(ERROR) << "Failed to parse JSON from file: " << frames_meta_file;
    return false;
  }
  const SimpleJson& json = *json_opt;

  bool found_depth = false;
  bool change_depth_extension_to_png = false;
  try {
    if (!json.contains("keyframes_metadata") ||
        json["keyframes_metadata"].is_null()) {
      LOG(WARNING) << "No keyframes_metadata found in JSON file";
      return true;  // Empty dataset is valid
    }

    for (size_t i = 0; i < json["keyframes_metadata"].size(); ++i) {
      const auto& metadata_j = json["keyframes_metadata"][i];
      nvblox::datasets::cusfm_data::KeyframeMetadata metadata;
      if (!metadata.fromJson(metadata_j)) {
        LOG(WARNING) << "Failed to parse keyframe metadata at index " << i
                     << ", skipping frame";
        continue;
      }

      if (!std::filesystem::exists(color_image_dir + "/" +
                                   metadata.color_image_path)) {
        continue;
      }

      if (!found_depth) {
        if (std::filesystem::exists(depth_image_dir + "/" +
                                    metadata.depth_image_path)) {
          found_depth = true;
        } else if (std::filesystem::exists(
                       depth_image_dir + "/" +
                       ChangeFileExtension(metadata.depth_image_path, "png"))) {
          found_depth = true;
          change_depth_extension_to_png = true;
          LOG(INFO) << "Found depth PNG files";
        }
      }

      if (change_depth_extension_to_png) {
        metadata.depth_image_path =
            ChangeFileExtension(metadata.depth_image_path, "png");
      }

      if (!std::filesystem::exists(depth_image_dir + "/" +
                                   metadata.depth_image_path)) {
        continue;
      }

      keyframe_metadatas->push_back(metadata);
    }

    // sort by timestamps
    std::sort(
        keyframe_metadatas->begin(), keyframe_metadatas->end(),
        [](const nvblox::datasets::cusfm_data::KeyframeMetadata& metadata1,
           const nvblox::datasets::cusfm_data::KeyframeMetadata& metadata2) {
          return metadata1.timestamp_microseconds <
                 metadata2.timestamp_microseconds;
        });
    LOG(INFO) << "Number of keyframes loaded: " << keyframe_metadatas->size();

    // Handle camera parameters safely
    if (json.contains("camera_params_id_to_camera_params") &&
        !json["camera_params_id_to_camera_params"].is_null()) {
      const auto& camera_params_map = json["camera_params_id_to_camera_params"];
      for (const auto& item : camera_params_map.object_value) {
        const std::string& key = item.first;
        const SimpleJson& value = item.second;
        if (value.contains("calibration_parameters") &&
            !value["calibration_parameters"].is_null()) {
          nvblox::Camera camera = cameraFromMonoCalibrationParametersJson(
              value["calibration_parameters"]);
          try {
            uint32_t camera_id = stringToNumber<uint32_t>(key);
            cameras->emplace(camera_id, camera);
            LOG(INFO) << "Loaded camera parameters for camera_id: "
                      << camera_id;
          } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to parse camera_params_id: " << key
                         << ", error: " << e.what();
          }
        } else {
          LOG(WARNING)
              << "Missing calibration_parameters for camera_params_id: " << key;
        }
      }
      LOG(INFO) << "Total cameras loaded: " << cameras->size();
    } else {
      LOG(WARNING) << "No camera_params_id_to_camera_params found in JSON file";
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error: " << e.what();
    return false;
  }

  return true;
}

namespace nvblox {
namespace datasets {
namespace cusfm_data {

std::unique_ptr<DataLoader> DataLoader::create(
    const std::string& color_image_dir, const std::string& depth_image_dir,
    const std::string& frames_meta_file, bool multithreaded,
    bool fit_to_z_plane, const std::string& output_dir) {
  LOG(INFO) << "Load color images from " << color_image_dir;
  LOG(INFO) << "Load depth images from " << depth_image_dir;
  LOG(INFO) << "Load frames_meta from " << frames_meta_file;

  // Construct a dataset loader but only return it if everything worked.
  std::vector<KeyframeMetadata> keyframe_metadatas;
  std::unordered_map<uint32_t, Camera> cameras;
  loadKeyframeMetadataCollection(color_image_dir, depth_image_dir,
                                 frames_meta_file, &keyframe_metadatas,
                                 &cameras);
  auto dataset_loader = std::make_unique<DataLoader>(
      color_image_dir, depth_image_dir, keyframe_metadatas, cameras,
      multithreaded, fit_to_z_plane, output_dir);
  if (dataset_loader->setup_success_) {
    return dataset_loader;
  } else {
    return std::unique_ptr<DataLoader>();
  }
}

bool KeyframeMetadata::fromJson(const SimpleJson& json) {
  // In protobuf json serialization, int and uint64 are converted to strings.
  // Use safe field access with default values

  std::string timestamp_str =
      getJsonValue<std::string>(json, "timestamp_microseconds", "0");
  try {
    timestamp_microseconds = stringToNumber<uint64_t>(timestamp_str);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to parse timestamp_microseconds: " << timestamp_str
               << ", error: " << e.what();
    return false;
  }

  color_image_path = getJsonValue<std::string>(json, "image_name", "");
  if (color_image_path.empty()) {
    LOG(ERROR) << "Missing or empty image_name field";
    return false;
  }
  depth_image_path = color_image_path;

  std::string camera_id_str =
      getJsonValue<std::string>(json, "camera_params_id", "0");
  try {
    camera_params_id = stringToNumber<uint32_t>(camera_id_str);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to parse camera_params_id: " << camera_id_str
               << ", error: " << e.what();
    return false;
  }

  // Handle camera_to_world transform safely
  if (json.contains("camera_to_world") && !json["camera_to_world"].is_null()) {
    camera_to_world =
        transformFromRigidTransform3dJson(json["camera_to_world"]);
  } else {
    LOG(ERROR) << "Missing or null camera_to_world field";
    return false;
  }

  return true;
}

DataLoader::DataLoader(const std::string& color_image_dir,
                       const std::string& depth_image_dir,
                       const std::vector<KeyframeMetadata>& keyframe_metadatas,
                       const std::unordered_map<uint32_t, Camera>& cameras,
                       bool multithreaded, bool fit_to_z_plane,
                       const std::string& output_dir)
    : RgbdDataLoaderInterface(),
      keyframe_metadatas_(keyframe_metadatas),
      cameras_(cameras),
      depth_image_loader_(createDepthImageLoader(
          depth_image_dir, keyframe_metadatas, multithreaded)),
      color_image_loader_(createColorImageLoader(
          color_image_dir, keyframe_metadatas, multithreaded)),
      fit_to_z_plane_(fit_to_z_plane),
      output_dir_(output_dir),
      T_world_to_z0_plane_(Transform::Identity()),
      has_z_plane_transform_(false) {
  if (fit_to_z_plane_) {
    computeZPlaneTransform();
  }
  setup_success_ = true;
}

DataLoadResult DataLoader::loadNext(DepthImage* depth_frame_ptr,
                                    Transform* T_L_C_ptr, Camera* camera_ptr,
                                    ColorImage* color_frame_ptr) {
  CHECK(setup_success_);
  CHECK_NOTNULL(depth_frame_ptr);
  CHECK_NOTNULL(T_L_C_ptr);
  CHECK_NOTNULL(camera_ptr);

  // Because we might fail along the way, increment the frame number before we
  // start.
  ++frame_number_;

  if (frame_number_ > static_cast<int32_t>(keyframe_metadatas_.size())) {
    LOG(INFO) << "Reached the last frame";
    return DataLoadResult::kNoMoreData;
  }

  // Load the image into a Depth Frame.
  CHECK(depth_image_loader_);
  timing::Timer timer_file_depth("file_loading/depth_image");
  if (!depth_image_loader_->getNextImage(depth_frame_ptr)) {
    LOG(INFO) << "Couldn't find depth image";
    return DataLoadResult::kBadFrame;
  }
  timer_file_depth.Stop();

  // Load the color image into a ColorImage
  if (color_frame_ptr) {
    CHECK(color_image_loader_);
    timing::Timer timer_file_color("file_loading/color_image");
    if (!color_image_loader_->getNextImage(color_frame_ptr)) {
      LOG(INFO) << "Couldn't find color image";
      return DataLoadResult::kBadFrame;
    }
    timer_file_color.Stop();
  }

  int32_t current_frame_id = frame_number_ - 1;

  // Get the camera for this frame.
  timing::Timer timer_file_intrinsics("file_loading/camera");
  uint32_t camera_id = keyframe_metadatas_[current_frame_id].camera_params_id;
  auto camera_it = cameras_.find(camera_id);
  if (camera_it != cameras_.end()) {
    *camera_ptr = camera_it->second;
  } else {
    LOG(ERROR) << "Camera parameters not found for camera_params_id: "
               << camera_id << ". Using default camera parameters.";
    return DataLoadResult::kBadFrame;
  }
  timer_file_intrinsics.Stop();

  // Get the next pose
  timing::Timer timer_file_pose("file_loading/pose");
  *T_L_C_ptr = keyframe_metadatas_[current_frame_id].camera_to_world;

  // Apply z-plane alignment transform if enabled
  if (fit_to_z_plane_ && has_z_plane_transform_) {
    *T_L_C_ptr = T_world_to_z0_plane_ * (*T_L_C_ptr);
  }

  // Check that the loaded data doesn't contain NaNs or a faulty rotation
  // matrix. This does occur. If we find one, skip that frame and move to the
  // next.
  constexpr float kRotationMatrixDetEpsilon = 1e-4;
  if (!T_L_C_ptr->matrix().allFinite() ||
      std::abs(T_L_C_ptr->matrix().block<3, 3>(0, 0).determinant() - 1.0f) >
          kRotationMatrixDetEpsilon) {
    LOG(WARNING) << "Bad camera to world transform matrix";
    return DataLoadResult::kBadFrame;  // Bad data, but keep going.
  }

  VLOG(1) << "Current frame_id: " << current_frame_id << ", timestamp: "
          << keyframe_metadatas_[current_frame_id].timestamp_microseconds
          << ", camera_to_world:\n"
          << T_L_C_ptr->matrix() << "\ncamera_id: "
          << keyframe_metadatas_[current_frame_id].camera_params_id
          << ", camera:\n"
          << *camera_ptr;

  timer_file_pose.Stop();
  return DataLoadResult::kSuccess;
}

DataLoadResult DataLoader::loadNext(
    DepthImage* depth_frame_ptr, Transform* T_L_D_ptr, Camera* depth_camera_ptr,
    ColorImage* color_frame_ptr, Transform* T_L_C_ptr, Camera* color_camera_ptr,
    Time*, Transform*, Time*) {
  // NOTE: The other pointers are checked non-null below
  CHECK_NOTNULL(color_frame_ptr);
  CHECK_NOTNULL(T_L_C_ptr);
  CHECK_NOTNULL(color_camera_ptr);
  // For the replica dataset the depth and color cameras are the same, so just
  // copying over.
  auto result =
      loadNext(depth_frame_ptr, T_L_D_ptr, depth_camera_ptr, color_frame_ptr);
  *T_L_C_ptr = *T_L_D_ptr;
  *color_camera_ptr = *depth_camera_ptr;
  return result;
}

void DataLoader::computeZPlaneTransform() {
  if (keyframe_metadatas_.empty()) {
    LOG(WARNING) << "No keyframe metadata available for z-plane alignment";
    return;
  }

  // Extract all camera positions
  std::vector<Eigen::Vector3f> positions;
  positions.reserve(keyframe_metadatas_.size());

  for (const auto& metadata : keyframe_metadatas_) {
    positions.push_back(metadata.camera_to_world.translation());
  }

  if (positions.size() < 3) {
    LOG(WARNING) << "Need at least 3 poses for z-plane alignment, got "
                 << positions.size();
    return;
  }

  // Compute centroid using double precision
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto& pos : positions) {
    centroid += pos.cast<double>();
  }
  centroid /= static_cast<double>(positions.size());

  // Build matrix of centered positions
  Eigen::Matrix<double, 3, Eigen::Dynamic> centered_positions(3,
                                                              positions.size());
  for (size_t i = 0; i < positions.size(); ++i) {
    centered_positions.col(i) = positions[i].cast<double>() - centroid;
  }

  // Compute SVD to find the best-fit plane
  Eigen::JacobiSVD<Eigen::Matrix<double, 3, Eigen::Dynamic>> svd(
      centered_positions, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // The plane normal is the column of U corresponding to the smallest singular
  // value
  Eigen::Vector3d plane_normal = svd.matrixU().col(2);

  // Ensure the normal points upward (positive z component in the original
  // frame)
  if (plane_normal.z() < 0) {
    plane_normal = -plane_normal;
  }

  // Compute rotation to align plane normal with z-axis (0, 0, 1)
  Eigen::Vector3d target_normal(0.0, 0.0, 1.0);

  // If the plane normal is already aligned with z-axis, use identity rotation
  if ((plane_normal - target_normal).norm() < 1e-10) {
    T_world_to_z0_plane_.linear() = Eigen::Matrix3f::Identity();
  } else if ((plane_normal + target_normal).norm() < 1e-10) {
    // If plane normal is opposite to z-axis, rotate 180 degrees around x-axis
    Eigen::AngleAxisd rotation_d(M_PI, Eigen::Vector3d::UnitX());
    T_world_to_z0_plane_.linear() = rotation_d.matrix().cast<float>();
  } else {
    // Compute rotation using Rodrigues' formula
    Eigen::Vector3d rotation_axis =
        plane_normal.cross(target_normal).normalized();
    double cos_angle = plane_normal.dot(target_normal);
    double angle = std::acos(std::clamp(cos_angle, -1.0, 1.0));

    Eigen::AngleAxisd rotation(angle, rotation_axis);
    T_world_to_z0_plane_.linear() = rotation.matrix().cast<float>();
  }

  // Transform the centroid to the new coordinate system and set translation
  // so that the centroid lies on the z=0 plane
  Eigen::Vector3d transformed_centroid_d =
      T_world_to_z0_plane_.linear().cast<double>() * centroid;
  T_world_to_z0_plane_.translation() = Eigen::Vector3f(
      0.0f, 0.0f, static_cast<float>(-transformed_centroid_d.z()));

  has_z_plane_transform_ = true;

  LOG(INFO) << "Computed z-plane alignment transform from " << positions.size()
            << " poses";
  LOG(INFO) << "Original plane normal: " << plane_normal.transpose();
  LOG(INFO) << "Centroid: " << centroid.transpose();
  LOG(INFO) << "Transform matrix:\n" << T_world_to_z0_plane_.matrix();

  // Save transform to JSON if output directory is provided
  if (!output_dir_.empty()) {
    saveTransformToJson();
  }
}

bool DataLoader::saveTransformToJson() const {
  if (!has_z_plane_transform_ || output_dir_.empty()) {
    return false;
  }

  try {
    std::stringstream json_stream;

    // Save rotation matrix
    const Eigen::Matrix3f& rotation = T_world_to_z0_plane_.linear();
    json_stream << "{\n";
    json_stream << "  \"rotation\": [\n";
    json_stream << "    [" << rotation(0, 0) << ", " << rotation(0, 1) << ", "
                << rotation(0, 2) << "],\n";
    json_stream << "    [" << rotation(1, 0) << ", " << rotation(1, 1) << ", "
                << rotation(1, 2) << "],\n";
    json_stream << "    [" << rotation(2, 0) << ", " << rotation(2, 1) << ", "
                << rotation(2, 2) << "]\n";
    json_stream << "  ],\n";

    // Save translation
    const Eigen::Vector3f& translation = T_world_to_z0_plane_.translation();
    json_stream << "  \"translation\": [" << translation.x() << ", "
                << translation.y() << ", " << translation.z() << "],\n";

    // Save as 4x4 homogeneous matrix for convenience
    const Eigen::Matrix4f& matrix = T_world_to_z0_plane_.matrix();
    json_stream << "  \"homogeneous_matrix\": [\n";
    json_stream << "    [" << matrix(0, 0) << ", " << matrix(0, 1) << ", "
                << matrix(0, 2) << ", " << matrix(0, 3) << "],\n";
    json_stream << "    [" << matrix(1, 0) << ", " << matrix(1, 1) << ", "
                << matrix(1, 2) << ", " << matrix(1, 3) << "],\n";
    json_stream << "    [" << matrix(2, 0) << ", " << matrix(2, 1) << ", "
                << matrix(2, 2) << ", " << matrix(2, 3) << "],\n";
    json_stream << "    [" << matrix(3, 0) << ", " << matrix(3, 1) << ", "
                << matrix(3, 2) << ", " << matrix(3, 3) << "]\n";
    json_stream << "  ]\n";
    json_stream << "}\n";

    std::string json_path = output_dir_ + "/T_world_to_z0.json";
    std::ofstream json_file(json_path);
    if (!json_file.is_open()) {
      LOG(ERROR) << "Failed to open file for writing: " << json_path;
      return false;
    }

    json_file << json_stream.str();
    json_file.close();

    LOG(INFO) << "Saved z-plane transform to: " << json_path;
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to save transform to JSON: " << e.what();
    return false;
  }
}

}  // namespace cusfm_data
}  // namespace datasets
}  // namespace nvblox
