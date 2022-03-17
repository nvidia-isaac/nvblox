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

#include <memory>
#include <string>

#include "nvblox/core/types.h"
#include "nvblox/datasets/image_loader.h"

namespace nvblox {
namespace datasets {
namespace threedmatch {

/// Parses a 4x4 matrix from a text file in 3D Match format: space-separated.
bool parsePoseFromFile(const std::string& filename, Transform* transform);

/// Parse 3x3 camera intrinsics matrix from 3D Match format: space-separated.
bool parseCameraFromFile(const std::string& filename,
                         Eigen::Matrix3f* intrinsics);

std::string getPathForCameraIntrinsics(const std::string& base_path);
std::string getPathForFramePose(const std::string& base_path, const int seq_id,
                                const int frame_id);
std::string getPathForDepthImage(const std::string& base_path, const int seq_id,
                                 const int frame_id);
std::string getPathForColorImage(const std::string& base_path, const int seq_id,
                                 const int frame_id);

// Factory functions for single-threaded 3DMatch image loaders
std::unique_ptr<nvblox::datasets::ImageLoader<DepthImage>>
createDepthImageLoader(const std::string& base_path, const int seq_id,
                       MemoryType memory_type = kDefaultImageMemoryType);
std::unique_ptr<nvblox::datasets::ImageLoader<ColorImage>>
createColorImageLoader(const std::string& base_path, const int seq_id,
                       MemoryType memory_type = kDefaultImageMemoryType);

// Factory functions for multi-threaded 3DMatch image loaders
std::unique_ptr<nvblox::datasets::ImageLoader<DepthImage>>
createMultithreadedDepthImageLoader(
    const std::string& base_path, const int seq_id, const int num_threads,
    MemoryType memory_type = kDefaultImageMemoryType);
std::unique_ptr<nvblox::datasets::ImageLoader<ColorImage>>
createMultithreadedColorImageLoader(
    const std::string& base_path, const int seq_id, const int num_threads,
    MemoryType memory_type = kDefaultImageMemoryType);

}  // namespace threedmatch
}  // namespace datasets
}  // namespace nvblox
