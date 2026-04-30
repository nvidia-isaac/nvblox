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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/renderer/visualizers/point_cloud_visualizer.h"
#include "nvblox/sensors/camera.h"

namespace nvblox {
namespace renderer {

/// Convert RGBD images to a colored point cloud.
///
/// This function takes depth and color images from nvblox sensors and generates
/// a point cloud suitable for visualization. The conversion is performed
/// entirely on the GPU using CUDA. Image dimensions are obtained from the
/// Camera objects.
///
/// @param depth_ptr CUDA device pointer to depth image (float, meters).
/// @param color_ptr CUDA device pointer to color image (RGB, 3 bytes/pixel).
/// @param depth_cam Camera intrinsics for the depth sensor (includes
/// width/height).
/// @param color_cam Camera intrinsics for the color sensor (includes
/// width/height).
/// @param T_color_depth Transform from depth to color frame (nullptr if same).
/// @param points_out Output device pointer for point cloud data.
/// @param max_points Maximum number of points that can be written to
/// points_out.
/// @param num_points_out Output device pointer for point count.
/// @param min_depth Minimum valid depth (meters).
/// @param max_depth Maximum valid depth (meters).
/// @param stream CUDA stream for async execution.
/// @return true if conversion succeeded, false on error (null pointers, CUDA
/// errors).
bool depthToColoredPointCloud(
    const float* depth_ptr, const uint8_t* color_ptr, const Camera& depth_cam,
    const Camera& color_cam,
    const Transform* T_color_depth,  // nullptr for identity
    PointCloudVisualizer::Point* points_out, int max_points,
    int* num_points_out, float min_depth, float max_depth,
    const CudaStream& stream);

}  // namespace renderer
}  // namespace nvblox
