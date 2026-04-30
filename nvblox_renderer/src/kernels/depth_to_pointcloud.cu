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

#include "nvblox/renderer/kernels/depth_to_pointcloud.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include "nvblox/core/internal/error_check.h"

namespace nvblox {
namespace renderer {

__global__ void depthToColoredPointCloudKernel(
    const float* __restrict__ depth,
    const uint8_t* __restrict__ color,  // RGB format, 3 bytes per pixel
    Camera depth_cam,  // Passed by value (small, device-compatible)
    Camera color_cam, const Transform T_color_depth, bool has_transform,
    PointCloudVisualizer::Point* __restrict__ points, int max_points,
    int* __restrict__ num_points, float min_depth, float max_depth) {
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;

  if (px >= depth_cam.width() || py >= depth_cam.height()) {
    return;
  }

  // Read depth value
  float d = depth[py * depth_cam.width() + px];

  // Skip invalid depth
  if (d <= min_depth || d > max_depth || !isfinite(d)) {
    return;
  }

  // Unproject to 3D in depth camera frame using Camera's method
  Vector3f p_d = depth_cam.unprojectFromPixelIndices(Index2D(px, py), d);

  // Transform to color camera frame
  const Vector3f p_c = has_transform ? T_color_depth * p_d : p_d;

  // Project to color image to sample color
  uint8_t r = 128, g = 128, b = 128;  // Default gray if lookup fails
  Vector2f u_color;
  if (color_cam.project(p_c, &u_color, 0.001f, true)) {
    int u = static_cast<int>(u_color.x());
    int v = static_cast<int>(u_color.y());
    // Bounds check: ensure pixel coordinates are within the color image
    if (u >= 0 && u < color_cam.width() && v >= 0 && v < color_cam.height()) {
      size_t idx = (v * color_cam.width() + u) * 3;
      r = color[idx + 0];
      g = color[idx + 1];
      b = color[idx + 2];
    }
  }

  // Atomically reserve a slot in the output buffer.
  int out_idx = atomicAdd(num_points, 1);

  // Bounds check: if buffer is full, decrement counter and discard point.
  // This keeps the counter accurate for the caller.
  if (out_idx >= max_points) {
    atomicSub(num_points, 1);
    return;
  }

  // Write output point
  // Convert from depth camera frame to viewer frame:
  // - Negate X for mirror-view (intuitive when looking at yourself)
  // - Negate Y for Y-up coordinate system
  points[out_idx].x = -p_d.x();  // Mirror X for intuitive left/right
  points[out_idx].y = -p_d.y();  // Flip Y for Y-up coordinate system
  points[out_idx].z = p_d.z();
  points[out_idx].r = r;
  points[out_idx].g = g;
  points[out_idx].b = b;
  points[out_idx].a = 255;
}

bool depthToColoredPointCloud(const float* depth_ptr, const uint8_t* color_ptr,
                              const Camera& depth_cam, const Camera& color_cam,
                              const Transform* T_color_depth,
                              PointCloudVisualizer::Point* points_out,
                              int max_points, int* num_points_out,
                              float min_depth, float max_depth,
                              const CudaStream& stream) {
  // Validate required pointers
  if (!depth_ptr || !color_ptr || !points_out || !num_points_out) {
    LOG(ERROR) << "Null pointer passed to depthToColoredPointCloud";
    return false;
  }

  cudaStream_t cuda_stream = stream;

  checkCudaErrors(cudaMemsetAsync(num_points_out, 0, sizeof(int), cuda_stream));

  dim3 block(16, 16);
  dim3 grid((depth_cam.width() + block.x - 1) / block.x,
            (depth_cam.height() + block.y - 1) / block.y);

  const bool has_transform = (T_color_depth != nullptr);
  const Transform transform =
      has_transform ? *T_color_depth : Transform::Identity();

  depthToColoredPointCloudKernel<<<grid, block, 0, cuda_stream>>>(
      depth_ptr, color_ptr, depth_cam, color_cam, transform, has_transform,
      points_out, max_points, num_points_out, min_depth, max_depth);
  checkCudaErrors(cudaPeekAtLastError());

  return true;
}

}  // namespace renderer
}  // namespace nvblox
