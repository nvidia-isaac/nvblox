/*
Copyright 2022-2024 NVIDIA CORPORATION

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
#include "nvblox/core/hash.h"
#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/geometry/transforms.h"
#include "nvblox/integrators/view_calculator.h"
#include "nvblox/rays/ray_caster.h"
#include "nvblox/utils/cuda_kernel_utils.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

ViewCalculator::ViewCalculator()
    : ViewCalculator(std::make_shared<CudaStreamOwning>()) {}

ViewCalculator::ViewCalculator(std::shared_ptr<CudaStream> cuda_stream)
    : raycasting_viewpoint_cache_(std::make_shared<ViewpointCache>()),
      projection_viewpoint_cache_(std::make_shared<ViewpointCache>()),
      cuda_stream_(cuda_stream) {}

unsigned int ViewCalculator::raycast_subsampling_factor() const {
  return raycast_subsampling_factor_;
}

void ViewCalculator::raycast_subsampling_factor(
    unsigned int raycast_subsampling_factor) {
  CHECK_GT(raycast_subsampling_factor, 0U);
  raycast_subsampling_factor_ = raycast_subsampling_factor;
}

WorkspaceBoundsType ViewCalculator::workspace_bounds_type() const {
  return workspace_bounds_type_;
}

void ViewCalculator::workspace_bounds_type(
    WorkspaceBoundsType workspace_bounds_type) {
  workspace_bounds_type_ = workspace_bounds_type;
}

Vector3f ViewCalculator::workspace_bounds_min_corner_m() const {
  return Vector3f(workspace_bounds_min_corner_x_m_,
                  workspace_bounds_min_corner_y_m_,
                  workspace_bounds_min_height_m_);
}

Vector3f ViewCalculator::workspace_bounds_max_corner_m() const {
  return Vector3f(workspace_bounds_max_corner_x_m_,
                  workspace_bounds_max_corner_y_m_,
                  workspace_bounds_max_height_m_);
}

void ViewCalculator::workspace_bounds_min_corner_m(
    const Vector3f& workspace_bounds_min_corner_m) {
  workspace_bounds_min_corner_x_m_ = workspace_bounds_min_corner_m.x();
  workspace_bounds_min_corner_y_m_ = workspace_bounds_min_corner_m.y();
  workspace_bounds_min_height_m_ = workspace_bounds_min_corner_m.z();
}

void ViewCalculator::workspace_bounds_max_corner_m(
    const Vector3f& workspace_bounds_max_corner_m) {
  workspace_bounds_max_corner_x_m_ = workspace_bounds_max_corner_m.x();
  workspace_bounds_max_corner_y_m_ = workspace_bounds_max_corner_m.y();
  workspace_bounds_max_height_m_ = workspace_bounds_max_corner_m.z();
}

bool ViewCalculator::cache_last_viewpoint() const {
  return cache_last_viewpoint_;
}

void ViewCalculator::cache_last_viewpoint(const bool cache_last_viewpoint) {
  cache_last_viewpoint_ = cache_last_viewpoint;
}

std::shared_ptr<ViewpointCache> ViewCalculator::get_viewpoint_cache(
    const CalculationType calculation_type) const {
  switch (calculation_type) {
    case CalculationType::kRaycasting:
      return raycasting_viewpoint_cache_;
    case CalculationType::kPlanes:
      return projection_viewpoint_cache_;
    default:
      CHECK(false)
          << "Requested viewpoint calculation type is not implemented.";
      return std::shared_ptr<ViewpointCache>();
  }
}

void ViewCalculator::set_viewpoint_cache(
    std::shared_ptr<ViewpointCache> viewpoint_cache,
    const CalculationType calculation_type) {
  switch (calculation_type) {
    case CalculationType::kRaycasting:
      raycasting_viewpoint_cache_ = viewpoint_cache;
      break;
    case CalculationType::kPlanes:
      projection_viewpoint_cache_ = viewpoint_cache;
      break;
    default:
      CHECK(false)
          << "Requested viewpoint calculation type is not implemented.";
  }
}

parameters::ParameterTreeNode ViewCalculator::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "view_calculator" : name_remap;
  // NOTE(alexmillane): Wrapping our weighting function to_string version in the
  // std::function for passing to the parameter tree node constructor because it
  // seems to have trouble with template deduction.
  std::function<std::string(const WorkspaceBoundsType&)>
      workspace_bounds_to_string =
          [](const WorkspaceBoundsType& w) { return to_string(w); };
  return ParameterTreeNode(
      name,
      {
          ParameterTreeNode("raycast_subsampling_factor",
                            raycast_subsampling_factor_),
          ParameterTreeNode("workspace_bounds_type", workspace_bounds_type_,
                            workspace_bounds_to_string),
          ParameterTreeNode("workspace_bounds_min_height_m",
                            workspace_bounds_min_height_m_),
          ParameterTreeNode("workspace_bounds_max_height_m",
                            workspace_bounds_max_height_m_),
          ParameterTreeNode("workspace_bounds_min_corner_x_m",
                            workspace_bounds_min_corner_x_m_),
          ParameterTreeNode("workspace_bounds_max_corner_x_m",
                            workspace_bounds_max_corner_x_m_),
          ParameterTreeNode("workspace_bounds_min_corner_y_m",
                            workspace_bounds_min_corner_y_m_),
          ParameterTreeNode("workspace_bounds_max_corner_y_m",
                            workspace_bounds_max_corner_y_m_),
      });
}

// AABB linear indexing
// - We index in x-major, i.e. x is varied first, then y, then z.
// - Linear indexing within an AABB is relative and starts at zero. This is
//   not true for AABB 3D indexing which is w.r.t. the layer origin.
__host__ __device__ inline size_t layerIndexToAabbLinearIndex(
    const Index3D& index, const Index3D& aabb_min, const Index3D& aabb_size) {
  const Index3D index_shifted = index - aabb_min;
  return index_shifted.x() +                                 // NOLINT
         index_shifted.y() * aabb_size.x() +                 // NOLINT
         index_shifted.z() * aabb_size.x() * aabb_size.y();  // NOLINT
}

__host__ __device__ inline Index3D aabbLinearIndexToLayerIndex(
    const size_t lin_idx, const Index3D& aabb_min, const Index3D& aabb_size) {
  const Index3D index(lin_idx % aabb_size.x(),                     // NOLINT
                      (lin_idx / aabb_size.x()) % aabb_size.y(),   // NOLINT
                      lin_idx / (aabb_size.x() * aabb_size.y()));  // NOLINT
  return index + aabb_min;
}

__device__ void setIndexUpdated(const Index3D& index_to_update,
                                const Index3D& aabb_min,
                                const Index3D& aabb_size, bool* aabb_updated) {
  const size_t linear_size = aabb_size.x() * aabb_size.y() * aabb_size.z();
  const size_t lin_idx =
      layerIndexToAabbLinearIndex(index_to_update, aabb_min, aabb_size);
  if (lin_idx < linear_size) {
    aabb_updated[lin_idx] = true;
  }
}

// Version producing: std::vector<Index3D>
void convertAabbUpdatedToVector(const host_vector<bool>& aabb_updated,
                                const Index3D& aabb_min,
                                const Index3D& aabb_size,
                                std::vector<Index3D>* indices) {
  indices->reserve(aabb_updated.size());
  for (size_t i = 0; i < aabb_updated.size(); i++) {
    if (aabb_updated[i]) {
      indices->push_back(aabbLinearIndexToLayerIndex(i, aabb_min, aabb_size));
    }
  }
}

template <typename SensorType>
__global__ void combinedBlockIndicesInImageKernel(
    const Transform T_L_C, const SensorType camera, const float* image,
    int rows, int cols, const float block_size,
    const float max_integration_distance_m,
    const float max_integration_distance_behind_surface_m,
    int raycast_subsampling_factor, const Index3D aabb_min,
    const Index3D aabb_size, bool* aabb_updated) {
  // First, figure out which pixel we're in.
  const int ray_idx_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int ray_idx_row = blockIdx.y * blockDim.y + threadIdx.y;
  int pixel_row = ray_idx_row * raycast_subsampling_factor;
  int pixel_col = ray_idx_col * raycast_subsampling_factor;

  // Hooray we do nothing.
  if (pixel_row >= (rows + raycast_subsampling_factor - 1) ||
      pixel_col >= (cols + raycast_subsampling_factor - 1)) {
    return;
  } else {
    // Move remaining overhanging pixels back to the borders.
    if (pixel_row >= rows) {
      pixel_row = rows - 1;
    }
    if (pixel_col >= cols) {
      pixel_col = cols - 1;
    }
  }

  // Look up the pixel we care about.
  float depth = image::access<float>(pixel_row, pixel_col, cols, image);
  if (depth <= 0.0f) {
    return;
  }

  // Ok now project this thing into space.
  Vector3f p_C = camera.unprojectFromPixelIndices(
      Index2D(pixel_col, pixel_row),
      depth + max_integration_distance_behind_surface_m);

  // Truncate if we're exceeding max integration distance
  if (max_integration_distance_m > 0.0f &&
      p_C.norm() > max_integration_distance_m) {
    p_C = p_C.normalized() * max_integration_distance_m;
  }
  Vector3f p_L = T_L_C * p_C;

  // Now we have the position of the thing in space. Now we need the block
  // index.
  Index3D block_index = getBlockIndexFromPositionInLayer(block_size, p_L);
  setIndexUpdated(block_index, aabb_min, aabb_size, aabb_updated);

  // Ok raycast to the correct point in the block.
  RayCaster raycaster(T_L_C.translation() / block_size, p_L / block_size);
  Index3D ray_index = Index3D::Zero();

  while (raycaster.nextRayIndex(&ray_index)) {
    setIndexUpdated(ray_index, aabb_min, aabb_size, aabb_updated);
  }
}

template <typename SensorType>
std::vector<Index3D> ViewCalculator::getBlocksInImageViewRaycast(
    const MaskedDepthImageConstView& depth_frame, const Transform& T_L_C,
    const SensorType& sensor, const float block_size,
    const float max_integration_distance_behind_surface_m,
    const float max_integration_distance_m) {
  timing::Timer total_timer("view_calculator/raycast");
  // Check cache
  CHECK_NOTNULL(raycasting_viewpoint_cache_);
  if (cache_last_viewpoint_) {
    if (auto cached_result =
            raycasting_viewpoint_cache_->getCachedResult(T_L_C, sensor);
        cached_result.has_value()) {
      return cached_result.value();
    }
  }

  timing::Timer setup_timer("view_calculator/raycast/setup");

  // Aight so first we have to get the AABB of this guy.
  AxisAlignedBoundingBox aabb_L =
      sensor.getViewAABB(T_L_C, 0.0f, max_integration_distance_m);
  NVBLOX_CHECK(aabb_L.volume() > 0, "Empty bounding box");
  NVBLOX_CHECK(!std::isinf(aabb_L.volume()), "Infinite bounding box");
  NVBLOX_CHECK(!std::isnan(aabb_L.volume()), "Nan bounding box");

  // Apply the workspace bounds,
  // i.e. make sure we only return blocks that are within the workspace limits.
  if (!applyWorkspaceBounds(aabb_L, workspace_bounds_type_,
                            workspace_bounds_min_corner_m(),
                            workspace_bounds_max_corner_m(), &aabb_L)) {
    // Return an empty vector of blocks to update if the workspace is not valid
    // (i.e. empty).
    return std::vector<Index3D>();
  }

  // Get the min index and the max index.
  const Index3D min_index =
      getBlockIndexFromPositionInLayer(block_size, aabb_L.min());
  const Index3D max_index =
      getBlockIndexFromPositionInLayer(block_size, aabb_L.max());
  const Index3D aabb_size = max_index - min_index + Index3D::Ones();
  const size_t aabb_linear_size = aabb_size.x() * aabb_size.y() * aabb_size.z();

  // A 3D grid of bools, one for each block in the AABB, which indicates if it
  // is in the view. The 3D grid is represented as a flat vector.
  if (aabb_linear_size > aabb_device_buffer_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size =
        static_cast<int>(kBufferExpansionFactor * aabb_linear_size);
    aabb_device_buffer_.reserveAsync(new_size, *cuda_stream_);
    aabb_host_buffer_.reserveAsync(new_size, *cuda_stream_);
  }

  aabb_device_buffer_.resizeAsync(aabb_linear_size, *cuda_stream_);
  aabb_device_buffer_.setZeroAsync(*cuda_stream_);
  aabb_host_buffer_.resizeAsync(aabb_linear_size, *cuda_stream_);

  setup_timer.Stop();

  // Raycast
  getBlocksByRaycastingPixelsAsync(T_L_C, sensor, depth_frame, block_size,
                                   max_integration_distance_behind_surface_m,
                                   max_integration_distance_m, min_index,
                                   aabb_size, aabb_device_buffer_.data());

  // Output vector.
  timing::Timer output_timer("view_calculator/raycast/output");
  checkCudaErrors(cudaMemcpyAsync(
      aabb_host_buffer_.data(), aabb_device_buffer_.data(),
      sizeof(bool) * aabb_linear_size, cudaMemcpyDeviceToHost, *cuda_stream_));
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  std::vector<Index3D> output_vector;
  convertAabbUpdatedToVector(aabb_host_buffer_, min_index, aabb_size,
                             &output_vector);

  // Cache
  if (cache_last_viewpoint_) {
    raycasting_viewpoint_cache_->storeResultInCache(T_L_C, sensor,
                                                    output_vector);
  }
  return output_vector;
}

template <typename SensorType>
void ViewCalculator::getBlocksByRaycastingPixelsAsync(
    const Transform& T_L_C, const SensorType& sensor,
    const MaskedDepthImageConstView& depth_frame, float block_size,
    const float max_integration_distance_behind_surface_m,
    const float max_integration_distance_m, const Index3D& min_index,
    const Index3D& aabb_size, bool* aabb_updated_cuda) {
  timing::Timer combined_kernel_timer(
      "view_calculator/raycast/raycast_pixels_kernel");
  // Number of rays per dimension. Depth frame size / subsampling rate.
  const int num_subsampled_rows =
      std::ceil(static_cast<float>(depth_frame.rows() + 1) /
                static_cast<float>(raycast_subsampling_factor_));
  const int num_subsampled_cols =
      std::ceil(static_cast<float>(depth_frame.cols() + 1) /
                static_cast<float>(raycast_subsampling_factor_));

  // We'll do warps of 16x16 pixels in the image. This is 1024 threads which
  // is in the recommended 512-1024 range.
  constexpr int kThreadDim = 16;
  const int rounded_rows = divideRoundUp(num_subsampled_rows, kThreadDim);
  const int rounded_cols = divideRoundUp(num_subsampled_cols, kThreadDim);
  dim3 block_dim(rounded_cols, rounded_rows);
  dim3 thread_dim(kThreadDim, kThreadDim);

  combinedBlockIndicesInImageKernel<<<block_dim, thread_dim, 0,
                                      *cuda_stream_>>>(
      T_L_C, sensor, depth_frame.dataConstPtr(), depth_frame.rows(),
      depth_frame.cols(), block_size, max_integration_distance_m,
      max_integration_distance_behind_surface_m, raycast_subsampling_factor_,
      min_index, aabb_size, aabb_updated_cuda);
  checkCudaErrors(cudaPeekAtLastError());
  combined_kernel_timer.Stop();
}

// Specialization for Camera that speeds up computation by making use of
// normalized image coordinates.
template <>
std::vector<Index3D> ViewCalculator::getVisibleBlocksByProjection<Camera>(
    const std::vector<Index3D>& block_indices, const Camera& camera,
    const Transform& T_C_L, const float block_size, const float min_distance) {
  // Extract rotation and translation component
  const Eigen::Matrix3f rotation_C_L = T_C_L.rotation();
  const Eigen::Vector3f translation_C_L = T_C_L.translation();

  // Get the 2D viewport of the camera. We use normalized image
  // coordinates rather than pixels to avoid having to apply the
  // camera intrinsics to each point we want to check. A small margin
  // is added to also capture blocks which intersect a frustum plane
  // but have their center point outside the plane.
  const CameraViewport normalized_viewport =
      camera.getNormalizedViewport(getViewportMargin(camera.height()));

  std::vector<Index3D> block_indices_in_view;

  for (const Index3D& block_index : block_indices) {
    // Transform the block center into camera frame
    const Eigen::Vector3f p3d_layer =
        getCenterPositionFromBlockIndex(block_size, block_index);
    const Eigen::Vector3f p3d_cam = rotation_C_L * p3d_layer + translation_C_L;

    if (p3d_cam[2] > min_distance) {
      // Project into normalized camera coordinates
      Eigen::Vector2f p2d_normalized_cam;
      if (camera.projectToNormalizedCoordinates(p3d_cam, &p2d_normalized_cam)) {
        // Check if the projected point is inside the viewport
        if (normalized_viewport.contains(p2d_normalized_cam)) {
          block_indices_in_view.push_back(block_index);
        }
      }
    }
  }
  return block_indices_in_view;
}

}  // namespace nvblox
