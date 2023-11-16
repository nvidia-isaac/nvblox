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
#include "nvblox/core/hash.h"
#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/integrators/view_calculator.h"
#include "nvblox/rays/ray_caster.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

ViewCalculator::ViewCalculator()
    : ViewCalculator(std::make_shared<CudaStreamOwning>()) {}

ViewCalculator::ViewCalculator(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

unsigned int ViewCalculator::raycast_subsampling_factor() const {
  return raycast_subsampling_factor_;
}

void ViewCalculator::raycast_subsampling_factor(
    unsigned int raycast_subsampling_factor) {
  CHECK_GT(raycast_subsampling_factor, 0);
  raycast_subsampling_factor_ = raycast_subsampling_factor;
}

parameters::ParameterTreeNode ViewCalculator::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "view_calculator" : name_remap;
  return ParameterTreeNode(
      name, {
                ParameterTreeNode("raycast_to_pixels:", raycast_to_pixels_),
                ParameterTreeNode("raycast_subsampling_factor:",
                                  raycast_subsampling_factor_),
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

template <typename T>
void convertAabbUpdatedToVector(const Index3D& aabb_min,
                                const Index3D& aabb_size,
                                size_t aabb_linear_size, bool* aabb_updated,
                                T* indices) {
  indices->reserve(aabb_linear_size);
  for (size_t i = 0; i < aabb_linear_size; i++) {
    if (aabb_updated[i]) {
      indices->push_back(aabbLinearIndexToLayerIndex(i, aabb_min, aabb_size));
    }
  }
}

template <typename SensorType>
__global__ void getBlockIndicesInImageKernel(
    const Transform T_L_C, const SensorType camera, const float* image,
    int rows, int cols, const float block_size,
    const float max_integration_distance_m,
    const float max_integration_distance_behind_surface_m,
    const Index3D aabb_min, const Index3D aabb_size, bool* aabb_updated) {
  // First, figure out which pixel we're in.
  int pixel_row = blockIdx.x * blockDim.x + threadIdx.x;
  int pixel_col = blockIdx.y * blockDim.y + threadIdx.y;

  // Hooray we do nothing.
  if (pixel_row >= rows || pixel_col >= cols) {
    return;
  }

  // Look up the pixel we care about.
  float depth = image::access<float>(pixel_row, pixel_col, cols, image);
  if (depth <= 0.0f) {
    return;
  }
  if (max_integration_distance_m > 0.0f && depth > max_integration_distance_m) {
    depth = max_integration_distance_m;
  }

  // Ok now project this thing into space.
  Vector3f p_C = (depth + max_integration_distance_behind_surface_m) *
                 camera.vectorFromPixelIndices(Index2D(pixel_col, pixel_row));
  Vector3f p_L = T_L_C * p_C;

  // Now we have the position of the thing in space. Now we need the block
  // index.
  Index3D block_index = getBlockIndexFromPositionInLayer(block_size, p_L);
  setIndexUpdated(block_index, aabb_min, aabb_size, aabb_updated);
}

__global__ void raycastToBlocksKernel(int num_blocks, Index3D* block_indices,
                                      const Transform T_L_C, float block_size,
                                      const Index3D aabb_min,
                                      const Index3D aabb_size,
                                      bool* aabb_updated) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int corner_index = threadIdx.y;

  if (index >= num_blocks) {
    return;
  }

  constexpr float corner_increment_table[9][3] = {
      {0.0f, 0.0f, 0.0f},  // NOLINT
      {1.0f, 0.0f, 0.0f},  // NOLINT
      {0.0f, 1.0f, 0.0f},  // NOLINT
      {0.0f, 0.0f, 1.0f},  // NOLINT
      {1.0f, 1.0f, 0.0f},  // NOLINT
      {1.0f, 0.0f, 1.0f},  // NOLINT
      {0.0f, 1.0f, 1.0f},  // NOLINT
      {1.0f, 1.0f, 1.0f},  // NOLINT
      {0.5f, 0.5f, 0.5f},  // NOLINT
  };

  const Vector3f increment(corner_increment_table[corner_index][0],
                           corner_increment_table[corner_index][1],
                           corner_increment_table[corner_index][2]);

  const Index3D& block_index = block_indices[index];

  RayCaster raycaster(T_L_C.translation() / block_size,
                      block_index.cast<float>() + increment);
  Index3D ray_index;
  while (raycaster.nextRayIndex(&ray_index)) {
    setIndexUpdated(ray_index, aabb_min, aabb_size, aabb_updated);
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
  const int ray_idx_row = blockIdx.x * blockDim.x + threadIdx.x;
  const int ray_idx_col = blockIdx.y * blockDim.y + threadIdx.y;
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
  if (max_integration_distance_m > 0.0f && depth > max_integration_distance_m) {
    depth = max_integration_distance_m;
  }

  // Ok now project this thing into space.
  Vector3f p_C = (depth + max_integration_distance_behind_surface_m) *
                 camera.vectorFromPixelIndices(Index2D(pixel_col, pixel_row));
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
std::vector<Index3D> ViewCalculator::getBlocksInImageViewRaycastTemplate(
    const DepthImage& depth_frame, const Transform& T_L_C,
    const SensorType& camera, const float block_size,
    const float max_integration_distance_behind_surface_m,
    const float max_integration_distance_m) {
  timing::Timer total_timer("view_calculator/raycast");
  timing::Timer setup_timer("view_calculator/raycast/setup");

  // Aight so first we have to get the AABB of this guy.
  const AxisAlignedBoundingBox aabb_L =
      camera.getViewAABB(T_L_C, 0.0f, max_integration_distance_m);

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
  if (raycast_to_pixels_) {
    getBlocksByRaycastingPixels(T_L_C, camera, depth_frame, block_size,
                                max_integration_distance_behind_surface_m,
                                max_integration_distance_m, min_index,
                                aabb_size, aabb_device_buffer_.data());
  } else {
    getBlocksByRaycastingCorners(T_L_C, camera, depth_frame, block_size,
                                 max_integration_distance_behind_surface_m,
                                 max_integration_distance_m, min_index,
                                 aabb_size, aabb_device_buffer_.data());
  }

  // Output vector.
  timing::Timer output_timer("view_calculator/raycast/output");
  checkCudaErrors(cudaMemcpyAsync(
      aabb_host_buffer_.data(), aabb_device_buffer_.data(),
      sizeof(bool) * aabb_linear_size, cudaMemcpyDeviceToHost, *cuda_stream_));
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  std::vector<Index3D> output_vector;
  convertAabbUpdatedToVector<std::vector<Index3D>>(
      min_index, aabb_size, aabb_linear_size, aabb_host_buffer_.data(),
      &output_vector);
  output_timer.Stop();

  return output_vector;
}

// Camera
std::vector<Index3D> ViewCalculator::getBlocksInImageViewRaycast(
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    const float block_size,
    const float max_integration_distance_behind_surface_m,
    const float max_integration_distance_m) {
  return getBlocksInImageViewRaycastTemplate(
      depth_frame, T_L_C, camera, block_size,
      max_integration_distance_behind_surface_m, max_integration_distance_m);
}

// Lidar
std::vector<Index3D> ViewCalculator::getBlocksInImageViewRaycast(
    const DepthImage& depth_frame, const Transform& T_L_C, const Lidar& lidar,
    const float block_size,
    const float max_integration_distance_behind_surface_m,
    const float max_integration_distance_m) {
  return getBlocksInImageViewRaycastTemplate(
      depth_frame, T_L_C, lidar, block_size,
      max_integration_distance_behind_surface_m, max_integration_distance_m);
}

template <typename SensorType>
void ViewCalculator::getBlocksByRaycastingCorners(
    const Transform& T_L_C, const SensorType& camera,
    const DepthImage& depth_frame, float block_size,
    const float max_integration_distance_behind_surface_m,
    const float max_integration_distance_m, const Index3D& min_index,
    const Index3D& aabb_size, bool* aabb_updated_cuda) {
  // Get the blocks touched by the ray endpoints
  // We'll do warps of 32x32 pixels in the image. This is 1024 threads which is
  // in the recommended 512-1024 range.
  constexpr int kThreadDim = 16;
  int rounded_rows = static_cast<int>(
      std::ceil(depth_frame.rows() / static_cast<float>(kThreadDim)));
  int rounded_cols = static_cast<int>(
      std::ceil(depth_frame.cols() / static_cast<float>(kThreadDim)));
  dim3 block_dim(rounded_rows, rounded_cols);
  dim3 thread_dim(kThreadDim, kThreadDim);

  timing::Timer image_blocks_timer("view_calculator/raycast/get_image_blocks");
  getBlockIndicesInImageKernel<<<block_dim, thread_dim, 0, *cuda_stream_>>>(
      T_L_C, camera, depth_frame.dataConstPtr(), depth_frame.rows(),
      depth_frame.cols(), block_size, max_integration_distance_m,
      max_integration_distance_behind_surface_m, min_index, aabb_size,
      aabb_updated_cuda);
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  image_blocks_timer.Stop();

  timing::Timer image_blocks_copy_timer(
      "view_calculator/raycast/image_blocks_copy");

  unified_vector<Index3D> initial_vector;
  const size_t aabb_linear_size = aabb_size.x() * aabb_size.y() * aabb_size.z();
  initial_vector.reserve(aabb_linear_size / 3);
  convertAabbUpdatedToVector<unified_vector<Index3D>>(
      min_index, aabb_size, aabb_linear_size, aabb_updated_cuda,
      &initial_vector);
  image_blocks_copy_timer.Stop();

  // Call the kernel to do raycasting.
  timing::Timer raycast_blocks_timer("view_calculator/raycast/raycast_kernel");

  int num_initial_blocks = initial_vector.size();
  constexpr int kNumCorners = 9;
  constexpr int kNumBlocksPerThreadBlock = 40;
  int raycast_block_dim = static_cast<int>(std::ceil(
      static_cast<float>(num_initial_blocks) / kNumBlocksPerThreadBlock));
  dim3 raycast_thread_dim(kNumBlocksPerThreadBlock, kNumCorners);
  raycastToBlocksKernel<<<raycast_block_dim, raycast_thread_dim, 0,
                          *cuda_stream_>>>(
      num_initial_blocks, initial_vector.data(), T_L_C, block_size, min_index,
      aabb_size, aabb_updated_cuda);
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
  raycast_blocks_timer.Stop();
}

template <typename SensorType>
void ViewCalculator::getBlocksByRaycastingPixels(
    const Transform& T_L_C, const SensorType& camera,
    const DepthImage& depth_frame, float block_size,
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

  // We'll do warps of 32x32 pixels in the image. This is 1024 threads which is
  // in the recommended 512-1024 range.
  constexpr int kThreadDim = 16;
  const int rounded_rows = static_cast<int>(
      std::ceil(num_subsampled_rows / static_cast<float>(kThreadDim)));
  const int rounded_cols = static_cast<int>(
      std::ceil(num_subsampled_cols / static_cast<float>(kThreadDim)));
  dim3 block_dim(rounded_rows, rounded_cols);
  dim3 thread_dim(kThreadDim, kThreadDim);

  combinedBlockIndicesInImageKernel<<<block_dim, thread_dim, 0,
                                      *cuda_stream_>>>(
      T_L_C, camera, depth_frame.dataConstPtr(), depth_frame.rows(),
      depth_frame.cols(), block_size, max_integration_distance_m,
      max_integration_distance_behind_surface_m, raycast_subsampling_factor_,
      min_index, aabb_size, aabb_updated_cuda);
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
  combined_kernel_timer.Stop();
}

std::vector<Index3D> ViewCalculator::getBlocksInViewPlanes(
    const Transform& T_L_C, const Camera& camera, const float block_size,
    const float max_distance) {
  CHECK_GT(max_distance, 0.0f);
  timing::Timer("view_calculator/get_blocks_in_view_planes");

  // Project all block centers into the image and check if they are
  // inside the image viewport.

  // View frustum with small positive min distance to avoid div-by-zero
  constexpr float kMinDistance = 1E-6f;
  const Frustum frustum =
      camera.getViewFrustum(T_L_C, kMinDistance, max_distance);

  // Coarse bound: AABB
  const AxisAlignedBoundingBox aabb_L = frustum.getAABB();
  const std::vector<Index3D> block_indices_in_aabb =
      getBlockIndicesTouchedByBoundingBox(block_size, aabb_L);

  // Get the 2D viewport of the camera. We use normalized image
  // coordinates rather than pixels to avoid having to apply the
  // camera intrinsics to each point we want to check. A small margin
  // is added to also capture blocks which intersect a frustum plane
  // but have their center point outside the plane.
  constexpr float kMargin{10.F};
  CameraViewport normalized_viewport = camera.getNormalizedViewport(kMargin);

  // Get the transform to camera from layer. To save some extra
  // cycles, we extract the rotation and translation components rather
  // than multiplying with the whole 4x4 matrix.
  const Transform T_C_L = T_L_C.inverse();
  const Eigen::Matrix3f rotation_C_L = T_C_L.rotation();
  const Eigen::Vector3f translation_C_L = T_C_L.translation();

  std::vector<Index3D> block_indices_in_frustum;
  for (const Index3D& block_index : block_indices_in_aabb) {
    // Transform the block center into camera frame
    const Eigen::Vector3f p3d_layer =
        getCenterPositionFromBlockIndex(block_size, block_index);
    const Eigen::Vector3f p3d_cam = rotation_C_L * p3d_layer + translation_C_L;

    if (p3d_cam[2] > kMinDistance) {
      // Project into normalized camera coordinates
      Eigen::Vector2f p2d_normalized_cam;
      camera.projectToNormalizedCoordinates(p3d_cam, &p2d_normalized_cam);

      // Check if the projected point is inside the viewport
      if (normalized_viewport.contains(p2d_normalized_cam)) {
        block_indices_in_frustum.push_back(block_index);
      }
    }
  }
  return block_indices_in_frustum;
}

std::vector<Index3D> ViewCalculator::getBlocksInImageViewPlanes(
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    const float block_size, const float truncation_distance_m,
    const float max_integration_distance_m) {
  timing::Timer("view_calculator/get_blocks_in_image_view_planes");
  float min_depth, max_depth;
  std::tie(min_depth, max_depth) = image::minmaxGPU(depth_frame);
  float max_depth_plus_trunc = max_depth + truncation_distance_m;
  if (max_integration_distance_m > 0.0f) {
    max_depth_plus_trunc =
        std::min<float>(max_depth_plus_trunc, max_integration_distance_m);
  }
  return getBlocksInViewPlanes(T_L_C, camera, block_size, max_depth_plus_trunc);
}

}  // namespace nvblox