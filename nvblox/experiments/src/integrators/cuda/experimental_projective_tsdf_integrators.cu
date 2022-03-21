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
#include "nvblox/experiments/integrators/experimental_projective_tsdf_integrators.h"

#include "nvblox/experiments/integrators/cuda/experimental_integrator_input_frames.cuh"

namespace nvblox {
namespace experiments {

__device__ inline float interpolateDepthTexture(
    cudaTextureObject_t depth_texture, const Eigen::Vector2f& u_px) {
  return tex2D<float>(depth_texture, u_px.x() + 0.5, u_px.y() + 0.5);
}

__device__ inline bool interpolateDepthImage(const float* image, int rows,
                                             int cols, Eigen::Vector2f u_px,
                                             float* value_ptr) {
  // If the projected point does not lie on the image plane, fail. (Here "on the
  // image plane" means having pixel centers surrounding the query point, ie no
  // extrapolation).
  if ((u_px.x() < 0.0f) || (u_px.y() < 0.0f) ||
      (u_px.x() > static_cast<float>(cols) - 1.0f) ||
      (u_px.y() > static_cast<float>(rows) - 1.0f)) {
    return false;
  }
  // Interpolation of a grid on with 1 pixel spacing.
  // https://en.wikipedia.org/wiki/Bilinear_interpolation#On_the_unit_square
  // Get the pixel coordinates of the pixel on the low side
  const Index2D u_low_side_px = (u_px).cast<int>();
  // Get the 4-neighbours values and put them in a matrix
  // clang-format off
  const Eigen::Matrix2f value_matrix =
      (Eigen::Matrix2f() <<
        image::access(u_low_side_px.y(), u_low_side_px.x(), cols, image),
        image::access(u_low_side_px.y() + 1, u_low_side_px.x(), cols, image),
        image::access(u_low_side_px.y(), u_low_side_px.x() + 1, cols, image),
        image::access(u_low_side_px.y() + 1, u_low_side_px.x() + 1, cols, image))
        .finished();
  // clang-format on
  // Offset of the requested point to the low side center.
  const Eigen::Vector2f u_offset = (u_px - u_low_side_px.cast<float>());
  const Eigen::Vector2f x_vec(1.0f - u_offset.x(), u_offset.x());
  const Eigen::Vector2f y_vec(1.0f - u_offset.y(), u_offset.y());
  *value_ptr = x_vec.transpose() * value_matrix * y_vec;
  return true;
}

__global__ void intergrateBlocksTextureBasedInterpolation(
    const Index3D* block_indices_device_ptr, const Camera* camera_device_ptr,
    cudaTextureObject_t depth_texture, const Eigen::Matrix3f* R_C_L_device_ptr,
    const Eigen::Vector3f* t_C_L_device_ptr, const float block_size,
    const float truncation_distance_m, const float max_weight,
    VoxelBlock<TsdfVoxel>** block_device_ptrs) {
  // Linear index of thread within block
  const int thread_index_linear =
      threadIdx.x + blockDim.x * (threadIdx.y + (blockDim.y * threadIdx.z));

  // Get the data which is common between all threads in a block into shared
  // memory
  // TODO(alexmillane): We could also get the camera into shared memory. But
  // maybe let's profile things first and see what is actually affecting the
  // performance.
  __shared__ Eigen::Matrix3f R_C_L;
  if (thread_index_linear < 9) {
    R_C_L.data()[thread_index_linear] =
        R_C_L_device_ptr->data()[thread_index_linear];
  }
  __shared__ Eigen::Vector3f t_C_L;
  if (thread_index_linear >= 9 && thread_index_linear < 12) {
    t_C_L.data()[thread_index_linear - 9] =
        t_C_L_device_ptr->data()[thread_index_linear - 9];
  }
  __syncthreads();

  // The indices of the voxel this thread will work on
  // blockIdx.x      - The index of the block we're working on (blockIdx.y/z
  //                   should be zero)
  // threadIdx.x/y/z - The indices of the voxel within the block (we
  //                   expect the threadBlockDims == voxelBlockDims)
  const Index3D block_idx = block_indices_device_ptr[blockIdx.x];
  const Index3D voxel_idx(threadIdx.z, threadIdx.y, threadIdx.x);

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order such
  // that adjacent threads (x-major) access adjacent memory locations in the
  // block (z-major).
  TsdfVoxel* voxel_ptr =
      &(block_device_ptrs[blockIdx.x]
            ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Voxel center point
  const Vector3f p_voxel_center_L = getCenterPostionFromBlockIndexAndVoxelIndex(
      block_size, block_idx, voxel_idx);
  // To camera frame
  const Vector3f p_voxel_center_C = R_C_L * p_voxel_center_L + t_C_L;

  // Project to image plane
  Eigen::Vector2f u_px;
  if (!camera_device_ptr->project(p_voxel_center_C, &u_px)) {
    return;
  }

  // If the projected point does not lie on the image plane, fail. (Here "on the
  // image plane" means having pixel centers surrounding the query point, ie no
  // extrapolation).
  if ((u_px.x() < 0.0f) || (u_px.y() < 0.0f) ||
      (u_px.x() > static_cast<float>(camera_device_ptr->width()) - 1.0f) ||
      (u_px.y() > static_cast<float>(camera_device_ptr->height()) - 1.0f)) {
    return;
  }

  // Get the MEASURED depth of the SURFACE, by interpolating the depth image
  const float surface_depth_mesured =
      interpolateDepthTexture(depth_texture, u_px);

  // Get the MEASURED depth of the VOXEL
  const float voxel_distance_measured =
      surface_depth_mesured - p_voxel_center_C.z();

  // If we're behind the negative truncation distance, just continue.
  if (voxel_distance_measured < -truncation_distance_m) {
    return;
  }

  // Read CURRENT voxel values (from global GPU memory)
  const float voxel_distance_current = voxel_ptr->distance;
  const float voxel_weight_current = voxel_ptr->weight;

  // NOTE(alexmillane): We could try to use CUDA math functions to speed up
  // below
  // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE

  // Fuse
  constexpr float measurement_weight = 1.0f;
  const float fused_distance = (voxel_distance_measured * measurement_weight +
                                voxel_distance_current * voxel_weight_current) /
                               (measurement_weight + voxel_weight_current);

  // Write back to voxel (to global GPU memory)
  voxel_ptr->distance = fused_distance > 0.0f
                            ? fmin(truncation_distance_m, fused_distance)
                            : fmax(-truncation_distance_m, fused_distance);
  voxel_ptr->weight =
      fmin(measurement_weight + voxel_weight_current, max_weight);
}

__global__ void intergrateBlocksGlobalBasedInterpolation(
    const Index3D* block_indices_device_ptr, const Camera* camera_device_ptr,
    const float* image, int rows, int cols,
    const Eigen::Matrix3f* R_C_L_device_ptr,
    const Eigen::Vector3f* t_C_L_device_ptr, const float block_size,
    const float truncation_distance_m, const float max_weight,
    VoxelBlock<TsdfVoxel>** block_device_ptrs) {
  // Linear index of thread within block
  const int thread_index_linear =
      threadIdx.x + blockDim.x * (threadIdx.y + (blockDim.y * threadIdx.z));

  // Get the data which is common between all threads in a block into shared
  // memory
  // TODO(alexmillane): We could also get the camera into shared memory. But
  // maybe let's profile things first and see what is actually affecting the
  // performance.
  __shared__ Eigen::Matrix3f R_C_L;
  if (thread_index_linear < 9) {
    R_C_L.data()[thread_index_linear] =
        R_C_L_device_ptr->data()[thread_index_linear];
  }
  __shared__ Eigen::Vector3f t_C_L;
  if (thread_index_linear >= 9 && thread_index_linear < 12) {
    t_C_L.data()[thread_index_linear - 9] =
        t_C_L_device_ptr->data()[thread_index_linear - 9];
  }
  __syncthreads();

  // The indices of the voxel this thread will work on
  // blockIdx.x      - The index of the block we're working on (blockIdx.y/z
  //                   should be zero)
  // threadIdx.x/y/z - The indices of the voxel within the block (we
  //                   expect the threadBlockDims == voxelBlockDims)
  const Index3D block_idx = block_indices_device_ptr[blockIdx.x];
  const Index3D voxel_idx(threadIdx.z, threadIdx.y, threadIdx.x);

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order such
  // that adjacent threads (x-major) access adjacent memory locations in the
  // block (z-major).
  TsdfVoxel* voxel_ptr =
      &(block_device_ptrs[blockIdx.x]
            ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Voxel center point
  const Vector3f p_voxel_center_L = getCenterPostionFromBlockIndexAndVoxelIndex(
      block_size, block_idx, voxel_idx);
  // To camera frame
  const Vector3f p_voxel_center_C = R_C_L * p_voxel_center_L + t_C_L;

  // Project to image plane
  Eigen::Vector2f u_px;
  if (!camera_device_ptr->project(p_voxel_center_C, &u_px)) {
    return;
  }

  // Get the MEASURED depth of the SURFACE, by interpolating the depth image
  float surface_depth_mesured;
  if (!interpolateDepthImage(image, rows, cols, u_px, &surface_depth_mesured)) {
    return;
  }

  // Get the MEASURED depth of the VOXEL
  const float voxel_distance_measured =
      surface_depth_mesured - p_voxel_center_C.z();

  // If we're behind the negative truncation distance, just continue.
  if (voxel_distance_measured < -truncation_distance_m) {
    return;
  }

  // Read CURRENT voxel values (from global GPU memory)
  const float voxel_distance_current = voxel_ptr->distance;
  const float voxel_weight_current = voxel_ptr->weight;

  // NOTE(alexmillane): We could try to use CUDA math functions to speed up
  // below
  // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE

  // Fuse
  constexpr float measurement_weight = 1.0f;
  const float fused_distance = (voxel_distance_measured * measurement_weight +
                                voxel_distance_current * voxel_weight_current) /
                               (measurement_weight + voxel_weight_current);

  // Write back to voxel (to global GPU memory)
  voxel_ptr->distance = fused_distance > 0.0f
                            ? fmin(truncation_distance_m, fused_distance)
                            : fmax(-truncation_distance_m, fused_distance);
  voxel_ptr->weight =
      fmin(measurement_weight + voxel_weight_current, max_weight);
}

ProjectiveTsdfIntegratorExperimentsBase::
    ProjectiveTsdfIntegratorExperimentsBase()
    : ProjectiveTsdfIntegrator() {
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}

ProjectiveTsdfIntegratorExperimentsBase::
    ~ProjectiveTsdfIntegratorExperimentsBase() {
  finish();
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

void ProjectiveTsdfIntegratorExperimentsBase::finish() const {
  cudaStreamSynchronize(integration_stream_);
}

void ProjectiveTsdfIntegratorExperimentsTexture::updateBlocks(
    const std::vector<Index3D>& block_indices, const DepthImage& depth_frame,
    const Transform& T_L_C, const Camera& camera,
    const float truncation_distance_m, TsdfLayer* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);

  // Create an integrator frame
  // Internally this object starts (asynchronous) transfers of it's inputs to
  // device memory. Kernels called the passed stream can therefore utilize the
  // input frame's device-side members.
  const IntegratorInputFrameExperimentsTexture input(
      block_indices, depth_frame, T_L_C, camera, truncation_distance_m,
      max_weight_, layer_ptr, integration_stream_);

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_blocks = input.num_blocks;
  // clang-format off
  intergrateBlocksTextureBasedInterpolation<<<num_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      input.block_indices_device_ptr,
      input.camera_device_ptr,
      input.depth_texture.texture_object(),
      input.R_C_L_device_ptr,
      input.t_C_L_device_ptr,
      input.block_size, 
      input.truncation_distance_m,
      input.max_weight,
      input.block_device_ptrs);
  // clang-format on
  checkCudaErrors(cudaPeekAtLastError());

  // Finish processing of the frame before returning control
  finish();
}

void ProjectiveTsdfIntegratorExperimentsGlobal::updateBlocks(
    const std::vector<Index3D>& block_indices, const DepthImage& depth_frame,
    const Transform& T_L_C, const Camera& camera,
    const float truncation_distance_m, TsdfLayer* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);

  // Create an integrator frame
  // Internally this object starts (asynchronous) transfers of it's inputs to
  // device memory. Kernels called the passed stream can therefore utilize the
  // input frame's device-side members.
  const IntegratorInputFrameExperimentsGlobal input(
      block_indices, depth_frame, T_L_C, camera, truncation_distance_m,
      max_weight_, layer_ptr, integration_stream_);

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_blocks = input.num_blocks;
  // clang-format off
  intergrateBlocksGlobalBasedInterpolation<<<num_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      input.block_indices_device_ptr,
      input.camera_device_ptr,
      input.depth_frame_unified_ptr,
      input.depth_frame_rows,
      input.depth_frame_cols,
      input.R_C_L_device_ptr,
      input.t_C_L_device_ptr,
      input.block_size, 
      input.truncation_distance_m,
      input.max_weight,
      input.block_device_ptrs);
  // clang-format on
  checkCudaErrors(cudaPeekAtLastError());

  // Finish processing of the frame before returning control
  finish();
}

}  // namespace experiments
}  // namespace nvblox
