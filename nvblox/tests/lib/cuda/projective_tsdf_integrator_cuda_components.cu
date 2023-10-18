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
#include "nvblox/tests/projective_tsdf_integrator_cuda_components.h"

#include "nvblox/core/internal/error_check.h"
#include "nvblox/interpolation/interpolation_2d.h"
#include "nvblox/map/blox.h"

namespace nvblox {
namespace test_utils {

__global__ void transformPointsOnGPU(const Eigen::Matrix3f* R_B_A_matrix_ptr,
                                     const Eigen::Vector3f* t_B_A_matrix_ptr,
                                     const float* vecs_A_ptr,
                                     const int num_vecs, float* vecs_B_ptr) {
  // We first load the transform into shared memory for use by all threads in
  // the block. The transformation matrix has 4x4=16 elements, so the first 16
  // threads of each block perform the load.
  __shared__ Eigen::Matrix3f R_B_A;
  if (threadIdx.x < 9) {
    R_B_A.data()[threadIdx.x] = R_B_A_matrix_ptr->data()[threadIdx.x];
  }
  __shared__ Eigen::Vector3f t_B_A;
  if (threadIdx.x >= 9 && threadIdx.x < 12) {
    t_B_A.data()[threadIdx.x - 9] = t_B_A_matrix_ptr->data()[threadIdx.x - 9];
  }
  __syncthreads();

  // Now perform transformation of the vectors.
  const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec_idx < num_vecs) {
    // Mapping the vecs
    const Eigen::Map<const Eigen::Matrix3Xf> vecs_A(vecs_A_ptr, 3, num_vecs);
    Eigen::Map<Eigen::Matrix3Xf> vecs_B(vecs_B_ptr, 3, num_vecs);
    // Transformation
    vecs_B.col(vec_idx) = R_B_A * vecs_A.col(vec_idx) + t_B_A;
  }
}

__global__ void projectBlocksToCamera(
    const Index3D* block_indices_device_ptr, const Camera* camera_device_ptr,
    const Eigen::Matrix3f* R_C_L_device_ptr,
    const Eigen::Vector3f* t_C_L_device_ptr, const float block_size,
    BlockProjectionResult* block_projection_results_device_ptr) {
  // Linear index of thread within block
  const int thread_index_linear =
      threadIdx.z + blockDim.z * (threadIdx.y + (blockDim.y * threadIdx.x));

  // Get data needed by all threads into shared memory
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
  const Index3D voxel_idx(threadIdx.x, threadIdx.y, threadIdx.z);

  // Voxel center point
  const Vector3f p_voxel_center_L =
      getCenterPositionFromBlockIndexAndVoxelIndex(block_size, block_idx,
                                                   voxel_idx);
  // To camera frame
  const Vector3f p_voxel_center_C = R_C_L * p_voxel_center_L + t_C_L;
  // Project to image plane
  Eigen::Vector2f u_px;
  if (!camera_device_ptr->project(p_voxel_center_C, &u_px)) {
    return;
  }

  // Map outputs
  BlockProjectionResult* result_ptr =
      &(block_projection_results_device_ptr[blockIdx.x]);
  result_ptr->row(thread_index_linear) = u_px;
}

__global__ void interpolate(const float* depth_frame,
                            const float* u_px_vec_device_ptr, const int rows,
                            const int cols, const int num_points,
                            float* interpolated_value_device_ptr) {
  // Map the interpolation locations
  const Eigen::Map<const Eigen::MatrixX2f> u_px_vec(u_px_vec_device_ptr,
                                                    num_points, 2);
  // Interpolate one of the points
  const int u_px_vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (u_px_vec_idx < num_points) {
    interpolation::interpolate2DLinear(
        depth_frame, u_px_vec.row(u_px_vec_idx), rows, cols,
        &interpolated_value_device_ptr[u_px_vec_idx]);
  }
}

__global__ void setVoxelBlock(VoxelBlock<TsdfVoxel>** block_device_ptrs) {
  // The VoxelBlock that this ThreadBlock is working on
  VoxelBlock<TsdfVoxel>* block_ptr = block_device_ptrs[blockIdx.x];
  block_ptr->voxels[threadIdx.z][threadIdx.y][threadIdx.x].distance =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

void setVoxelBlockOnGPU(TsdfLayer* layer) {
  // Get a list of blocks to be modified on the CPU
  const std::vector<Index3D> block_indices = layer->getAllBlockIndices();
  std::vector<VoxelBlock<TsdfVoxel>*> block_ptrs;
  block_ptrs.reserve(block_indices.size());
  for (const Index3D& block_index : block_indices) {
    block_ptrs.push_back(layer->getBlockAtIndex(block_index).get());
  }

  // Move the list to the GPU
  VoxelBlock<TsdfVoxel>** block_device_ptrs;
  checkCudaErrors(cudaMalloc(
      &block_device_ptrs, block_ptrs.size() * sizeof(VoxelBlock<TsdfVoxel>*)));
  checkCudaErrors(cudaMemcpy(block_device_ptrs, block_ptrs.data(),
                             block_ptrs.size() * sizeof(VoxelBlock<TsdfVoxel>*),
                             cudaMemcpyHostToDevice));

  // Kernal - One thread block per block
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_blocks = block_indices.size();
  setVoxelBlock<<<num_blocks, kThreadsPerBlock>>>(block_device_ptrs);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(block_device_ptrs));
}

Eigen::VectorXf interpolatePointsOnGPU(const DepthImage& depth_frame,
                                       const Eigen::MatrixX2f& u_px_vec) {
  // Transfer data to the GPU
  float* depth_frame_device_ptr;
  checkCudaErrors(
      cudaMalloc(&depth_frame_device_ptr, depth_frame.numel() * sizeof(float)));
  checkCudaErrors(cudaMemcpy(depth_frame_device_ptr, depth_frame.dataConstPtr(),
                             depth_frame.numel() * sizeof(float),
                             cudaMemcpyHostToDevice));
  float* u_px_vec_device_ptr;
  checkCudaErrors(
      cudaMalloc(&u_px_vec_device_ptr, u_px_vec.rows() * 2 * sizeof(float)));
  checkCudaErrors(cudaMemcpy(u_px_vec_device_ptr, u_px_vec.data(),
                             u_px_vec.rows() * 2 * sizeof(float),
                             cudaMemcpyHostToDevice));

  // Output location
  float* interpolated_values_device_ptr;
  checkCudaErrors(cudaMalloc(&interpolated_values_device_ptr,
                             u_px_vec.rows() * sizeof(float)));

  // Kernel - interpolation
  const int num_points = u_px_vec.rows();
  constexpr int threadsPerBlock = 512;
  const int blocksInGrid = (num_points / threadsPerBlock) + 1;
  interpolate<<<blocksInGrid, threadsPerBlock>>>(
      depth_frame_device_ptr, u_px_vec_device_ptr, depth_frame.rows(),
      depth_frame.cols(), u_px_vec.rows(), interpolated_values_device_ptr);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Return the result
  Eigen::VectorXf results(u_px_vec.rows());
  checkCudaErrors(cudaMemcpy(results.data(), interpolated_values_device_ptr,
                             u_px_vec.rows() * sizeof(float),
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(depth_frame_device_ptr));
  checkCudaErrors(cudaFree(u_px_vec_device_ptr));
  checkCudaErrors(cudaFree(interpolated_values_device_ptr));

  return results;
}

std::vector<BlockProjectionResult> projectBlocksOnGPU(
    const std::vector<Index3D>& block_indices, const Camera& camera,
    const Transform& T_C_L, TsdfLayer* distance_layer_ptr) {
  // Camera
  Camera* camera_device_ptr;
  checkCudaErrors(cudaMalloc(&camera_device_ptr, sizeof(Camera)));
  checkCudaErrors(cudaMemcpy(camera_device_ptr, &camera, sizeof(Camera),
                             cudaMemcpyHostToDevice));

  // Transformation
  // NOTE(alexmillane): For some reason I only got things to work by separating
  // the Eigen::Affine3f into the rotation matrix and translation vector... I
  // cannot explain why it didn't work, but I spent hours trying to get it and I
  // couldn't.
  const Eigen::Matrix3f R_C_L = T_C_L.rotation();
  const Eigen::Vector3f t_C_L = T_C_L.translation();
  Eigen::Matrix3f* R_C_L_device_ptr;
  Eigen::Vector3f* t_C_L_device_ptr;
  checkCudaErrors(cudaMalloc(&R_C_L_device_ptr, sizeof(Eigen::Matrix3f)));
  checkCudaErrors(cudaMemcpy(R_C_L_device_ptr, R_C_L.data(),
                             sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&t_C_L_device_ptr, sizeof(Eigen::Vector3f)));
  checkCudaErrors(cudaMemcpy(t_C_L_device_ptr, t_C_L.data(),
                             sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));

  // Copy the block indices to the GPU for projection
  Index3D* block_indices_device_ptr;
  checkCudaErrors(cudaMalloc(&block_indices_device_ptr,
                             block_indices.size() * sizeof(Index3D)));
  checkCudaErrors(cudaMemcpy(block_indices_device_ptr, block_indices.data(),
                             block_indices.size() * sizeof(Index3D),
                             cudaMemcpyHostToDevice));

  // Output space
  BlockProjectionResult* block_projection_results_device_ptr;
  checkCudaErrors(
      cudaMalloc(&block_projection_results_device_ptr,
                 block_indices.size() * sizeof(BlockProjectionResult)));

  // TODO: CURRENTLY ASSUMES WE CAN LAUNCH AN INFINITE NUMBER OF THREAD BLOX
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_blocks = block_indices.size();

  projectBlocksToCamera<<<num_blocks, kThreadsPerBlock>>>(
      block_indices_device_ptr, camera_device_ptr, R_C_L_device_ptr,
      t_C_L_device_ptr, distance_layer_ptr->block_size(),
      block_projection_results_device_ptr);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Copy over results
  std::vector<BlockProjectionResult> projection_results;
  projection_results.resize(block_indices.size());
  checkCudaErrors(
      cudaMemcpy(projection_results.data(), block_projection_results_device_ptr,
                 block_indices.size() * sizeof(BlockProjectionResult),
                 cudaMemcpyDeviceToHost));

  // Free
  checkCudaErrors(cudaFree(R_C_L_device_ptr));
  checkCudaErrors(cudaFree(t_C_L_device_ptr));
  checkCudaErrors(cudaFree(block_indices_device_ptr));
  checkCudaErrors(cudaFree(block_projection_results_device_ptr));

  return projection_results;
}

Eigen::Matrix3Xf transformPointsOnGPU(const Transform& T_B_A,
                                      const Eigen::Matrix3Xf& vecs_A) {
  // Move inputs
  float* vecs_A_device_ptr;
  const int num_elements = vecs_A.rows() * vecs_A.cols();
  checkCudaErrors(cudaMalloc(&vecs_A_device_ptr, num_elements * sizeof(float)));
  checkCudaErrors(cudaMemcpy(vecs_A_device_ptr, vecs_A.data(),
                             num_elements * sizeof(float),
                             cudaMemcpyHostToDevice));
  // Transformation
  // NOTE(alexmillane): For some reason I only got things to work by separating
  // the Eigen::Affine3f into the rotation matrix and translation vector... I
  // cannot explain why it didn't work, but I spent hours trying to get it and I
  // couldn't.
  const Eigen::Matrix3f R_B_A = T_B_A.rotation();
  const Eigen::Vector3f t_B_A = T_B_A.translation();
  Eigen::Matrix3f* R_A_B_device_ptr;
  Eigen::Vector3f* t_A_B_device_ptr;
  checkCudaErrors(cudaMalloc(&R_A_B_device_ptr, sizeof(Eigen::Matrix3f)));
  checkCudaErrors(cudaMemcpy(R_A_B_device_ptr, R_B_A.data(),
                             sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&t_A_B_device_ptr, sizeof(Eigen::Vector3f)));
  checkCudaErrors(cudaMemcpy(t_A_B_device_ptr, t_B_A.data(),
                             sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));

  // Output space
  float* vecs_B_device;
  checkCudaErrors(cudaMalloc(&vecs_B_device, num_elements * sizeof(float)));

  // Kernel
  const int num_vecs = vecs_A.cols();
  constexpr int threadsPerBlock = 512;
  const int blocksInGrid = (num_vecs / threadsPerBlock) + 1;
  transformPointsOnGPU<<<blocksInGrid, threadsPerBlock>>>(
      R_A_B_device_ptr, t_A_B_device_ptr, vecs_A_device_ptr, num_vecs,
      vecs_B_device);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Retrieve output
  Eigen::Matrix3Xf vecs_B(3, num_vecs);
  checkCudaErrors(cudaMemcpy(vecs_B.data(), vecs_B_device,
                             num_elements * sizeof(float),
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(vecs_A_device_ptr));
  checkCudaErrors(cudaFree(R_A_B_device_ptr));
  checkCudaErrors(cudaFree(t_A_B_device_ptr));
  checkCudaErrors(cudaFree(vecs_B_device));

  return vecs_B;
}

}  // namespace test_utils
}  // namespace nvblox
