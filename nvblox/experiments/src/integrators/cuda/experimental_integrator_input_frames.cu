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
#include "nvblox/experiments/integrators/cuda/experimental_integrator_input_frames.cuh"

#include "nvblox/integrators/integrators_common.h"

namespace nvblox {
namespace experiments {

IntegratorInputFrameExperimentsBase::IntegratorInputFrameExperimentsBase(
    const std::vector<Index3D>& block_indices, const DepthImage& depth_frame,
    const Transform& T_L_C, const Camera& camera,
    const float _truncation_distance_m, const float _max_weight,
    TsdfLayer* layer_ptr, cudaStream_t stream)
    : block_size(layer_ptr->block_size()),
      truncation_distance_m(_truncation_distance_m),
      max_weight(_max_weight),
      num_blocks(block_indices.size()),
      R_C_L(T_L_C.inverse().rotation()),
      t_C_L(T_L_C.inverse().translation()),
      block_ptrs(getBlockPtrsFromIndices(block_indices, layer_ptr)),
      transfer_stream(stream) {
  // Allocate GPU memory
  checkCudaErrors(cudaMalloc(&R_C_L_device_ptr, sizeof(Eigen::Matrix3f)));
  checkCudaErrors(cudaMalloc(&t_C_L_device_ptr, sizeof(Eigen::Vector3f)));
  checkCudaErrors(cudaMalloc(&camera_device_ptr, sizeof(Camera)));
  checkCudaErrors(cudaMalloc(&block_indices_device_ptr,
                             block_indices.size() * sizeof(Index3D)));
  checkCudaErrors(
      cudaMalloc(&block_device_ptrs,
                 block_indices.size() * sizeof(VoxelBlock<TsdfVoxel>*)));

  // Host -> GPU transfers
  checkCudaErrors(cudaMemcpyAsync(block_indices_device_ptr,
                                  block_indices.data(),
                                  block_indices.size() * sizeof(Index3D),
                                  cudaMemcpyHostToDevice, transfer_stream));
  checkCudaErrors(cudaMemcpyAsync(R_C_L_device_ptr, R_C_L.data(),
                                  sizeof(Eigen::Matrix3f),
                                  cudaMemcpyHostToDevice, transfer_stream));
  checkCudaErrors(cudaMemcpyAsync(t_C_L_device_ptr, t_C_L.data(),
                                  sizeof(Eigen::Vector3f),
                                  cudaMemcpyHostToDevice, transfer_stream));
  checkCudaErrors(cudaMemcpyAsync(camera_device_ptr, &camera, sizeof(Camera),
                                  cudaMemcpyHostToDevice, transfer_stream));
  checkCudaErrors(
      cudaMemcpyAsync(block_device_ptrs, block_ptrs.data(),
                      block_ptrs.size() * sizeof(VoxelBlock<TsdfVoxel>*),
                      cudaMemcpyHostToDevice, transfer_stream));
}

IntegratorInputFrameExperimentsBase::~IntegratorInputFrameExperimentsBase() {
  cudaStreamSynchronize(transfer_stream);
  checkCudaErrors(cudaFree(block_indices_device_ptr));
  checkCudaErrors(cudaFree(camera_device_ptr));
  checkCudaErrors(cudaFree(R_C_L_device_ptr));
  checkCudaErrors(cudaFree(t_C_L_device_ptr));
  checkCudaErrors(cudaFree(block_device_ptrs));
}

IntegratorInputFrameExperimentsTexture::IntegratorInputFrameExperimentsTexture(
    const std::vector<Index3D>& block_indices, const DepthImage& depth_frame,
    const Transform& T_L_C, const Camera& camera,
    const float truncation_distance_m, const float max_weight,
    TsdfLayer* layer_ptr, cudaStream_t stream)
    : IntegratorInputFrameExperimentsBase(block_indices, depth_frame, T_L_C,
                                          camera, truncation_distance_m,
                                          max_weight, layer_ptr, stream),
      depth_texture(depth_frame, stream){
          // The base class handles all the other transfers.
      };

IntegratorInputFrameExperimentsGlobal::IntegratorInputFrameExperimentsGlobal(
    const std::vector<Index3D>& block_indices, const DepthImage& depth_frame,
    const Transform& T_L_C, const Camera& camera,
    const float truncation_distance_m, const float max_weight,
    TsdfLayer* layer_ptr, cudaStream_t stream)
    : IntegratorInputFrameExperimentsBase(block_indices, depth_frame, T_L_C,
                                          camera, truncation_distance_m,
                                          max_weight, layer_ptr, stream),
      depth_frame_unified_ptr(depth_frame.dataConstPtr()),
      depth_frame_rows(depth_frame.rows()),
      depth_frame_cols(depth_frame.cols()) {
  // Send the depth-frame to GPU
  depth_frame.toGPU();
};

}  // namespace experiments
}  // namespace nvblox
