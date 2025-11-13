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

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/map/layer.h"
#include "nvblox/utils/cuda_kernel_utils.h"

namespace nvblox {

template <typename VoxelType>
__global__ void queryVoxelsKernel(
    int num_queries, Index3DDeviceHashMapType<VoxelBlock<VoxelType>> block_hash,
    float block_size, const Vector3f* query_locations_ptr,
    VoxelType* voxels_ptr, bool* success_flags_ptr) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_queries) {
    return;
  }
  const Vector3f query_location = query_locations_ptr[idx];

  VoxelType* voxel;
  if (!getVoxelAtPosition<VoxelType>(block_hash, query_location, block_size,
                                     &voxel)) {
    success_flags_ptr[idx] = false;
  } else {
    success_flags_ptr[idx] = true;
    voxels_ptr[idx] = *voxel;
  }
}

template <typename VoxelType>
void VoxelBlockLayer<VoxelType>::getVoxelsGPU(
    const device_vector<Vector3f>& positions_L,
    device_vector<VoxelType>* voxels_ptr,
    device_vector<bool>* success_flags_ptr) const {
  // Call the underlying streamed method on a newly created stream.
  CudaStreamOwning cuda_stream;
  getVoxelsGPU(positions_L, voxels_ptr, success_flags_ptr, &cuda_stream);
}

template <typename VoxelType>
void VoxelBlockLayer<VoxelType>::getVoxelsGPU(
    const device_vector<Vector3f>& positions_L,
    device_vector<VoxelType>* voxels_ptr,
    device_vector<bool>* success_flags_ptr, CudaStream* cuda_stream_ptr) const {
  CHECK_NOTNULL(voxels_ptr);
  CHECK_NOTNULL(success_flags_ptr);
  CHECK_NOTNULL(cuda_stream_ptr);

  const int num_queries = positions_L.size();
  voxels_ptr->resizeAsync(num_queries, *cuda_stream_ptr);
  success_flags_ptr->resizeAsync(num_queries, *cuda_stream_ptr);

  constexpr int kNumThreads = 512;
  const int num_blocks = divideRoundUp(num_queries, kNumThreads);

  queryVoxelsKernel<VoxelType>
      <<<num_blocks, kNumThreads, 0, *cuda_stream_ptr>>>(
          num_queries, this->getGpuLayerView(*cuda_stream_ptr).getHash().impl_,
          this->block_size_, positions_L.data(), voxels_ptr->data(),
          success_flags_ptr->data());

  cuda_stream_ptr->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

// MeshBlockLayer

struct add_constant_functor {
  const int constant_;

  add_constant_functor(int constant) : constant_(constant) {}

  __host__ __device__ int operator()(const int& x) const {
    return x + constant_;
  }
};

template <typename AppearanceType>
const std::shared_ptr<Mesh<AppearanceType>>
MeshBlockLayer<AppearanceType>::getMesh(const CudaStream& cuda_stream) const {
  // Count total vertices and triangles across all blocks
  int total_num_vertices = 0;
  int total_num_triangles = 0;
  const std::vector<Index3D> block_indices = this->getAllBlockIndices();
  for (const Index3D& index : block_indices) {
    typename MeshBlockType::ConstPtr block = this->getBlockAtIndex(index);
    total_num_vertices += block->vertices.size();
    total_num_triangles += block->triangles.size();
  }

  // Clear the mesh (without deallocating to avoid reallocating).
  // Expand the mesh buffers if the total number of vertices and triangles
  // is greater than the current capacity.
  mesh_->clearNoDeallocate();
  expandBuffersIfRequired(total_num_vertices, cuda_stream, &mesh_->vertices);
  expandBuffersIfRequired(total_num_vertices, cuda_stream,
                          &mesh_->vertex_normals);
  expandBuffersIfRequired(total_num_vertices, cuda_stream,
                          &mesh_->vertex_appearances);
  expandBuffersIfRequired(total_num_triangles, cuda_stream, &mesh_->triangles);

  // Keep track of the indices.
  int next_vertex_index = 0;
  int next_triangle_index = 0;

  // Loop over mesh blocks copying them to the monolithic mesh.
  for (const Index3D& index : block_indices) {
    typename MeshBlockType::ConstPtr block = this->getBlockAtIndex(index);

    // Check that the mesh block has:
    // - per vertex appearances
    // - per vertex vertex_normals
    const size_t num_vertices_in_block = block->vertices.size();
    const size_t num_triangles_in_block = block->triangles.size();
    CHECK((num_vertices_in_block == block->vertex_normals.size()) ||
          (block->vertex_normals.size() == 0));
    CHECK((num_vertices_in_block == block->vertex_appearances.size()) ||
          (block->vertex_appearances.size() == 0));

    // Append the vertices.
    mesh_->vertices.resizeAsync(mesh_->vertices.size() + num_vertices_in_block,
                                cuda_stream);
    block->vertices.copyToAsync(mesh_->vertices.data() + next_vertex_index,
                                cuda_stream);

    // Append the normals.
    mesh_->vertex_normals.resizeAsync(
        mesh_->vertex_normals.size() + num_vertices_in_block, cuda_stream);
    block->vertex_normals.copyToAsync(
        mesh_->vertex_normals.data() + next_vertex_index, cuda_stream);

    // Append the appearances.
    mesh_->vertex_appearances.resizeAsync(
        mesh_->vertex_appearances.size() + num_vertices_in_block, cuda_stream);
    block->vertex_appearances.copyToAsync(
        mesh_->vertex_appearances.data() + next_vertex_index, cuda_stream);

    // Append the triangles.
    mesh_->triangles.resizeAsync(
        mesh_->triangles.size() + num_triangles_in_block, cuda_stream);
    // The triangle indices are relative to the vertex index, so we need to add
    // the current vertex index to each triangle index.
    thrust::device_ptr<const int> block_triangles_thrust(
        block->triangles.data());
    thrust::device_ptr<int> mesh_triangles_thrust(mesh_->triangles.data() +
                                                  next_triangle_index);
    thrust::transform(thrust::device.on(cuda_stream), block_triangles_thrust,
                      block_triangles_thrust + num_triangles_in_block,
                      mesh_triangles_thrust,
                      add_constant_functor(next_vertex_index));

    // Increment the indices.
    next_vertex_index += num_vertices_in_block;
    next_triangle_index += num_triangles_in_block;
  }

  // Check that the output mesh has:
  // - per vertex appearances
  // - per vertex vertex_normals
  CHECK((mesh_->vertices.size() == mesh_->vertex_normals.size()) ||
        (mesh_->vertex_normals.size() == 0));
  CHECK((mesh_->vertices.size() == mesh_->vertex_appearances.size()) ||
        (mesh_->vertex_appearances.size() == 0));

  return mesh_;
}

}  // namespace nvblox
