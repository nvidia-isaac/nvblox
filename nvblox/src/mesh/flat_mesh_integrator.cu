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

#include "nvblox/mesh/flat_mesh_integrator.h"

#include <algorithm>

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "nvblox/core/indexing.h"
#include "nvblox/core/internal/error_check.h"
#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/mesh/internal/appearance_getter.h"
#include "nvblox/mesh/internal/cuda/marching_cubes.cuh"
#include "nvblox/mesh/internal/cuda/voxel_access.cuh"
#include "nvblox/mesh/internal/marching_cubes.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

namespace {

constexpr int kVoxelsPerSide = VoxelBlock<TsdfVoxel>::kVoxelsPerSide;

/// Read the 8 corner SDF values and positions for a voxel's marching cubes
/// cube, handling boundary crossover via the GPU hash map.
/// Returns false if any corner has insufficient weight.
__device__ bool readVoxelCorners(
    const TsdfBlock* block, const Index3D& block_index, int vx, int vy, int vz,
    const Index3DDeviceHashMapType<TsdfBlock>& tsdf_hash, float min_weight,
    float block_size, float sdf_values[marching_cubes::kNumCorners],
    Vector3f corner_positions[marching_cubes::kNumCorners]) {
  for (int i = 0; i < marching_cubes::kNumCorners; ++i) {
    int cx = vx + marching_cubes::kCornerIndexOffsets[i][0];
    int cy = vy + marching_cubes::kCornerIndexOffsets[i][1];
    int cz = vz + marching_cubes::kCornerIndexOffsets[i][2];

    if (!getTsdfVoxelAtLocalCoord(block, block_index, cx, cy, cz, tsdf_hash,
                                  min_weight, &sdf_values[i],
                                  &corner_positions[i], block_size)) {
      return false;
    }
  }
  return true;
}

/// Write one triangle's geometry (vertices + normals) to output buffers.
/// @param input_triangle_idx  Triangle index within the marching cubes table
/// row.
/// @param output_triangle_idx Absolute triangle index in the output buffer.
__device__ void writeTriangleGeometry(
    const int8_t* table_row, int input_triangle_idx,
    const Vector3f edge_vertices[marching_cubes::kNumEdges],
    int output_triangle_idx, Vector3f* out_vertices, Vector3f* out_normals) {
  const int in = input_triangle_idx * marching_cubes::kVerticesPerTriangle;
  int e0 = table_row[in + 0];
  int e1 = table_row[in + 1];
  int e2 = table_row[in + 2];

  const int out = output_triangle_idx * marching_cubes::kVerticesPerTriangle;

  // Reversed winding to match nvblox convention
  out_vertices[out + 0] = edge_vertices[e2];
  out_vertices[out + 1] = edge_vertices[e1];
  out_vertices[out + 2] = edge_vertices[e0];

  Vector3f p0 = out_vertices[out + 0];
  Vector3f p1 = out_vertices[out + 1];
  Vector3f p2 = out_vertices[out + 2];
  Vector3f normal = (p1 - p0).cross(p2 - p0).normalized();
  out_normals[out + 0] = normal;
  out_normals[out + 1] = normal;
  out_normals[out + 2] = normal;
}

/// No-op appearance writer for geometry-only extraction.
struct NoAppearanceWriter {
  __device__ void operator()(int, const Vector3f*, float) const {}
};

/// Appearance writer that samples from a voxel hash map.
/// @param output_triangle_idx Absolute triangle index in the output buffer.
template <typename AppearanceVoxelType>
struct AppearanceWriter {
  Index3DDeviceHashMapType<VoxelBlock<AppearanceVoxelType>> hash;
  typename AppearanceVoxelType::ArrayType* output;

  __device__ void operator()(int output_triangle_idx, const Vector3f* vertices,
                             float block_size) const {
    const int base = output_triangle_idx * marching_cubes::kVerticesPerTriangle;
    output[base + 0] = getAppearanceAtPosition<AppearanceVoxelType>(
        hash, vertices[base + 0], block_size);
    output[base + 1] = getAppearanceAtPosition<AppearanceVoxelType>(
        hash, vertices[base + 1], block_size);
    output[base + 2] = getAppearanceAtPosition<AppearanceVoxelType>(
        hash, vertices[base + 2], block_size);
  }
};

/// Marching cubes kernel with compile-time appearance policy.
/// AppearanceWriterT is either NoAppearanceWriter or AppearanceWriter<T>.
template <typename AppearanceWriterT>
__global__ void flatMarchingCubesKernel(
    int num_blocks, const TsdfBlock* const* tsdf_blocks,
    const Index3D* block_indices, Index3DDeviceHashMapType<TsdfBlock> tsdf_hash,
    float block_size, float min_weight, Vector3f* out_vertices,
    Vector3f* out_normals, int* triangle_counter, int max_triangles,
    AppearanceWriterT appearance_writer) {
  if (blockIdx.x >= num_blocks) return;

  const int vx = threadIdx.x;
  const int vy = threadIdx.y;
  const int vz = threadIdx.z;

  const TsdfBlock* block = tsdf_blocks[blockIdx.x];
  if (block == nullptr) return;

  const Index3D block_index = block_indices[blockIdx.x];

  float sdf_values[marching_cubes::kNumCorners];
  Vector3f corner_positions[marching_cubes::kNumCorners];
  if (!readVoxelCorners(block, block_index, vx, vy, vz, tsdf_hash, min_weight,
                        block_size, sdf_values, corner_positions)) {
    return;
  }

  Vector3f edge_vertices[marching_cubes::kNumEdges];
  const int8_t* table_row;
  int num_triangles = marching_cubes::marchingCubesFromCorners(
      sdf_values, corner_positions, edge_vertices, &table_row);
  if (num_triangles == 0) return;

  int tri_start = atomicAdd(triangle_counter, num_triangles);

  for (int triangle_idx = 0; triangle_idx < num_triangles; ++triangle_idx) {
    int output_triangle_idx = tri_start + triangle_idx;
    if (output_triangle_idx >= max_triangles) continue;

    writeTriangleGeometry(table_row, triangle_idx, edge_vertices,
                          output_triangle_idx, out_vertices, out_normals);
    appearance_writer(output_triangle_idx, out_vertices, block_size);
  }
}

}  // namespace

// --- FlatMeshIntegrator implementation ---

template <typename AppearanceVoxelType>
FlatMeshIntegrator<AppearanceVoxelType>::FlatMeshIntegrator()
    : FlatMeshIntegrator(std::make_shared<CudaStreamOwning>()) {}

template <typename AppearanceVoxelType>
FlatMeshIntegrator<AppearanceVoxelType>::FlatMeshIntegrator(
    std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(std::move(cuda_stream)) {
  // NOTE: triangle_counter_ is lazily allocated on first use in
  // integrateBlocksImpl(). Doing CUDA allocations in the constructor caused
  // segfaults on Jetson when many FlatMeshIntegrators were constructed
  // (each Mapper owns two), and also makes Mapper construction touch the GPU
  // unnecessarily.
}

template <typename AppearanceVoxelType>
FlatMeshIntegrator<AppearanceVoxelType>::~FlatMeshIntegrator() = default;

template <typename AppearanceVoxelType>
FlatMeshIntegrator<AppearanceVoxelType>::FlatMeshIntegrator(
    FlatMeshIntegrator&&) = default;

template <typename AppearanceVoxelType>
FlatMeshIntegrator<AppearanceVoxelType>&
FlatMeshIntegrator<AppearanceVoxelType>::operator=(FlatMeshIntegrator&&) =
    default;

template <typename AppearanceVoxelType>
void FlatMeshIntegrator<AppearanceVoxelType>::prepareBlockData(
    const TsdfLayer& tsdf_layer, const std::vector<Index3D>& block_indices) {
  transferBlockPointersToDeviceAsync<TsdfBlock>(
      block_indices, tsdf_layer, &tsdf_block_ptrs_host_,
      &tsdf_block_ptrs_device_, *cuda_stream_);

  block_indices_host_.resizeAsync(block_indices.size(), *cuda_stream_);
  block_indices_host_.copyFromAsync(block_indices, *cuda_stream_);
  block_indices_device_.copyFromAsync(block_indices_host_, *cuda_stream_);
}

template <typename AppearanceVoxelType>
int FlatMeshIntegrator<AppearanceVoxelType>::integrateBlocks(
    const TsdfLayer& tsdf_layer, const std::vector<Index3D>& block_indices,
    MeshType* mesh, bool fill_triangle_indices) {
  return integrateBlocksImpl(tsdf_layer, nullptr, block_indices, mesh,
                             fill_triangle_indices);
}

template <typename AppearanceVoxelType>
int FlatMeshIntegrator<AppearanceVoxelType>::integrateBlocks(
    const TsdfLayer& tsdf_layer, const AppearanceLayerType& appearance_layer,
    const std::vector<Index3D>& block_indices, MeshType* mesh,
    bool fill_triangle_indices) {
  return integrateBlocksImpl(tsdf_layer, &appearance_layer, block_indices, mesh,
                             fill_triangle_indices);
}

template <typename AppearanceVoxelType>
int FlatMeshIntegrator<AppearanceVoxelType>::integrateBlocksImpl(
    const TsdfLayer& tsdf_layer, const AppearanceLayerType* appearance_layer,
    const std::vector<Index3D>& block_indices, MeshType* mesh,
    bool fill_triangle_indices) {
  CHECK_NOTNULL(mesh);
  const bool has_appearance = (appearance_layer != nullptr);
  const std::string timer_prefix =
      has_appearance ? "flat_mesh/with_appearance" : "flat_mesh/geometry";
  timing::Timer total_timer(timer_prefix + "/total");

  if (block_indices.empty()) {
    mesh->clearNoDeallocate();
    return 0;
  }

  const float block_size = tsdf_layer.block_size();
  const int num_blocks = block_indices.size();

  const auto maxVertices = [&]() {
    return static_cast<size_t>(max_num_triangles_) *
           marching_cubes::kVerticesPerTriangle;
  };

  {
    timing::Timer prep_timer(timer_prefix + "/prep");
    if (!triangle_counter_) {
      triangle_counter_ = make_unified<int>(MemoryType::kUnified);
    }
    mesh->resizeAsync(maxVertices(), *cuda_stream_);
    prepareBlockData(tsdf_layer, block_indices);
  }

  using AppearanceHashType =
      Index3DDeviceHashMapType<VoxelBlock<AppearanceVoxelType>>;
  AppearanceHashType appearance_hash{};

  timing::Timer gpu_view_timer(timer_prefix + "/get_gpu_views");
  const auto& tsdf_gpu_view = tsdf_layer.getGpuLayerView(*cuda_stream_);
  if (has_appearance) {
    appearance_hash =
        appearance_layer->getGpuLayerView(*cuda_stream_).getHash().impl_;
  }
  gpu_view_timer.Stop();

  // Zero the counter and launch the kernel as an atomic operation.
  // The counter must be zero before the kernel runs; zeroing is done host-side
  // (not in the kernel) because CUDA has no grid-wide sync primitive that could
  // safely zero before all blocks start atomicAdd'ing. Stream ordering
  // guarantees the memset completes before the kernel executes.
  auto launchKernel = [&](auto appearance_writer) {
    triangle_counter_.setZeroAsync(*cuda_stream_);
    dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
    flatMarchingCubesKernel<<<num_blocks, dim_threads, 0, *cuda_stream_>>>(
        num_blocks, tsdf_block_ptrs_device_.data(),
        block_indices_device_.data(), tsdf_gpu_view.getHash().impl_, block_size,
        min_weight_, mesh->vertices.data(), mesh->vertex_normals.data(),
        triangle_counter_.get(), max_num_triangles_, appearance_writer);
    checkCudaErrors(cudaPeekAtLastError());
  };

  auto launch = [&]() {
    if (has_appearance) {
      launchKernel(AppearanceWriter<AppearanceVoxelType>{
          appearance_hash, mesh->vertex_appearances.data()});
    } else {
      launchKernel(NoAppearanceWriter{});
    }
  };

  // Launch kernel, synchronize, and read back the triangle count.
  // The timer spans launch + sync so Nsight captures the full GPU execution.
  auto launchAndReadCount = [&]() {
    timing::Timer kernel_timer(timer_prefix + "/kernel");
    launch();
    cuda_stream_->synchronize();
    kernel_timer.Stop();
    return *triangle_counter_;
  };

  int actual_count = launchAndReadCount();

  if (actual_count > max_num_triangles_) {
    timing::Timer retry_timer(timer_prefix + "/retry_overflow");
    LOG(WARNING) << "Flat mesh buffer overflow: " << actual_count
                 << " triangles, buffer=" << max_num_triangles_
                 << ". Growing and re-running.";
    max_num_triangles_ =
        std::max(static_cast<int>(actual_count * 1.5f), kMinTriangleBufferSize);
    mesh->resizeAsync(maxVertices(), *cuda_stream_);
    actual_count = launchAndReadCount();
  }

  {
    timing::Timer finalize_timer(timer_prefix + "/finalize");
    const int clamped = std::min(actual_count, max_num_triangles_);
    const int num_vertices = clamped * marching_cubes::kVerticesPerTriangle;
    if (num_vertices == 0) {
      mesh->clearNoDeallocate();
      return actual_count;
    }
    mesh->resizeAsync(num_vertices, *cuda_stream_);
    if (!has_appearance) {
      mesh->vertex_appearances.clearNoDeallocate();
    }
    if (fill_triangle_indices) {
      thrust::sequence(thrust::device.on(*cuda_stream_), mesh->triangles.data(),
                       mesh->triangles.data() + num_vertices);
    } else {
      mesh->triangles.clearNoDeallocate();
    }
    // UVs are produced by ProjectiveTextureMapper, not here.
    mesh->vertex_uvs.clearNoDeallocate();
    return actual_count;
  }
}

template <typename AppearanceVoxelType>
parameters::ParameterTreeNode
FlatMeshIntegrator<AppearanceVoxelType>::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "flat_mesh_integrator" : name_remap;
  return ParameterTreeNode(
      name, {ParameterTreeNode("min_weight", min_weight_),
             ParameterTreeNode("max_triangles", max_num_triangles_)});
}

// Explicit template instantiation
template class FlatMeshIntegrator<ColorVoxel>;
template class FlatMeshIntegrator<FeatureVoxel>;

}  // namespace nvblox
