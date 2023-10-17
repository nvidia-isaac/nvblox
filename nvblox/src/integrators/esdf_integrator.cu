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
#include "nvblox/integrators/esdf_integrator.h"

#include "cub/block/block_radix_sort.cuh"
#include "nvblox/core/internal/cuda/atomic_float.cuh"
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh"
#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/gpu_hash/internal/cuda/gpu_set.cuh"
#include "nvblox/utils/timing.h"
#include "thrust/sort.h"
#include "thrust/unique.h"

namespace nvblox {

__device__ bool isVoxelFreespace(const FreespaceBlock* freespace_block_ptr,
                                 const dim3& voxel_index) {
  if (freespace_block_ptr == nullptr) {
    return false;
  } else {
    const FreespaceVoxel* freespace_voxel_ptr =
        &freespace_block_ptr
             ->voxels[voxel_index.x][voxel_index.y][voxel_index.z];
    return freespace_voxel_ptr->is_high_confidence_freespace;
  }
}

struct TsdfSiteFunctor {
  __device__ bool isVoxelObserved(const TsdfVoxel& tsdf_voxel) const {
    return tsdf_voxel.weight >= min_weight;
  }

  __device__ bool isVoxelInsideObject(const TsdfVoxel& tsdf_voxel) const {
    return tsdf_voxel.distance <= 0.0f;
  }

  __device__ bool isVoxelNearSurface(const TsdfVoxel& tsdf_voxel) const {
    return fabsf(tsdf_voxel.distance) <= max_site_distance_m;
  }

  __device__ void updateSquashedExtremumAtomic(const TsdfVoxel& tsdf_voxel,
                                               const bool is_freespace,
                                               TsdfVoxel* current_value) const {
    if (is_freespace) {
      // Ignore voxels that are marked as freespace.
      return;
    }
    atomicMinFloat(&current_value->distance, tsdf_voxel.distance);
  }

  float min_weight;
  float max_site_distance_m;
};

struct OccupancySiteFunctor {
  __device__ bool isVoxelObserved(const OccupancyVoxel& occupancy_voxel) const {
    constexpr float kEps = 1e-4;
    constexpr float kLogOddsZeroPointFive = 0;
    return fabsf(occupancy_voxel.log_odds - kLogOddsZeroPointFive) > kEps;
  }

  __device__ bool isVoxelInsideObject(
      const OccupancyVoxel& occupancy_voxel) const {
    return occupancy_voxel.log_odds > occupied_threshold_log_odds;
  }

  __device__ bool isVoxelNearSurface(
      const OccupancyVoxel& occupancy_voxel) const {
    return true;
  }

  __device__ void updateSquashedExtremumAtomic(
      const OccupancyVoxel& occupancy_voxel, const bool is_freespace,
      OccupancyVoxel* current_voxel) const {
    if (is_freespace) {
      // Ignore voxels that are marked as freespace.
      return;
    }
    atomicMaxFloat(&current_voxel->log_odds, occupancy_voxel.log_odds);
  }

  float occupied_threshold_log_odds;
};

EsdfIntegrator::EsdfIntegrator()
    : EsdfIntegrator(std::make_shared<CudaStreamOwning>()) {}

EsdfIntegrator::EsdfIntegrator(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

float EsdfIntegrator::max_esdf_distance_m() const {
  return max_esdf_distance_m_;
}

float EsdfIntegrator::max_site_distance_vox() const {
  return max_tsdf_site_distance_vox_;
}

float EsdfIntegrator::min_weight() const { return tsdf_min_weight_; }

void EsdfIntegrator::max_esdf_distance_m(float max_esdf_distance_m) {
  CHECK_GT(max_esdf_distance_m, 0.0f);
  max_esdf_distance_m_ = max_esdf_distance_m;
}

void EsdfIntegrator::max_site_distance_vox(float max_site_distance_vox) {
  CHECK_GT(max_site_distance_vox, 0.0f);
  max_tsdf_site_distance_vox_ = max_site_distance_vox;
}

void EsdfIntegrator::min_weight(float min_weight) {
  CHECK_GT(min_weight, 0.0f);
  tsdf_min_weight_ = min_weight;
}

float EsdfIntegrator::occupied_threshold() const {
  return probabilityFromLogOdds(occupied_threshold_log_odds_);
}

void EsdfIntegrator::occupied_threshold(float occupied_threshold) {
  CHECK_GE(occupied_threshold, 0.0f);
  CHECK_LE(occupied_threshold, 1.0f);
  occupied_threshold_log_odds_ = logOddsFromProbability(occupied_threshold);
}

parameters::ParameterTreeNode EsdfIntegrator::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "esdf_integrator" : name_remap;
  return ParameterTreeNode(
      name, {
                ParameterTreeNode("max_esdf_distance_m:", max_esdf_distance_m_),
                ParameterTreeNode("max_tsdf_site_distance_vox:",
                                  max_tsdf_site_distance_vox_),
                ParameterTreeNode("tsdf_min_weight:", tsdf_min_weight_),
                ParameterTreeNode("occupied_threshold_log_odds:",
                                  occupied_threshold_log_odds_),
            });
}

// Integrate the entire layer.
void EsdfIntegrator::integrateLayer(const TsdfLayer& tsdf_layer,
                                    EsdfLayer* esdf_layer) {
  std::vector<Index3D> block_indices = tsdf_layer.getAllBlockIndices();
  integrateBlocks(tsdf_layer, block_indices, esdf_layer);
}

void EsdfIntegrator::integrateLayer(const TsdfLayer& tsdf_layer,
                                    const FreespaceLayer& freespace_layer,
                                    EsdfLayer* esdf_layer) {
  std::vector<Index3D> block_indices = tsdf_layer.getAllBlockIndices();
  integrateBlocks(tsdf_layer, freespace_layer, block_indices, esdf_layer);
}

void EsdfIntegrator::integrateLayer(const OccupancyLayer& occupancy_layer,
                                    EsdfLayer* esdf_layer) {
  std::vector<Index3D> block_indices = occupancy_layer.getAllBlockIndices();
  integrateBlocks(occupancy_layer, block_indices, esdf_layer);
}

template <typename LayerType>
void EsdfIntegrator::integrateBlocksTemplate(
    const LayerType& layer, const std::vector<Index3D>& block_indices,
    EsdfLayer* esdf_layer, const FreespaceLayer* freespace_layer_ptr) {
  timing::Timer esdf_timer("esdf/integrate");

  if (block_indices.empty()) {
    return;
  }

  timing::Timer allocate_timer("esdf/integrate/allocate");
  // First, allocate all the destination blocks.
  allocateBlocksOnCPU(block_indices, esdf_layer);
  allocate_timer.Stop();

  timing::Timer mark_timer("esdf/integrate/mark_sites");
  // Then, mark all the sites on GPU.
  // This finds all the blocks that are eligible to be parents.
  markAllSites(layer, block_indices, freespace_layer_ptr, esdf_layer,
               &updated_indices_device_, &to_clear_indices_device_);
  mark_timer.Stop();

  if (!to_clear_indices_device_.empty()) {
    timing::Timer compute_timer("esdf/integrate/clear");
    clearAllInvalid(to_clear_indices_device_.toVector(), esdf_layer,
                    &cleared_block_indices_device_);
    cuda_stream_->synchronize();
  }

  timing::Timer compute_timer("esdf/integrate/compute");
  // Parallel block banding on GPU.
  computeEsdf(updated_indices_device_, esdf_layer);
  if (!cleared_block_indices_device_.empty()) {
    computeEsdf(cleared_block_indices_device_, esdf_layer);
  }
  compute_timer.Stop();
}

void EsdfIntegrator::integrateBlocks(const TsdfLayer& tsdf_layer,
                                     const std::vector<Index3D>& block_indices,
                                     EsdfLayer* esdf_layer) {
  integrateBlocksTemplate<TsdfLayer>(tsdf_layer, block_indices, esdf_layer);
}

void EsdfIntegrator::integrateBlocks(const TsdfLayer& tsdf_layer,
                                     const FreespaceLayer& freespace_layer,
                                     const std::vector<Index3D>& block_indices,
                                     EsdfLayer* esdf_layer) {
  integrateBlocksTemplate<TsdfLayer>(tsdf_layer, block_indices, esdf_layer,
                                     &freespace_layer);
}

void EsdfIntegrator::integrateBlocks(const OccupancyLayer& occupancy_layer,
                                     const std::vector<Index3D>& block_indices,
                                     EsdfLayer* esdf_layer) {
  integrateBlocksTemplate<OccupancyLayer>(occupancy_layer, block_indices,
                                          esdf_layer);
}

template <typename LayerType>
void EsdfIntegrator::integrateSliceTemplate(
    const LayerType& layer, const std::vector<Index3D>& block_indices,
    float z_min, float z_max, float z_output, EsdfLayer* esdf_layer,
    const FreespaceLayer* freespace_layer_ptr) {
  timing::Timer esdf_timer("esdf/integrate_slice");

  if (block_indices.empty()) {
    return;
  }

  timing::Timer mark_timer("esdf/integrate_slice/mark_sites");
  // Then, mark all the sites on GPU.
  // This finds all the blocks that are eligible to be parents.
  markSitesInSlice(layer, block_indices, z_min, z_max, z_output,
                   freespace_layer_ptr, esdf_layer, &updated_indices_device_,
                   &to_clear_indices_device_);
  mark_timer.Stop();

  if (!to_clear_indices_device_.empty()) {
    timing::Timer compute_timer("esdf/integrate/clear");
    clearAllInvalid(to_clear_indices_device_.toVectorAsync(*cuda_stream_),
                    esdf_layer, &cleared_block_indices_device_);
    cuda_stream_->synchronize();
  }

  timing::Timer compute_timer("esdf/integrate_slice/compute");
  // Parallel block banding on GPU.
  computeEsdf(updated_indices_device_, esdf_layer);
  if (!cleared_block_indices_device_.empty()) {
    computeEsdf(cleared_block_indices_device_, esdf_layer);
  }
  compute_timer.Stop();
}

void EsdfIntegrator::integrateSlice(const TsdfLayer& tsdf_layer,
                                    const std::vector<Index3D>& block_indices,
                                    float z_min, float z_max, float z_output,
                                    EsdfLayer* esdf_layer) {
  integrateSliceTemplate<TsdfLayer>(tsdf_layer, block_indices, z_min, z_max,
                                    z_output, esdf_layer);
}

void EsdfIntegrator::integrateSlice(const TsdfLayer& tsdf_layer,
                                    const FreespaceLayer& freespace_layer,
                                    const std::vector<Index3D>& block_indices,
                                    float z_min, float z_max, float z_output,
                                    EsdfLayer* esdf_layer) {
  integrateSliceTemplate<TsdfLayer>(tsdf_layer, block_indices, z_min, z_max,
                                    z_output, esdf_layer, &freespace_layer);
}

void EsdfIntegrator::integrateSlice(const OccupancyLayer& occupancy_layer,
                                    const std::vector<Index3D>& block_indices,
                                    float z_min, float z_max, float z_output,
                                    EsdfLayer* esdf_layer) {
  integrateSliceTemplate<OccupancyLayer>(occupancy_layer, block_indices, z_min,
                                         z_max, z_output, esdf_layer);
}

void EsdfIntegrator::allocateBlocksOnCPU(
    const std::vector<Index3D>& block_indices, EsdfLayer* esdf_layer) {
  // We want to allocate all ESDF layer blocks and copy over the sites.
  for (const Index3D& block_index : block_indices) {
    esdf_layer->allocateBlockAtIndexAsync(block_index, *cuda_stream_);
  }
}

__device__ void clearVoxelDevice(EsdfVoxel* voxel,
                                 float max_squared_esdf_distance_vox) {
  voxel->parent_direction.setZero();
  voxel->squared_distance_vox = max_squared_esdf_distance_vox;
  voxel->is_site = false;
}

// Mark sites to lower & clear.
// Block size MUST be voxels_per_side x voxels_per_side x voxel_per_size.
// Grid size can be anything.
template <typename BlockType, typename SiteFunctorType>
__global__ void markAllSitesKernel(
    int num_blocks, Index3D* block_indices,
    const Index3DDeviceHashMapType<BlockType> input_layer_block_hash,
    const Index3DDeviceHashMapType<FreespaceBlock> freespace_block_hash,
    Index3DDeviceHashMapType<EsdfBlock> esdf_block_hash,
    const SiteFunctorType site_functor, float max_squared_esdf_distance_vox,
    Index3D* updated_vec, int* updated_vec_size, Index3D* to_clear_vec,
    int* to_clear_vec_size) {
  dim3 voxel_index = threadIdx;
  int block_idx = blockIdx.x;

  using VoxelType = typename BlockType::VoxelType;

  __shared__ BlockType* block_ptr;
  __shared__ FreespaceBlock* freespace_block_ptr;
  __shared__ EsdfBlock* esdf_block;
  __shared__ int updated;
  __shared__ int to_clear;
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    block_ptr = nullptr;
    freespace_block_ptr = nullptr;
    esdf_block = nullptr;
    updated = false;
    to_clear = false;
    auto tsdf_it = input_layer_block_hash.find(block_indices[block_idx]);
    if (tsdf_it != input_layer_block_hash.end()) {
      block_ptr = tsdf_it->second;
    }
    if (!freespace_block_hash.empty()) {
      auto freespace_it = freespace_block_hash.find(block_indices[block_idx]);
      if (freespace_it != freespace_block_hash.end()) {
        freespace_block_ptr = freespace_it->second;
      }
    }

    auto esdf_it = esdf_block_hash.find(block_indices[block_idx]);
    if (esdf_it != esdf_block_hash.end()) {
      esdf_block = esdf_it->second;
    }
  }
  __syncthreads();
  if (block_ptr == nullptr || esdf_block == nullptr) {
    // We do not check the freespace_block_ptr here.
    // If the freespace block is not allocated we assume it to not be freespace.
    return;
  }

  // Get the correct voxel for this index.
  const VoxelType* voxel_ptr =
      &block_ptr->voxels[voxel_index.x][voxel_index.y][voxel_index.z];
  EsdfVoxel* esdf_voxel =
      &esdf_block->voxels[voxel_index.x][voxel_index.y][voxel_index.z];
  if (site_functor.isVoxelObserved(*voxel_ptr)) {
    // Mark as inside if the voxel distance is negative.
    bool is_inside = site_functor.isVoxelInsideObject(*voxel_ptr);
    // Voxels being freespace can not be inside an object.
    is_inside &= !isVoxelFreespace(freespace_block_ptr, voxel_index);
    // Esdf sites are ignored if they fall into freespace
    if (esdf_voxel->is_inside && is_inside == false) {
      clearVoxelDevice(esdf_voxel, max_squared_esdf_distance_vox);
      to_clear = true;
    }
    esdf_voxel->is_inside = is_inside;
    if (is_inside && site_functor.isVoxelNearSurface(*voxel_ptr)) {
      esdf_voxel->is_site = true;
      esdf_voxel->squared_distance_vox = 0.0f;
      esdf_voxel->parent_direction.setZero();
      updated = true;
    } else {
      if (esdf_voxel->is_site) {
        esdf_voxel->is_site = false;
        // This voxel needs to be cleared.
        clearVoxelDevice(esdf_voxel, max_squared_esdf_distance_vox);
        to_clear = true;
      } else if (!esdf_voxel->observed) {
        // This is a brand new voxel.
        clearVoxelDevice(esdf_voxel, max_squared_esdf_distance_vox);
      } else if (esdf_voxel->squared_distance_vox <= 1e-4) {
        // This is an invalid voxel that should be cleared.
        clearVoxelDevice(esdf_voxel, max_squared_esdf_distance_vox);
        to_clear = true;
      }
    }
    esdf_voxel->observed = true;
  } else {
    clearVoxelDevice(esdf_voxel, max_squared_esdf_distance_vox);
    to_clear = true;
    esdf_voxel->observed = false;
  }

  __syncthreads();

  if (threadIdx.x == 1 && threadIdx.y == 1 && threadIdx.z == 1) {
    if (updated) {
      updated_vec[atomicAdd(updated_vec_size, 1)] = block_indices[block_idx];
    }
    if (to_clear) {
      to_clear_vec[atomicAdd(to_clear_vec_size, 1)] = block_indices[block_idx];
    }
  }
}

template <typename T>
struct conditional_false : std::false_type {};

// NOTE(alexmillane): We use these types below to eliminate the default
// constructors of the voxel types such that they can be used with shared
// memory. In our testing this is no longer required after CUDA 11.8. At some
// point in the future (once the jetpack moves to >= 11.8) we can remove this.
namespace {

struct TsdfVoxelShared : public TsdfVoxel {
  TsdfVoxelShared() {}
};

struct OccupancyVoxelShared : public OccupancyVoxel {
  OccupancyVoxelShared() {}
};

template <typename VoxelType>
struct SharedVoxel;

template <>
struct SharedVoxel<TsdfVoxel> {
  typedef TsdfVoxelShared type;
};

template <>
struct SharedVoxel<OccupancyVoxel> {
  typedef OccupancyVoxelShared type;
};

}  // namespace

// ThreadsPerBlock: kVoxelsPerSide * kVoxelsPerSide * num_vertical_blocks
// ThreadBlockDim: number_of_blocks_in_slice * 1 * 1.
// NOTE(remos): All block indices have the same z-value (output_block_index_z)
// and no block index is duplicated.
template <typename BlockType, typename SiteFunctorType>
__global__ void markSitesInSliceKernel(
    Index3D* block_indices_in_output_slice,
    const Index3DDeviceHashMapType<BlockType> input_layer_block_hash,
    const Index3DDeviceHashMapType<FreespaceBlock> freespace_block_hash,
    Index3DDeviceHashMapType<EsdfBlock> esdf_block_hash,
    const SiteFunctorType site_functor, float max_squared_esdf_distance_vox,
    int min_input_block_index_z, int min_input_voxel_index_z,
    int max_input_voxel_index_z, int output_slice_voxel_index_z,
    float block_size, Index3D* updated_vec, int* updated_vec_size,
    Index3D* to_clear_vec, int* to_clear_vec_size) {
  // Voxel indices on the slice (x/y plane)
  int voxel_idx_x = threadIdx.x;
  int voxel_idx_y = threadIdx.y;
  // The vertical index offset of the block we want to process
  // Ranging from [0, num_vertical_blocks]
  int vertical_block_idx_offset = threadIdx.z;
  // Number of blocks in a vertical column
  int num_vertical_blocks = blockDim.z;
  // The vector index used to get the block from the
  // block_indices_in_output_slice vector
  int block_vector_idx = blockIdx.x;

  using VoxelType = typename BlockType::VoxelType;

  constexpr int kVoxelsPerSide = BlockType::kVoxelsPerSide;

  // First port-of-call is squashing a 3D band of the surface reconstruction to
  // 2D. First we allocating 2D arrays for the output.
  __shared__ bool observed[kVoxelsPerSide][kVoxelsPerSide];
  __shared__ typename SharedVoxel<VoxelType>::type voxel_slice[kVoxelsPerSide]
                                                              [kVoxelsPerSide];
  __shared__ Index3D esdf_slice_block_index;
  __shared__ EsdfBlock* esdf_slice_block_ptr;
  __shared__ bool updated, cleared;

  // Initialize this once for each voxel in the x/y plane.
  if (vertical_block_idx_offset == 0) {
    observed[voxel_idx_x][voxel_idx_y] = false;
    if constexpr (std::is_same<TsdfVoxel, VoxelType>::value) {
      // NOTE(alexmillane): We don't use the weight in the slice, so we don't
      // initialize it.
      voxel_slice[voxel_idx_x][voxel_idx_y].distance =
          2 * max_squared_esdf_distance_vox;
    } else if constexpr (std::is_same<OccupancyVoxel, VoxelType>::value) {
      voxel_slice[voxel_idx_x][voxel_idx_y].log_odds = 0.0f;
    } else {
      static_assert(conditional_false<BlockType>::value,
                    "Slicing not specialized to LayerType yet.");
    }
  }
  // Initialize this once for each block in the x/y plane (i.e. for each block
  // in block_indices_in_output_slice).
  if (voxel_idx_x == 0 && voxel_idx_y == 0 && vertical_block_idx_offset == 0) {
    updated = false;
    cleared = false;
    esdf_slice_block_index = block_indices_in_output_slice[block_vector_idx];
    esdf_slice_block_ptr = nullptr;
    auto it = esdf_block_hash.find(esdf_slice_block_index);
    if (it != esdf_block_hash.end()) {
      esdf_slice_block_ptr = it->second;
    }
  }
  __syncthreads();

  // This shouldn't happen.
  if (esdf_slice_block_ptr == nullptr) {
    printf(
        "No output block exists in markSitesInSliceKernel(). Shouldn't "
        "happen.\n");
    return;
  }

  // Get the block in the vertical column depending on the current offset.
  Index3D block_in_column_index = esdf_slice_block_index;
  block_in_column_index.z() =
      min_input_block_index_z + vertical_block_idx_offset;

  // Get the corresponding block pointers.
  const BlockType* block_in_column_ptr = nullptr;
  auto it = input_layer_block_hash.find(block_in_column_index);
  if (it != input_layer_block_hash.end()) {
    block_in_column_ptr = it->second;
  }
  const FreespaceBlock* freespace_block_ptr = nullptr;
  if (!freespace_block_hash.empty()) {
    auto freespace_it = freespace_block_hash.find(block_in_column_index);
    if (freespace_it != freespace_block_hash.end()) {
      freespace_block_ptr = freespace_it->second;
    }
  }

  // There's also null pointers in there.
  if (block_in_column_ptr != nullptr) {
    // Iterate over all of the voxels in this block column:
    // - block in block column in parallel (see block_in_column_ptr above)
    // - x/y voxel index in parallel (with voxel_idx_x/voxel_idx_y)
    // - z voxel index with for loop
    // Write the minimum values found vertically into the 2D-voxel_slice.
    int start_index = 0;
    int end_index = kVoxelsPerSide;
    if (vertical_block_idx_offset == 0) {
      start_index = min_input_voxel_index_z;
    }
    if (vertical_block_idx_offset == num_vertical_blocks - 1) {
      end_index = max_input_voxel_index_z;
    }
    for (int i = start_index; i < end_index; i++) {
      const VoxelType* voxel_ptr =
          &block_in_column_ptr->voxels[voxel_idx_x][voxel_idx_y][i];
      // Get the correct voxel for this index.
      if (site_functor.isVoxelObserved(*voxel_ptr)) {
        observed[voxel_idx_x][voxel_idx_y] = true;
        const bool is_freespace = isVoxelFreespace(
            freespace_block_ptr, dim3(voxel_idx_x, voxel_idx_y, i));
        site_functor.updateSquashedExtremumAtomic(
            *voxel_ptr, is_freespace, &voxel_slice[voxel_idx_x][voxel_idx_y]);
      }
    }
  }

  // sync threads across everyone trying to update this voxel
  __syncthreads();

  // Ok now we compare the new and old values and decide what to output.
  // Do this once for each voxel in the x/y plane.
  if (vertical_block_idx_offset == 0) {
    EsdfVoxel* esdf_voxel =
        &esdf_slice_block_ptr
             ->voxels[voxel_idx_x][voxel_idx_y][output_slice_voxel_index_z];

    // Case 0: Just skip it if it's unobserved. We don't care.
    if (observed[voxel_idx_x][voxel_idx_y]) {
      // Determine if the new value puts us inside or in a site.
      const bool is_inside = site_functor.isVoxelInsideObject(
          voxel_slice[voxel_idx_x][voxel_idx_y]);
      const bool is_site =
          is_inside && site_functor.isVoxelNearSurface(
                           voxel_slice[voxel_idx_x][voxel_idx_y]);

      // First handle the case where the voxel is a site.
      if (is_site) {
        if (esdf_voxel->is_site) {
          // Ok whatever. Add to the site list.
          // Its existing values are fine.
          updated = true;
        } else {
          // Wasn't a site before, is now.
          esdf_voxel->observed = true;
          esdf_voxel->squared_distance_vox = 0.0f;
          esdf_voxel->parent_direction.setZero();
          esdf_voxel->is_site = true;
          updated = true;
        }
      } else {
        // Here we have to double-check what's going on.
        // If it was a site before, and isn't anymore, we have to clear it.
        if (esdf_voxel->is_site) {
          esdf_voxel->is_site = false;
          clearVoxelDevice(esdf_voxel, max_squared_esdf_distance_vox);
          cleared = true;
        }
        // Otherwise just leave it alone unless it's brand new.
        if (!esdf_voxel->observed) {
          esdf_voxel->observed = true;
          clearVoxelDevice(esdf_voxel, max_squared_esdf_distance_vox);
        } else if (esdf_voxel->is_inside != is_inside) {
          // In case the sidedness swapped, clear the voxel.
          clearVoxelDevice(esdf_voxel, max_squared_esdf_distance_vox);
          cleared = true;
        } else if (esdf_voxel->squared_distance_vox <= 0.0f) {
          // This is somehow invalidly marked as a site despite the fact
          // it shouldn't be.
          clearVoxelDevice(esdf_voxel, max_squared_esdf_distance_vox);
          cleared = true;
        }
      }
      // Make the sidedness match.
      esdf_voxel->is_inside = is_inside;
    } else {
      clearVoxelDevice(esdf_voxel, max_squared_esdf_distance_vox);
      cleared = true;
      esdf_voxel->observed = false;
    }
  }

  __syncthreads();
  // Now output the updated and cleared.
  // Do this once for each block in the slice.
  if (voxel_idx_x == 0 && voxel_idx_y == 0 && vertical_block_idx_offset == 0) {
    if (updated) {
      updated_vec[atomicAdd(updated_vec_size, 1)] = esdf_slice_block_index;
    }
    if (cleared) {
      to_clear_vec[atomicAdd(to_clear_vec_size, 1)] = esdf_slice_block_index;
    }
  }
}

__device__ void sweepSingleBand(Index3D voxel_index, int sweep_axis,
                                float max_squared_esdf_distance_vox,
                                EsdfBlock* esdf_block) {
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  Index3D last_site;
  bool site_found;
  // Sweep sweep sweep.
  // First we sweep forward, then backwards.
  for (int i = 0; i < 2; i++) {
    last_site = Index3D::Zero();
    site_found = false;
    int direction = 1;
    int start_voxel = 0;
    int end_voxel = kVoxelsPerSide;
    if (i == 1) {
      direction = -1;
      start_voxel = kVoxelsPerSide - 1;
      end_voxel = -1;
    }

    for (voxel_index(sweep_axis) = start_voxel;
         voxel_index(sweep_axis) != end_voxel;
         voxel_index(sweep_axis) += direction) {
      EsdfVoxel* esdf_voxel =
          &esdf_block
               ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
      if (!esdf_voxel->observed) {
        continue;
      }
      // If this voxel is itself a site, then mark this for future voxels.
      if (esdf_voxel->is_site) {
        last_site = voxel_index;
        site_found = true;
      } else if (!site_found) {
        // If this voxel isn't a site but we haven't found a site yet,
        // then if this voxel is valid we set it as the site.
        if (esdf_voxel->squared_distance_vox < max_squared_esdf_distance_vox) {
          site_found = true;
          last_site = esdf_voxel->parent_direction + voxel_index;
        }
      } else {
        // If we've found the site, then should just decide what to do
        // here.
        Index3D potential_direction = last_site - voxel_index;
        float potential_distance = potential_direction.squaredNorm();
        // Either it hasn't been set at all or it's closer to the site
        // than to its current value.
        if (esdf_voxel->squared_distance_vox > potential_distance) {
          esdf_voxel->parent_direction = potential_direction;
          esdf_voxel->squared_distance_vox = potential_distance;
        } else if (esdf_voxel->squared_distance_vox <
                   max_squared_esdf_distance_vox) {
          // If the current value is a better site, then set it as a site.
          last_site = esdf_voxel->parent_direction + voxel_index;
        }
      }
    }
  }
}

__device__ bool updateSingleNeighbor(const EsdfBlock* esdf_block,
                                     const Index3D& voxel_index,
                                     const Index3D& neighbor_voxel_index,
                                     int axis, int direction,
                                     float max_squared_esdf_distance_vox,
                                     EsdfBlock* neighbor_block) {
  const EsdfVoxel* esdf_voxel =
      &esdf_block->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
  EsdfVoxel* neighbor_voxel =
      &neighbor_block
           ->voxels[neighbor_voxel_index.x()][neighbor_voxel_index.y()]
                   [neighbor_voxel_index.z()];
  if (!esdf_voxel->observed || !neighbor_voxel->observed ||
      neighbor_voxel->is_site ||
      esdf_voxel->squared_distance_vox >= max_squared_esdf_distance_vox) {
    return false;
  }
  // Determine if we can update this.
  Eigen::Vector3i potential_direction = esdf_voxel->parent_direction;
  potential_direction(axis) -= direction;
  float potential_distance = potential_direction.squaredNorm();
  // TODO: might be some concurrency issues here, have to be a bit careful
  // on the corners/edges.
  if (neighbor_voxel->squared_distance_vox > potential_distance) {
    neighbor_voxel->parent_direction = potential_direction;
    neighbor_voxel->squared_distance_vox = potential_distance;
    return true;
  }
  return false;
}

__device__ bool clearSingleNeighbor(const EsdfBlock* esdf_block,
                                    const Index3D& voxel_index,
                                    const Index3D& neighbor_voxel_index,
                                    int axis, int direction,
                                    float max_squared_esdf_distance_vox,
                                    EsdfBlock* neighbor_block) {
  const EsdfVoxel* esdf_voxel =
      &esdf_block->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
  EsdfVoxel* neighbor_voxel =
      &neighbor_block
           ->voxels[neighbor_voxel_index.x()][neighbor_voxel_index.y()]
                   [neighbor_voxel_index.z()];

  if (esdf_voxel->squared_distance_vox < max_squared_esdf_distance_vox ||
      !esdf_voxel->observed || neighbor_voxel->is_site ||
      neighbor_voxel->squared_distance_vox >= max_squared_esdf_distance_vox) {
    return false;
  }
  // Determine if we can update this.
  Index3D parent_voxel_dir = neighbor_voxel->parent_direction;
  if ((direction > 0 && parent_voxel_dir(axis) > 0) ||
      (direction < 0 && parent_voxel_dir(axis) < 0)) {
    return false;
  }

  clearVoxelDevice(neighbor_voxel, max_squared_esdf_distance_vox);
  return true;
}

OccupancySiteFunctor EsdfIntegrator::getSiteFunctor(const OccupancyLayer&) {
  OccupancySiteFunctor functor;
  functor.occupied_threshold_log_odds = occupied_threshold_log_odds_;
  return functor;
}

TsdfSiteFunctor EsdfIntegrator::getSiteFunctor(const TsdfLayer& layer) {
  TsdfSiteFunctor functor;
  functor.min_weight = tsdf_min_weight_;
  functor.max_site_distance_m =
      max_tsdf_site_distance_vox_ * layer.voxel_size();
  return functor;
}

template <typename LayerType>
void EsdfIntegrator::markAllSites(const LayerType& layer,
                                  const std::vector<Index3D>& block_indices,
                                  const FreespaceLayer* freespace_layer_ptr,
                                  EsdfLayer* esdf_layer,
                                  device_vector<Index3D>* blocks_with_sites,
                                  device_vector<Index3D>* cleared_blocks) {
  CHECK_NOTNULL(esdf_layer);
  CHECK_NOTNULL(blocks_with_sites);

  if (block_indices.empty()) {
    return;
  }

  // Caching.
  const float voxel_size = layer.voxel_size();
  const float max_esdf_distance_vox = max_esdf_distance_m_ / voxel_size;
  const float max_squared_esdf_distance_vox =
      max_esdf_distance_vox * max_esdf_distance_vox;

  int num_blocks = block_indices.size();

  block_indices_device_.copyFromAsync(block_indices, *cuda_stream_);
  blocks_with_sites->resizeAsync(num_blocks, *cuda_stream_);
  cleared_blocks->resizeAsync(num_blocks, *cuda_stream_);

  if (updated_counter_device_ == nullptr || updated_counter_host_ == nullptr) {
    updated_counter_device_ = make_unified<int>(MemoryType::kDevice);
    updated_counter_host_ = make_unified<int>(MemoryType::kHost);
  }
  updated_counter_device_.setZeroAsync(*cuda_stream_);
  if (cleared_counter_device_ == nullptr || cleared_counter_host_ == nullptr) {
    cleared_counter_device_ = make_unified<int>(MemoryType::kDevice);
    cleared_counter_host_ = make_unified<int>(MemoryType::kHost);
  }
  cleared_counter_device_.setZeroAsync(*cuda_stream_);

  GPULayerView<EsdfBlock> esdf_layer_view = esdf_layer->getGpuLayerView();
  GPULayerView<typename LayerType::BlockType> input_layer_view =
      layer.getGpuLayerView();

  Index3DDeviceHashMapType<FreespaceBlock> freespace_hash_map;
  if (freespace_layer_ptr != nullptr) {
    freespace_hash_map = freespace_layer_ptr->getGpuLayerView().getHash().impl_;
  }

  // Get the marking functions for this layer type
  auto site_functor = getSiteFunctor(layer);

  // Call the kernel.
  int dim_block = num_blocks;
  constexpr int kVoxelsPerSide = EsdfBlock::kVoxelsPerSide;
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  // Call kernel, passing functor
  markAllSitesKernel<<<dim_block, dim_threads, 0, *cuda_stream_>>>(
      num_blocks, block_indices_device_.data(),  // NOLINT
      input_layer_view.getHash().impl_,          // NOLINT
      freespace_hash_map,                        // NOLINT
      esdf_layer_view.getHash().impl_,           // NOLINT
      site_functor,                              // NOLINT
      max_squared_esdf_distance_vox,             // NOLINT
      blocks_with_sites->data(),                 // NOLINT
      updated_counter_device_.get(),             // NOLINT
      cleared_blocks->data(),                    // NOLINT
      cleared_counter_device_.get());

  checkCudaErrors(cudaPeekAtLastError());

  timing::Timer pack_out_timer("esdf/integrate/mark_sites/pack_out");
  updated_counter_device_.copyToAsync(updated_counter_host_, *cuda_stream_);
  cleared_counter_device_.copyToAsync(cleared_counter_host_, *cuda_stream_);
  cuda_stream_->synchronize();

  blocks_with_sites->resize(*updated_counter_host_);
  cleared_blocks->resize(*cleared_counter_host_);
  pack_out_timer.Stop();
}

// Helper function
std::pair<int, int> getBlockAndVoxelZIndexFromHeightInLayer(
    const float block_size, float height_in_layer) {
  Index3D block_index;
  Index3D voxel_index;
  getBlockAndVoxelIndexFromPositionInLayer(
      block_size, Vector3f(0.0f, 0.0f, height_in_layer), &block_index,
      &voxel_index);
  return {block_index.z(), voxel_index.z()};
}

template <typename LayerType>
void EsdfIntegrator::markSitesInSlice(const LayerType& input_layer,
                                      const std::vector<Index3D>& block_indices,
                                      float min_z, float max_z, float output_z,
                                      const FreespaceLayer* freespace_layer_ptr,
                                      EsdfLayer* esdf_layer,
                                      device_vector<Index3D>* updated_blocks,
                                      device_vector<Index3D>* cleared_blocks) {
  if (block_indices.empty()) {
    return;
  }
  const float voxel_size = input_layer.voxel_size();
  const float block_size = input_layer.block_size();
  const float max_esdf_distance_vox = max_esdf_distance_m_ / voxel_size;
  const float max_squared_esdf_distance_vox =
      max_esdf_distance_vox * max_esdf_distance_vox;
  constexpr int kVoxelsPerSide = EsdfBlock::kVoxelsPerSide;

  // We are going to subsample the block_indices.
  // We need to figure out all the output blocks, which will be a subset
  // of the input blocks.

  // Find the minimum z-index that should be included in the slice
  const auto [min_input_block_index_z, min_input_voxel_index_z] =
      getBlockAndVoxelZIndexFromHeightInLayer(block_size, min_z);
  // Find the maximum z-index that should be included in the slice
  const auto [max_input_block_index_z, max_input_voxel_index_z] =
      getBlockAndVoxelZIndexFromHeightInLayer(block_size, max_z);
  // And figure out what the z-index on the output esdf slice is.
  const auto [output_slice_block_index_z, output_slice_voxel_index_z] =
      getBlockAndVoxelZIndexFromHeightInLayer(block_size, output_z);

  // Figure out how many blocks we have in one vertical column (at least 1).
  CHECK_GE(max_input_block_index_z, min_input_block_index_z);
  int num_blocks_in_vertical_column =
      max_input_block_index_z - min_input_block_index_z + 1;

  // Next get a set of all the valid output blocks:
  // - lying in between [z_min, z_max]
  // - z-index set to output_z
  // - only one block per vertical column (because z-index is overwritten and a
  //   set implementation is used)
  Index3DSet block_indices_in_output_slice_set;
  for (const Index3D& block_index : block_indices) {
    if (block_index.z() >= min_input_block_index_z &&
        block_index.z() <= max_input_block_index_z) {
      block_indices_in_output_slice_set.insert(Index3D(
          block_index.x(), block_index.y(), output_slice_block_index_z));
    }
  }

  // Resize everything to the final size.
  size_t num_blocks_in_output_slice = block_indices_in_output_slice_set.size();
  block_indices_host_.resize(num_blocks_in_output_slice);
  updated_indices_device_.resizeAsync(num_blocks_in_output_slice,
                                      *cuda_stream_);
  to_clear_indices_device_.resizeAsync(num_blocks_in_output_slice,
                                       *cuda_stream_);
  if (num_blocks_in_output_slice == 0) {
    return;
  }

  // Convert the block set to a vector and allocate output esdf blocks.
  size_t i = 0;
  for (const Index3D& block_index : block_indices_in_output_slice_set) {
    block_indices_host_[i] = block_index;
    // Allocate the esdf block at the output slice height.
    esdf_layer->allocateBlockAtIndexAsync(block_index, *cuda_stream_);
    i++;
  }
  block_indices_device_.copyFromAsync(block_indices_host_, *cuda_stream_);

  // Reset the counters.
  if (updated_counter_device_ == nullptr || updated_counter_host_ == nullptr) {
    updated_counter_device_ = make_unified<int>(MemoryType::kDevice);
    updated_counter_host_ = make_unified<int>(MemoryType::kHost);
  }
  updated_counter_device_.setZeroAsync(*cuda_stream_);
  if (cleared_counter_device_ == nullptr || cleared_counter_host_ == nullptr) {
    cleared_counter_device_ = make_unified<int>(MemoryType::kDevice);
    cleared_counter_host_ = make_unified<int>(MemoryType::kHost);
  }
  cleared_counter_device_.setZeroAsync(*cuda_stream_);

  using BlockType = typename LayerType::BlockType;

  // Get the GPU hash of both the TSDF and the ESDF.
  GPULayerView<EsdfBlock> esdf_layer_view = esdf_layer->getGpuLayerView();
  GPULayerView<BlockType> tsdf_layer_view = input_layer.getGpuLayerView();

  Index3DDeviceHashMapType<FreespaceBlock> freespace_hash_map;
  if (freespace_layer_ptr != nullptr) {
    freespace_hash_map = freespace_layer_ptr->getGpuLayerView().getHash().impl_;
  }

  // Get the marking functions for this layer type
  auto site_functor = getSiteFunctor(input_layer);

  // Figure out the size of the kernel.
  int dim_block = num_blocks_in_output_slice;
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide,
                   num_blocks_in_vertical_column);
  // Call the kernel!
  markSitesInSliceKernel<BlockType>
      <<<dim_block, dim_threads, 0, *cuda_stream_>>>(
          block_indices_device_.data(),     // NOLINT
          tsdf_layer_view.getHash().impl_,  // NOLINT
          freespace_hash_map,               // NOLINT
          esdf_layer_view.getHash().impl_,  // NOLINT
          site_functor,                     // NOLINT
          max_squared_esdf_distance_vox,    // NOLINT
          min_input_block_index_z,          // NOLINT
          min_input_voxel_index_z,          // NOLINT
          max_input_voxel_index_z,          // NOLINT
          output_slice_voxel_index_z,       // NOLINT
          input_layer.block_size(),         // NOLINT
          updated_blocks->data(),           // NOLINT
          updated_counter_device_.get(),    // NOLINT
          cleared_blocks->data(),           // NOLINT
          cleared_counter_device_.get());
  checkCudaErrors(cudaPeekAtLastError());

  timing::Timer pack_out_timer("esdf/integrate/mark_sites/pack_out");
  updated_counter_device_.copyToAsync(updated_counter_host_, *cuda_stream_);
  cleared_counter_device_.copyToAsync(cleared_counter_host_, *cuda_stream_);
  cuda_stream_->synchronize();

  updated_blocks->resize(*updated_counter_host_);
  cleared_blocks->resize(*cleared_counter_host_);
  pack_out_timer.Stop();
}

__host__ __device__ void getDirectionAndVoxelIndicesFromThread(
    dim3 thread_index, Index3D* block_direction, Index3D* voxel_index,
    Index3D* neighbor_voxel_index, int* axis, int* direction) {
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  *block_direction = Index3D::Zero();
  // Thread index z is used for the neighbor number. x & y are actual voxel
  // indices.
  (*block_direction)(thread_index.z / 2) = thread_index.z % 2 ? -1 : 1;
  // This is to to make it easier to pick voxel directions.
  *axis = thread_index.z / 2;
  *direction = thread_index.z % 2 ? -1 : 1;

  // Fill in the axes.
  if (*axis == 0) {
    *voxel_index << 0, thread_index.x, thread_index.y;
  } else if (*axis == 1) {
    *voxel_index << thread_index.x, 0, thread_index.y;
  } else if (*axis == 2) {
    *voxel_index << thread_index.x, thread_index.y, 0;
  }
  *neighbor_voxel_index = *voxel_index;
  // If we're looking backwards...
  if (*direction < 0) {
    (*voxel_index)(*axis) = 0;
    (*neighbor_voxel_index)(*axis) = kVoxelsPerSide - 1;
  } else {
    (*voxel_index)(*axis) = kVoxelsPerSide - 1;
    (*neighbor_voxel_index)(*axis) = 0;
  }
}

// Thread size MUST be 8x8x6, 8x8 being the side of the cube, and 6 being the
// number of neighbors considered per block. Block size can be whatever.
__global__ void updateNeighborBandsKernel(
    int i, int num_blocks, Index3DDeviceHashMapType<EsdfBlock> block_hash,
    float max_squared_esdf_distance_vox, Index3D* block_indices,
    Index3D* output_vector, int* updated_size) {
  // Luckily the direction is the same for all processed blocks by this thread.
  Index3D block_direction, voxel_index, neighbor_voxel_index;
  int axis, direction;
  getDirectionAndVoxelIndicesFromThread(threadIdx, &block_direction,
                                        &voxel_index, &neighbor_voxel_index,
                                        &axis, &direction);

  __shared__ bool block_updated;
  // Allow block size to be whatever.
  __shared__ EsdfBlock* block_ptr;
  EsdfBlock* neighbor_block_ptr = nullptr;
  for (int block_idx = blockIdx.x; block_idx < num_blocks;
       block_idx += gridDim.x) {
    __syncthreads();
    // Get the current block for this... block.
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      block_ptr = nullptr;
      auto it = block_hash.find(block_indices[block_idx]);
      if (it != block_hash.end()) {
        block_ptr = it->second;
      }
      block_updated = false;
    }
    __syncthreads();
    // This block doesn't exist. Who knows why. This shouldn't happen.
    if (block_ptr == nullptr) {
      continue;
    }

    dim3 specific_thread = threadIdx;
    specific_thread.z = i;
    Index3D block_direction, voxel_index, neighbor_voxel_index;
    int axis, direction;
    getDirectionAndVoxelIndicesFromThread(specific_thread, &block_direction,
                                          &voxel_index, &neighbor_voxel_index,
                                          &axis, &direction);

    // Get the neighbor block for this thread.
    neighbor_block_ptr = nullptr;
    auto it = block_hash.find(block_indices[block_idx] + block_direction);
    if (it != block_hash.end()) {
      neighbor_block_ptr = it->second;
    }
    // Our neighbor doesn't exist. This is fine and normal. Happens to
    // everyone.
    if (neighbor_block_ptr == nullptr) {
      continue;
    }

    bool updated = updateSingleNeighbor(
        block_ptr, voxel_index, neighbor_voxel_index, axis, direction,
        max_squared_esdf_distance_vox, neighbor_block_ptr);
    // No bother with atomics.
    if (updated) {
      block_updated = updated;
    }

    __syncthreads();
    if ((threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) &&
        block_updated) {
      //*any_updated = true;
      output_vector[atomicAdd(updated_size, 1)] =
          block_indices[block_idx] + block_direction;
    }
  }
}

template <int kBlockThreads, int kItemsPerThread>
__global__ void sortUniqueKernel(Index3D* indices, int num_indices,
                                 int* num_output_indices) {
  typedef uint64_t IndexHashValue;
  typedef int OriginalIndex;

  typedef cub::BlockRadixSort<uint64_t, kBlockThreads, kItemsPerThread,
                              OriginalIndex>
      BlockRadixSortT;
  typedef cub::BlockDiscontinuity<IndexHashValue, kBlockThreads>
      BlockDiscontinuityT;
  typedef cub::BlockScan<OriginalIndex, kBlockThreads> BlockScanT;

  // Allocate type-safe, repurposable shared memory for collectives
  __shared__ union {
    typename BlockRadixSortT::TempStorage sort;
    typename BlockDiscontinuityT::TempStorage discontinuity;
    typename BlockScanT::TempStorage scan;
  } temp_storage;

  // First we create a values list which is actually the indicies.
  // Obtain this block's segment of consecutive keys (blocked across threads)
  uint64_t thread_keys[kItemsPerThread];
  Index3D thread_values[kItemsPerThread];
  int thread_inds[kItemsPerThread];
  int head_flags[kItemsPerThread];
  int head_indices[kItemsPerThread];
  int thread_offset = threadIdx.x * kItemsPerThread;

  // Fill in the keys from the values.
  // I guess we can just do a for loop. kItemsPerThread should be fairly small.
  Index3DHash index_hash;
  for (int i = 0; i < kItemsPerThread; i++) {
    if (thread_offset + i >= num_indices) {
      // We just pack the key with a large value.
      thread_values[i] = Index3D::Zero();
      thread_keys[i] = SIZE_MAX;
      thread_inds[i] = -1;
    } else {
      thread_values[i] = indices[thread_offset + i];
      thread_keys[i] = index_hash(thread_values[i]);
      thread_inds[i] = thread_offset + i;
    }
  }

  // We then sort the values.
  __syncthreads();
  // Collectively sort the keys
  BlockRadixSortT(temp_storage.sort).Sort(thread_keys, thread_inds);
  __syncthreads();
  // We remove duplicates by find when the discontinuities happen.
  BlockDiscontinuityT(temp_storage.discontinuity)
      .FlagHeads(head_flags, thread_keys, cub::Inequality());
  __syncthreads();
  // Get the indices that'll be assigned to the new unique values.
  BlockScanT(temp_storage.scan)
      .InclusiveSum<kItemsPerThread>(head_flags, head_indices);
  __syncthreads();

  // Cool now write only 1 instance of the unique entries to the output.
  for (int i = 0; i < kItemsPerThread; i++) {
    if (thread_offset + i < num_indices) {
      if (head_flags[i] == 1) {
        // Get the proper value out. Cache this for in-place ops next step.
        thread_values[i] = indices[thread_inds[i]];
        atomicMax(num_output_indices, head_indices[i]);
      }
    }
  }
  __syncthreads();

  // Have to do this twice since we do this in-place. Now actually replace
  // the values.
  for (int i = 0; i < kItemsPerThread; i++) {
    if (thread_offset + i < num_indices) {
      if (head_flags[i] == 1) {
        // Get the proper value out.
        indices[head_indices[i] - 1] = thread_values[i];
      }
    }
  }
}

void EsdfIntegrator::sortAndTakeUniqueIndices(
    device_vector<Index3D>* block_indices) {
  if (block_indices->size() == 0) {
    return;
  }
  // Together this should be >> the number of indices
  constexpr int kNumThreads = 128;
  constexpr int kNumItemsPerThread = 4;
  if (block_indices->size() >= kNumThreads * kNumItemsPerThread) {
    LOG(INFO) << "Vector too big to sort. Falling back to thrust.";
    // sort vertices to bring duplicates together
    thrust::sort(thrust::device, block_indices->begin(), block_indices->end(),
                 VectorCompare<Index3D>());

    // Find unique vertices and erase redundancies. The iterator will point to
    // the new last index.
    auto iterator = thrust::unique(thrust::device, block_indices->begin(),
                                   block_indices->end());

    // Figure out the new size.
    size_t new_size = iterator - block_indices->begin();
    block_indices->resizeAsync(new_size, *cuda_stream_);
    return;
  }
  if (updated_counter_device_ == nullptr || updated_counter_host_ == nullptr) {
    updated_counter_device_ = make_unified<int>(MemoryType::kDevice);
    updated_counter_host_ = make_unified<int>(MemoryType::kHost);
  }
  updated_counter_device_.setZeroAsync(*cuda_stream_);

  sortUniqueKernel<kNumThreads, kNumItemsPerThread>
      <<<1, kNumThreads, 0, *cuda_stream_>>>(block_indices->data(),  // NOLINT
                                             block_indices->size(),  // NOLINT
                                             updated_counter_device_.get());
  checkCudaErrors(cudaPeekAtLastError());

  updated_counter_device_.copyToAsync(updated_counter_host_, *cuda_stream_);
  cuda_stream_->synchronize();

  block_indices->resize(*updated_counter_host_);
}

void EsdfIntegrator::updateNeighborBands(
    device_vector<Index3D>* block_indices, EsdfLayer* esdf_layer,
    float max_squared_esdf_distance_vox,
    device_vector<Index3D>* updated_block_indices) {
  if (block_indices->empty()) {
    return;
  }
  timing::Timer sweep_timer("esdf/integrate/compute/neighbor_bands");

  // This function just copies neighbors across block boundaries.
  constexpr int kNumNeighbors = 6;
  constexpr int kUpdatedBlockMultiple = kNumNeighbors;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;

  updated_block_indices->resizeAsync(
      block_indices->size() * kUpdatedBlockMultiple, *cuda_stream_);
  updated_block_indices->setZeroAsync(*cuda_stream_);

  // Create an output variable.
  if (updated_counter_device_ == nullptr || updated_counter_host_ == nullptr) {
    updated_counter_device_ = make_unified<int>(MemoryType::kDevice);
    updated_counter_host_ = make_unified<int>(MemoryType::kHost);
  }
  updated_counter_device_.setZeroAsync(*cuda_stream_);

  timing::Timer gpu_view("esdf/integrate/compute/neighbor_bands/gpu_view");
  GPULayerView<EsdfBlock> gpu_layer_view = esdf_layer->getGpuLayerView();
  gpu_view.Stop();

  // Call the kernel.
  int dim_block = block_indices->size();
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, 1);
  for (int i = 0; i < kNumNeighbors; i++) {
    updateNeighborBandsKernel<<<dim_block, dim_threads, 0, *cuda_stream_>>>(
        i, block_indices->size(),        // NOLINT
        gpu_layer_view.getHash().impl_,  // NOLINT
        max_squared_esdf_distance_vox,   // NOLINT
        block_indices->data(),           // NOLINT
        updated_block_indices->data(),   // NOLINT
        updated_counter_device_.get());
  }
  checkCudaErrors(cudaPeekAtLastError());

  updated_counter_device_.copyToAsync(updated_counter_host_, *cuda_stream_);
  cuda_stream_->synchronize();

  updated_block_indices->resize(*updated_counter_host_);

  if (*updated_counter_host_ == 0) {
    return;
  }

  timing::Timer copy_out_timer(
      "esdf/integrate/compute/neighbor_bands/copy_out");
  sortAndTakeUniqueIndices(updated_block_indices);
}

/// Thread size MUST be 8x8xN (where N is a number of blocks up to ???), block
/// size can be anything.
__global__ void sweepBlockBandKernel(
    int num_blocks, Index3DDeviceHashMapType<EsdfBlock> block_hash,
    float max_squared_esdf_distance_vox, Index3D* block_indices) {
  // We go one axis at a time, syncing threads in between.
  dim3 thread_index = threadIdx;
  thread_index.z = 0;

  __shared__ EsdfBlock* esdf_block;

  for (int block_idx = blockIdx.x * blockDim.z + threadIdx.z;
       block_idx < num_blocks; block_idx += gridDim.x * blockDim.z) {
    // For simplicity we have to have the same number of blocks in the CUDA
    // kernel call as we have actual blocks.
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      esdf_block = nullptr;
      auto it = block_hash.find(block_indices[block_idx]);
      if (it != block_hash.end()) {
        esdf_block = it->second;
      }
    }
    __syncthreads();
    // This block doesn't exist. Who knows why. This shouldn't happen.
    if (esdf_block == nullptr) {
      continue;
    }
    Index3D voxel_index(0, thread_index.x, thread_index.y);

    // X axis done.
    sweepSingleBand(voxel_index, 0, max_squared_esdf_distance_vox, esdf_block);
    __syncthreads();

    // Y axis done.
    voxel_index << thread_index.x, 0, thread_index.y;
    sweepSingleBand(voxel_index, 1, max_squared_esdf_distance_vox, esdf_block);
    __syncthreads();

    // Z axis done.
    voxel_index << thread_index.x, thread_index.y, 0;
    sweepSingleBand(voxel_index, 2, max_squared_esdf_distance_vox, esdf_block);
    __syncthreads();
  }
}

void EsdfIntegrator::sweepBlockBand(device_vector<Index3D>* block_indices,
                                    EsdfLayer* esdf_layer,
                                    float max_squared_esdf_distance_vox) {
  if (block_indices->empty()) {
    return;
  }
  timing::Timer sweep_timer("esdf/integrate/compute/sweep");

  // Caching.
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const int num_blocks = block_indices->size();

  GPULayerView<EsdfBlock> gpu_layer_view = esdf_layer->getGpuLayerView();

  // Call the kernel.
  // We do 2-dimensional sweeps in this kernel. Each thread does 3 sweeps.
  // We do 1 blocks at a time because it's faster.
  constexpr int kNumBlocksPerCudaBlock = 1;
  int dim_block = std::max(
      static_cast<int>(
          std::ceil(num_blocks / static_cast<float>(kNumBlocksPerCudaBlock))),
      1);
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, kNumBlocksPerCudaBlock);
  sweepBlockBandKernel<<<dim_block, dim_threads, 0, *cuda_stream_>>>(
      block_indices->size(),           // NOLINT
      gpu_layer_view.getHash().impl_,  // NOLINT
      max_squared_esdf_distance_vox,   // NOLINT
      block_indices->data());
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

void EsdfIntegrator::computeEsdf(
    const device_vector<Index3D>& blocks_with_sites, EsdfLayer* esdf_layer) {
  CHECK_NOTNULL(esdf_layer);

  if (blocks_with_sites.size() == 0) {
    return;
  }
  // Cache everything.
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const float voxel_size = esdf_layer->block_size() / kVoxelsPerSide;
  const float max_esdf_distance_vox = max_esdf_distance_m_ / voxel_size;
  const float max_squared_esdf_distance_vox =
      max_esdf_distance_vox * max_esdf_distance_vox;

  // First we go over all of the blocks with sites.
  // We compute all the proximal sites inside the block first.
  block_indices_device_.copyFromAsync(blocks_with_sites, *cuda_stream_);
  sweepBlockBand(&block_indices_device_, esdf_layer,
                 max_squared_esdf_distance_vox);

  while (!block_indices_device_.empty()) {
    updateNeighborBands(&block_indices_device_, esdf_layer,
                        max_squared_esdf_distance_vox,
                        &updated_indices_device_);
    sweepBlockBand(&updated_indices_device_, esdf_layer,
                   max_squared_esdf_distance_vox);

    timing::Timer swap_timer("esdf/integrate/compute/swap");
    std::swap(block_indices_device_, updated_indices_device_);
    swap_timer.Stop();
  }
}

__device__ void getBlockAndVoxelIndexFromOffset(const Index3D& block_index,
                                                const Index3D& voxel_index,
                                                const Index3D& voxel_offset,
                                                Index3D* neighbor_block_index,
                                                Index3D* neighbor_voxel_index) {
  // For each axis we have to get the mod and div to get the block index and
  // voxel index.
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;

  for (int i = 0; i < 3; i++) {
    (*neighbor_block_index)(i) =
        block_index(i) + voxel_offset(i) / kVoxelsPerSide;
    (*neighbor_voxel_index)(i) =
        voxel_index(i) + voxel_offset(i) % kVoxelsPerSide;
    if ((*neighbor_voxel_index)(i) >= kVoxelsPerSide) {
      (*neighbor_voxel_index)(i) -= kVoxelsPerSide;
      (*neighbor_block_index)(i)++;
    } else if ((*neighbor_voxel_index)(i) < 0) {
      (*neighbor_voxel_index)(i) += kVoxelsPerSide;
      (*neighbor_block_index)(i)--;
    }
  }
}

__global__ void clearAllInvalidKernel(
    Index3D* block_indices, Index3DDeviceHashMapType<EsdfBlock> block_hash,
    float max_squared_esdf_distance_vox, Index3D* output_vector,
    int* updated_size) {
  __shared__ int block_updated;
  // Allow block size to be whatever.
  __shared__ EsdfBlock* block_ptr;
  // Get the current block for this... block.
  __shared__ Index3D block_index;
  Index3D voxel_index = Index3D(threadIdx.x, threadIdx.y, threadIdx.z);
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    block_ptr = nullptr;
    block_index = block_indices[blockIdx.x];
    auto it = block_hash.find(block_index);
    if (it != block_hash.end()) {
      block_ptr = it->second;
    }
    block_updated = false;
  }
  __syncthreads();
  // This block doesn't exist. Who knows why. This shouldn't happen.
  if (block_ptr == nullptr) {
    return;
  }

  // Now for our specific voxel we should look up its parent and see if it's
  // still there.
  EsdfVoxel* esdf_voxel =
      &block_ptr->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];

  if (esdf_voxel->observed && !esdf_voxel->is_site &&
      esdf_voxel->parent_direction != Index3D::Zero()) {
    Index3D neighbor_block_index, neighbor_voxel_index;
    getBlockAndVoxelIndexFromOffset(
        block_index, voxel_index, esdf_voxel->parent_direction,
        &neighbor_block_index, &neighbor_voxel_index);

    EsdfVoxel* neighbor_voxel = nullptr;
    if (neighbor_block_index == block_index) {
      neighbor_voxel =
          &block_ptr->voxels[neighbor_voxel_index.x()][neighbor_voxel_index.y()]
                            [neighbor_voxel_index.z()];
    } else {
      // Get the neighboring block.
      auto it = block_hash.find(neighbor_block_index);
      if (it != block_hash.end()) {
        neighbor_voxel =
            &it->second
                 ->voxels[neighbor_voxel_index.x()][neighbor_voxel_index.y()]
                         [neighbor_voxel_index.z()];
      }
    }
    if (neighbor_voxel == nullptr || !neighbor_voxel->is_site) {
      // Clear this voxel.
      esdf_voxel->parent_direction.setZero();
      esdf_voxel->squared_distance_vox = max_squared_esdf_distance_vox;
      block_updated = true;
    }
  }
  __syncthreads();
  if ((threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) &&
      block_updated) {
    output_vector[atomicAdd(updated_size, 1)] = block_index;
  }
}

void EsdfIntegrator::clearAllInvalid(
    const std::vector<Index3D>& blocks_to_clear, EsdfLayer* esdf_layer,
    device_vector<Index3D>* updated_blocks) {
  if (blocks_to_clear.size() == 0) {
    return;
  }

  // TODO: start out just getting all the blocks in the whole map.
  // Then replace with blocks within a radius of the cleared blocks.
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const float max_esdf_distance_vox =
      max_esdf_distance_m_ / esdf_layer->voxel_size();
  const float max_squared_esdf_distance_vox =
      max_esdf_distance_vox * max_esdf_distance_vox;

  timing::Timer get_blocks_timer("esdf/integrate/clear/get_blocks");

  temp_indices_host_.copyFromAsync(
      getBlocksWithinRadiusOfAABB(
          esdf_layer->getAllBlockIndices(), esdf_layer->block_size(),
          getAABBOfBlocks(esdf_layer->block_size(), blocks_to_clear),
          max_esdf_distance_m_),
      *cuda_stream_);
  get_blocks_timer.Stop();
  temp_indices_device_.copyFromAsync(temp_indices_host_, *cuda_stream_);

  // Get the hash map of the whole ESDF map.
  GPULayerView<EsdfBlock> gpu_layer_view = esdf_layer->getGpuLayerView();

  // Create an output variable.
  if (updated_counter_device_ == nullptr || updated_counter_host_ == nullptr) {
    updated_counter_device_ = make_unified<int>(MemoryType::kDevice);
    updated_counter_host_ = make_unified<int>(MemoryType::kHost);
  }
  updated_counter_device_.setZeroAsync(*cuda_stream_);

  // Make sure we have enough space if EVERYTHING had to be cleared.
  updated_blocks->resizeAsync(temp_indices_device_.size(), *cuda_stream_);

  // Call a kernel.
  dim3 dim_threads(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  if (temp_indices_device_.size() > 0) {
    clearAllInvalidKernel<<<temp_indices_device_.size(), dim_threads, 0,
                            *cuda_stream_>>>(
        temp_indices_device_.data(),     // NOLINT
        gpu_layer_view.getHash().impl_,  // NOLINT
        max_squared_esdf_distance_vox,   // NOLINT
        updated_blocks->data(),          // NOLINT
        updated_counter_device_.get());
    checkCudaErrors(cudaPeekAtLastError());

    // Pack out the updated blocks.
    updated_counter_device_.copyToAsync(updated_counter_host_, *cuda_stream_);
    cuda_stream_->synchronize();

    updated_blocks->resize(*updated_counter_host_);
  } else {
    updated_blocks->resize(0);
  }
}

}  // namespace nvblox
