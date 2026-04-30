/*
Copyright 2024 NVIDIA CORPORATION

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

#include "nvblox/serialization/internal/serialization_gpu.h"

#include <cuda_runtime.h>
#include <string>

#include "glog/logging.h"
#include "nvblox/core/internal/error_check.h"

namespace nvblox {

// Kernel that copies several vectors into a contigous chunk of memory
//
// Number of blocks:  Must equal num_vectors.
// Number of threads: Can be any positive value but a larger number than the
//   maximum vector size will not bring any gain.
//
// @param num_vectors        Number of vectors to serialize
// @param vectors            Vectors to serialize. Size: num_vectors
// @param offsets            Output buffer offsets. Last elements contain total
//                           num elements. Size: num_vectors+1
// @param serialized_buffer  Resulting buffer. Must have capacity for all
//                           elements
template <typename T>
void __global__ SerializeVectorsKernel(const int32_t num_vectors,
                                       const T** vectors,
                                       const int32_t* offsets,
                                       T* serialized_buffer) {
  const int32_t vector_index =
      blockIdx.x;  // Which vector does this block serialize?
  const int32_t element_index_start = threadIdx.x;

  // This should not happen if the kernel was launched with
  // num_blocks=num_vectors
  if (vector_index >= num_vectors) {
    return;
  }

  const int32_t offset = offsets[vector_index];
  const int32_t num_elements = offsets[vector_index + 1] - offset;

  // If vector size is larger than number of threads, we let each thread handle
  // several elements.
  for (int32_t index = element_index_start; index < num_elements;
       index += blockDim.x) {
    serialized_buffer[offset + index] = vectors[vector_index][index];
  }
}

template <typename LayerType, typename T>
template <template <typename> class OutputVector>
void LayerSerializerGpuInternal<LayerType, T>::serializeAsync(
    const LayerType& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    OutputVector<T>& serialized_output, host_vector<int32_t>& offsets_output,
    std::function<
        std::pair<const T*, int>(const typename LayerType::BlockType* block)>
        get_data_and_size,
    const CudaStream& cuda_stream) {
  if (block_indices_to_serialize.empty()) {
    return;
  }

  // Iterate over all blocks to serialize, store their data pointers and
  // offsets
  offsets_output.resizeAsync(block_indices_to_serialize.size() + 1,
                             cuda_stream);
  vector_ptrs_.resizeAsync(block_indices_to_serialize.size(), cuda_stream);
  cuda_stream.synchronize();
  int32_t total_num_elements = 0;
  int32_t max_block_size = 0;
  for (size_t i = 0; i < block_indices_to_serialize.size(); ++i) {
    const typename LayerType::BlockType* block =
        layer.getBlockAtIndex(block_indices_to_serialize[i]).get();

    const auto data_and_size = get_data_and_size(block);
    offsets_output[i] = total_num_elements;
    vector_ptrs_[i] = data_and_size.first;
    total_num_elements += data_and_size.second;

    max_block_size = std::max<int32_t>(max_block_size, data_and_size.second);
  }

  // We'll need the total num of elements as well so we can compute the
  // size of the last vector
  offsets_output[offsets_output.size() - 1] = total_num_elements;

  // We use thread_id to determine which vector element to copy. This
  // allow for coalesced memory transfers since all threads in one warp
  // will read from contiguous memory.
  constexpr int32_t kMaxNumThreads = 1024;
  const int32_t num_threads = std::min(max_block_size, kMaxNumThreads);

  // Process one layer-block per cuda-block
  const int32_t num_cuda_blocks = block_indices_to_serialize.size();

  // Run serialization.
  serialized_output.resizeAsync(total_num_elements, cuda_stream);
  if (num_threads > 0 && num_cuda_blocks > 0) {
    SerializeVectorsKernel<<<num_cuda_blocks, num_threads, 0, cuda_stream>>>(
        block_indices_to_serialize.size(), vector_ptrs_.data(),
        offsets_output.data(), serialized_output.data());
  }

  checkCudaErrors(cudaPeekAtLastError());
}

// Instantiation of serialize function for TSDF layer
template void
LayerSerializerGpuInternal<TsdfLayer, TsdfVoxel>::serializeAsync<host_vector>(
    const TsdfLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    host_vector<TsdfVoxel>& serialized_output,
    host_vector<int32_t>& offsets_output,
    std::function<std::pair<const TsdfVoxel*, int>(const TsdfBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

// Instantiation of serialize function for Color layer
template void
LayerSerializerGpuInternal<ColorLayer, ColorVoxel>::serializeAsync<host_vector>(
    const ColorLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    host_vector<ColorVoxel>& serialized_output,
    host_vector<int32_t>& offsets_output,
    std::function<std::pair<const ColorVoxel*, int>(const ColorBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

// Instantiation of serialize function for Feature layer
template void
LayerSerializerGpuInternal<FeatureLayer, FeatureVoxel>::serializeAsync<
    host_vector>(const FeatureLayer& layer,
                 const std::vector<Index3D>& block_indices_to_serialize,
                 host_vector<FeatureVoxel>& serialized_output,
                 host_vector<int32_t>& offsets_output,
                 std::function<std::pair<const FeatureVoxel*, int>(
                     const FeatureBlock* block)>
                     get_data_and_size,
                 const CudaStream& cuda_stream);

// Instantiation of serialize function for Occupancy layer
template void
LayerSerializerGpuInternal<OccupancyLayer, OccupancyVoxel>::serializeAsync<
    host_vector>(const OccupancyLayer& layer,
                 const std::vector<Index3D>& block_indices_to_serialize,
                 host_vector<OccupancyVoxel>& serialized_output,
                 host_vector<int32_t>& offsets_output,
                 std::function<std::pair<const OccupancyVoxel*, int>(
                     const OccupancyBlock* block)>
                     get_data_and_size,
                 const CudaStream& cuda_stream);

// Instantiation of serialize function for Freespace layer
template void
LayerSerializerGpuInternal<FreespaceLayer, FreespaceVoxel>::serializeAsync<
    host_vector>(const FreespaceLayer& layer,
                 const std::vector<Index3D>& block_indices_to_serialize,
                 host_vector<FreespaceVoxel>& serialized_output,
                 host_vector<int32_t>& offsets_output,
                 std::function<std::pair<const FreespaceVoxel*, int>(
                     const FreespaceBlock* block)>
                     get_data_and_size,
                 const CudaStream& cuda_stream);

// Instantiation of serialize function for Esdf layer
template void
LayerSerializerGpuInternal<EsdfLayer, EsdfVoxel>::serializeAsync<host_vector>(
    const EsdfLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    host_vector<EsdfVoxel>& serialized_output,
    host_vector<int32_t>& offsets_output,
    std::function<std::pair<const EsdfVoxel*, int>(const EsdfBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

// Instantiation of serialize function for Mesh layer::Vector3f
template void
LayerSerializerGpuInternal<ColorMeshLayer, Vector3f>::serializeAsync<
    host_vector>(
    const ColorMeshLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    host_vector<Vector3f>& serialized_output,
    host_vector<int32_t>& offsets_output,
    std::function<std::pair<const Vector3f*, int>(const ColorMeshBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

template void
LayerSerializerGpuInternal<ColorMeshLayer, Vector3f>::serializeAsync<
    device_vector>(
    const ColorMeshLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    device_vector<Vector3f>& serialized_output,
    host_vector<int32_t>& offsets_output,
    std::function<std::pair<const Vector3f*, int>(const ColorMeshBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

// Instantiation of serialize function for Mesh layer::Color
template void
LayerSerializerGpuInternal<ColorMeshLayer, Color>::serializeAsync<host_vector>(
    const ColorMeshLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    host_vector<Color>& serialized_output, host_vector<int32_t>& offsets_output,
    std::function<std::pair<const Color*, int>(const ColorMeshBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

template void LayerSerializerGpuInternal<ColorMeshLayer, Color>::serializeAsync<
    device_vector>(
    const ColorMeshLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    device_vector<Color>& serialized_output,
    host_vector<int32_t>& offsets_output,
    std::function<std::pair<const Color*, int>(const ColorMeshBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

// Instantiation of serialize function for Mesh layer::float
template void
LayerSerializerGpuInternal<ColorMeshLayer, float>::serializeAsync<host_vector>(
    const ColorMeshLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    host_vector<float>& serialized_output, host_vector<int32_t>& offsets_output,
    std::function<std::pair<const float*, int>(const ColorMeshBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

// Instantiation of serialize function for Mesh layer::int
template void
LayerSerializerGpuInternal<ColorMeshLayer, int>::serializeAsync<host_vector>(
    const ColorMeshLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    host_vector<int>& serialized_output, host_vector<int32_t>& offsets_output,
    std::function<std::pair<const int*, int>(const ColorMeshBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

template void
LayerSerializerGpuInternal<ColorMeshLayer, int>::serializeAsync<device_vector>(
    const ColorMeshLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    device_vector<int>& serialized_output, host_vector<int32_t>& offsets_output,
    std::function<std::pair<const int*, int>(const ColorMeshBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

// Instantiation of serialize function for Mesh layer::Vector3f
template void
LayerSerializerGpuInternal<FeatureMeshLayer, Vector3f>::serializeAsync<
    host_vector>(const FeatureMeshLayer& layer,
                 const std::vector<Index3D>& block_indices_to_serialize,
                 host_vector<Vector3f>& serialized_output,
                 host_vector<int32_t>& offsets_output,
                 std::function<std::pair<const Vector3f*, int>(
                     const FeatureMeshBlock* block)>
                     get_data_and_size,
                 const CudaStream& cuda_stream);

template void
LayerSerializerGpuInternal<FeatureMeshLayer, Vector3f>::serializeAsync<
    device_vector>(const FeatureMeshLayer& layer,
                   const std::vector<Index3D>& block_indices_to_serialize,
                   device_vector<Vector3f>& serialized_output,
                   host_vector<int32_t>& offsets_output,
                   std::function<std::pair<const Vector3f*, int>(
                       const FeatureMeshBlock* block)>
                       get_data_and_size,
                   const CudaStream& cuda_stream);

// Instantiation of serialize function for Mesh layer::FeatureArray
template void
LayerSerializerGpuInternal<FeatureMeshLayer, FeatureArray>::serializeAsync<
    host_vector>(const FeatureMeshLayer& layer,
                 const std::vector<Index3D>& block_indices_to_serialize,
                 host_vector<FeatureArray>& serialized_output,
                 host_vector<int32_t>& offsets_output,
                 std::function<std::pair<const FeatureArray*, int>(
                     const FeatureMeshBlock* block)>
                     get_data_and_size,
                 const CudaStream& cuda_stream);

template void
LayerSerializerGpuInternal<FeatureMeshLayer, FeatureArray>::serializeAsync<
    device_vector>(const FeatureMeshLayer& layer,
                   const std::vector<Index3D>& block_indices_to_serialize,
                   device_vector<FeatureArray>& serialized_output,
                   host_vector<int32_t>& offsets_output,
                   std::function<std::pair<const FeatureArray*, int>(
                       const FeatureMeshBlock* block)>
                       get_data_and_size,
                   const CudaStream& cuda_stream);

// Instantiation of serialize function for Mesh layer::float
template void
LayerSerializerGpuInternal<FeatureMeshLayer, float>::serializeAsync<
    host_vector>(
    const FeatureMeshLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    host_vector<float>& serialized_output, host_vector<int32_t>& offsets_output,
    std::function<std::pair<const float*, int>(const FeatureMeshBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

// Instantiation of serialize function for Mesh layer::int
template void
LayerSerializerGpuInternal<FeatureMeshLayer, int>::serializeAsync<host_vector>(
    const FeatureMeshLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    host_vector<int>& serialized_output, host_vector<int32_t>& offsets_output,
    std::function<std::pair<const int*, int>(const FeatureMeshBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

template void LayerSerializerGpuInternal<FeatureMeshLayer, int>::serializeAsync<
    device_vector>(
    const FeatureMeshLayer& layer,
    const std::vector<Index3D>& block_indices_to_serialize,
    device_vector<int>& serialized_output, host_vector<int32_t>& offsets_output,
    std::function<std::pair<const int*, int>(const FeatureMeshBlock* block)>
        get_data_and_size,
    const CudaStream& cuda_stream);

}  // namespace nvblox
