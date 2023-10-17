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

namespace nvblox {

template <typename VoxelType>
LayerParameterStruct serializeLayerParameters(
    const VoxelBlockLayer<VoxelType>& layer) {
  // Cool so we just need to populate a few parameters here.
  // Actually for now only block size. :)
  LayerParameterStruct params;
  params.float_params.emplace("block_size", layer.block_size());
  return params;
}

template <typename VoxelType>
std::vector<Index3D> getLayerDataIndices(
    const VoxelBlockLayer<VoxelType>& layer) {
  return layer.getAllBlockIndices();
}

template <typename VoxelType>
std::vector<Byte> serializeLayerDataAtIndex(
    const VoxelBlockLayer<VoxelType>& layer, const Index3D& index,
    const CudaStream cuda_stream) {
  std::vector<Byte> data;
  typename VoxelBlockLayer<VoxelType>::BlockType::ConstPtr block =
      layer.getBlockAtIndex(index);
  if (block == nullptr) {
    data.resize(0);
    return std::vector<Byte>();
  }

  return serializeBlock(block, cuda_stream);
}

template <typename VoxelType>
std::unique_ptr<VoxelBlockLayer<VoxelType>> deserializeLayerParameters(
    MemoryType memory_type, const LayerParameterStruct& params) {
  auto it = params.float_params.find("block_size");
  if (it == params.float_params.end()) {
    return std::unique_ptr<VoxelBlockLayer<VoxelType>>();
  }
  float block_size = it->second;
  // Create a layer.
  std::unique_ptr<VoxelBlockLayer<VoxelType>> layer_ptr;
  layer_ptr.reset(new VoxelBlockLayer<VoxelType>(
      block_size / VoxelBlock<VoxelType>::kVoxelsPerSide, memory_type));

  return layer_ptr;
}

template <typename VoxelType>
void addDataToLayer(const Index3D& index, const std::vector<Byte>& data,
                    VoxelBlockLayer<VoxelType>* layer,
                    const CudaStream cuda_stream) {
  // Create a block at the relevant index.
  auto block = layer->allocateBlockAtIndexAsync(index, cuda_stream);

  // Populate it using block serialialization.
  deserializeBlock(data, block, cuda_stream);
}

}  // namespace nvblox
