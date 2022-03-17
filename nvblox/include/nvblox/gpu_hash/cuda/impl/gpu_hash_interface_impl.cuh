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

template <typename BlockType>
GPUHashImpl<BlockType>::GPUHashImpl(int max_num_blocks)
    : max_num_blocks_(max_num_blocks),
      impl_(Index3DDeviceHashMapType<BlockType>::createDeviceObject(
          max_num_blocks_)) {
  VLOG(3) << "Creating a GPUHashImpl with max capacity of " << max_num_blocks_
          << " blocks";
}

template <typename BlockType>
GPUHashImpl<BlockType>::~GPUHashImpl() {
  if (impl_.size() > 0) {
    Index3DDeviceHashMapType<BlockType>::destroyDeviceObject(impl_);
    VLOG(3) << "Destroying a GPUHashImpl";
  }
}

}  // namespace nvblox
