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

namespace nvblox {

template <typename LayerType>
DecayIntegratorBase<LayerType>::DecayIntegratorBase(DecayMode decay_mode) {
  if (decay_mode == DecayMode::kDecayToDeallocate) {
    deallocate_decayed_blocks(true);
  } else if (decay_mode == DecayMode::kDecayToFree) {
    deallocate_decayed_blocks(false);
  } else {
    LOG(FATAL) << "Decay mode not implemented";
  }
}

template <typename LayerType>
bool DecayIntegratorBase<LayerType>::deallocate_decayed_blocks() const {
  return deallocate_decayed_blocks_;
}

template <typename LayerType>
void DecayIntegratorBase<LayerType>::deallocate_decayed_blocks(
    bool deallocate_decayed_blocks) {
  deallocate_decayed_blocks_ = deallocate_decayed_blocks;
}

template <typename LayerType>
parameters::ParameterTreeNode DecayIntegratorBase<LayerType>::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "decay_integrator" : name_remap;
  return ParameterTreeNode(name,
                           {
                               ParameterTreeNode("deallocate_decayed_blocks:",
                                                 deallocate_decayed_blocks_),
                           });
}

}  // namespace nvblox
