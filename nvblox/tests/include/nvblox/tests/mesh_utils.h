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
#include <vector>

#include "nvblox/core/types.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {

void weldVerticesThrust(const std::vector<Index3D>& block_indices,
                        BlockLayer<MeshBlock>* mesh_layer);
void weldSingleBlockThrust(device_vector<Vector3f>* input_vertices,
                           device_vector<int>* input_indices,
                           device_vector<Vector3f>* output_vertices,
                           device_vector<int>* output_indices);

void sortSingleBlockThrust(device_vector<Vector3f>* vertices);
void sortSingleBlockCub(device_vector<Vector3f>* input_vertices,
                        device_vector<Vector3f>* output_vertices);

void uniqueSingleBlockCub(device_vector<Vector3f>* input_vertices,
                          device_vector<Vector3f>* output_vertices);

void combinedSingleBlockCub(device_vector<Vector3f>* input_vertices,
                            device_vector<int>* input_indices,
                            device_vector<Vector3f>* output_vertices,
                            device_vector<int>* output_indices);

}  // namespace nvblox