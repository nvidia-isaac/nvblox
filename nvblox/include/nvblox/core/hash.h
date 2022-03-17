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

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "nvblox/core/types.h"

namespace nvblox {

/**
 * Performs deco hashing on block indexes. Based on recommendations of
 * "Investigating the impact of Suboptimal Hashing Functions" by L. Buckley et
 * al.
 */
struct Index3DHash {
  /// number was arbitrarily chosen with no good justification
  static constexpr size_t sl = 17191;
  static constexpr size_t sl2 = sl * sl;

  __host__ __device__ std::size_t operator()(const Index3D& index) const {
    return static_cast<unsigned int>(index.x() + index.y() * sl +
                                     index.z() * sl2);
  }
};

template <typename T>
struct VectorCompare {
  __host__ __device__ inline bool operator()(const T& p_1, const T& p_2) const {
    if (p_1.x() != p_2.x()) {
      return p_1.x() < p_2.x();
    }
    if (p_1.y() != p_2.y()) {
      return p_1.y() < p_2.y();
    }
    return p_1.z() < p_2.z();
  };
};

template <typename ValueType>
struct Index3DHashMapType {
  typedef std::unordered_map<
      Index3D, ValueType, Index3DHash, std::equal_to<Index3D>,
      Eigen::aligned_allocator<std::pair<const Index3D, ValueType>>>
      type;
};

typedef std::unordered_set<Index3D, Index3DHash, std::equal_to<Index3D>,
                           Eigen::aligned_allocator<Index3D>>
    Index3DSet;

}  // namespace nvblox
