/*
Copyright 2025 NVIDIA CORPORATION

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

#include <type_traits>

namespace nvblox {

/// Compute ceiling division for kernel block calculations.
/// Returns ceil(num_elements / num_elements_per_block) without floating point
/// operations. This is the correct way to calculate the number of blocks needed
/// to cover a given number of elements with a fixed block size.
///
/// Example Usage:
///   const dim3 num_blocks(
///     divideRoundUp(depth_ptr->cols(), kThreadsPerThreadBlock.x),
///     divideRoundUp(depth_ptr->rows(), kThreadsPerThreadBlock.y), 1);
///
/// Example Output:
///   divideRoundUp(64, 8) = 8  (exact division)
///   divideRoundUp(65, 8) = 9  (rounds up)
///
/// @param num_elements The total number of elements to process
/// @param num_elements_per_block The number of elements per block
/// @return The minimum number of blocks needed
template <typename T1, typename T2>
constexpr auto divideRoundUp(T1 num_elements, T2 num_elements_per_block) {
  static_assert(std::is_integral<T1>::value && std::is_integral<T2>::value,
                "divideRoundUp only works with integer types");
  return (num_elements + num_elements_per_block - 1) / num_elements_per_block;
}

}  // namespace nvblox
