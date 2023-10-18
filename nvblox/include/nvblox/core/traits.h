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

#include "nvblox/core/types.h"

namespace nvblox {
namespace traits {

// Trait which is true if the BlockType defines an allocate function
// NOTE(alexmillane): This type trait is not totally complete as it doesn't
// check the return type of allocate function... it will do for now. The intent
// is just to give people implementing new blocks a useful message when they
// forget to implement this function.
template <typename T>
struct make_void {
  typedef void type;
};
template <typename T>
using void_t = typename make_void<T>::type;

template <typename, typename = void>
struct has_allocate : std::false_type {};

template <typename BlockType>
struct has_allocate<
    BlockType,
    void_t<decltype(std::declval<BlockType&>().allocate(MemoryType::kUnified))>>
    : std::true_type {};

template <typename, typename = void>
struct has_allocate_async : std::false_type {};

// FIXME: cannot use CudaStreamOwning, since it will break for
// cudaStreamNonOwning()kowning
template <typename BlockType>
struct has_allocate_async<
    BlockType, void_t<decltype(std::declval<BlockType&>().allocateAsync(
                   MemoryType::kUnified, CudaStreamOwning()))>>
    : std::true_type {};

}  // namespace traits
}  // namespace nvblox
