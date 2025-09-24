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

#ifdef __CUDACC__

#include "cub/thread/thread_operators.cuh"
#include "thrust/functional.h"

namespace nvblox {
// To maintain support for CUDA11 we wrap certain operators that changed name in
// more recent versions.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12)
template <class T>
using not_equal_to = ::cuda::std::not_equal_to<T>;
#else
template <class T>
using not_equal_to = cub::Inequality;  // Note that we wrap cub inside
                                       // the nvblox namespace.
#endif
}  // namespace nvblox

#endif
