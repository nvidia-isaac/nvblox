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

#include <cuda_runtime.h>
#include <cstddef>
#include <iterator>

namespace nvblox {

/// Iterator class for unified_vectors that enables us to use thrust and STL
/// operators on our own vectors.
template <typename T>
struct RawIterator {
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;

  __host__ __device__ RawIterator(pointer ptr) : ptr_(ptr) {}

  __host__ __device__ reference operator*() const { return *ptr_; }
  __host__ __device__ pointer operator->() { return ptr_; }
  __host__ __device__ RawIterator& operator++() {
    ptr_++;
    return *this;
  }
  __host__ __device__ RawIterator& operator--() {
    ptr_--;
    return *this;
  }
  __host__ __device__ RawIterator& operator++(int) {
    RawIterator tmp = *this;
    ++(*this);
    return tmp;
  }
  __host__ __device__ RawIterator& operator+=(const difference_type& movement) {
    ptr_ += movement;
    return *this;
  }

  // Suspiciously need this for CUB.
  __host__ __device__ RawIterator& operator=(const T& value) {
    *ptr_ = value;
    return *this;
  }

  __host__ __device__ RawIterator
  operator+(const difference_type& movement) const {
    pointer old_ptr = ptr_;
    old_ptr += movement;
    return RawIterator(old_ptr);
  }
  __host__ __device__ RawIterator
  operator-(const difference_type& movement) const {
    pointer old_ptr = ptr_;
    old_ptr -= movement;
    return RawIterator(old_ptr);
  }

  __host__ __device__ RawIterator
  operator[](const difference_type& movement) const {
    pointer old_ptr = ptr_;
    old_ptr += movement;
    return RawIterator(old_ptr);
  }

  // These two are needed for thrust:
  __host__ __device__ operator T*() const { return ptr_; }
  __host__ __device__ operator T() const { return *ptr_; }

  __host__ __device__ friend bool operator==(const RawIterator& a,
                                             const RawIterator& b) {
    return a.ptr_ == b.ptr_;
  };
  __host__ __device__ friend bool operator!=(const RawIterator& a,
                                             const RawIterator& b) {
    return a.ptr_ != b.ptr_;
  };

 private:
  pointer ptr_;
};

}  // namespace nvblox
