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
#include <algorithm>
#include <cstring>
#include <type_traits>

#include <memory>

#include "nvblox/core/internal/error_check.h"

namespace nvblox {

// Constructors and destructors.
template <typename T>
unified_vector<T>::unified_vector(MemoryType memory_type)
    : memory_type_(memory_type),
      buffer_(nullptr),
      buffer_size_(0),
      buffer_capacity_(0) {
  static_assert(std::is_default_constructible<T>::value,
                "Need to have a default constructor to use unified_vector.");
  // NOTE(alexmillane): Some structures (eg. Ray) define custom destructors,
  // (empty in Ray's case), even though they only contain data by value, and can
  // be stored by unified_vector. We therefore have to remove the check below.
  // The onus is therefore transferred to the user to not use the vector for
  // stuff which needs a destructor called.
  // static_assert(std::is_trivially_destructible<T>::value,
  //               "Need to have a trivial destructor to use unified_vector.");
}

template <typename T>
unified_vector<T>::unified_vector(size_t size, MemoryType memory_type,
                                  const CudaStream& cuda_stream)
    : unified_vector(memory_type) {
  resizeAsync(size, cuda_stream);
}

template <typename T>
unified_vector<T>::unified_vector(size_t size, const T& initial,
                                  MemoryType memory_type,
                                  const CudaStream& cuda_stream)
    : unified_vector(memory_type) {
  CHECK_NE(static_cast<int>(memory_type_),
           static_cast<int>(MemoryType::kDevice))
      << "Can't set initial values for device memory. Use copy instead.";
  resizeAsync(size, cuda_stream);
  for (size_t i = 0; i < size; i++) {
    buffer_[i] = initial;
  }
}

// Move constructor.
template <typename T>
unified_vector<T>::unified_vector(unified_vector<T>&& other)
    : memory_type_(other.memory_type_),
      buffer_(other.buffer_),
      buffer_size_(other.buffer_size_),
      buffer_capacity_(other.buffer_capacity_) {
  other.buffer_ = nullptr;
  other.buffer_size_ = 0;
  other.buffer_capacity_ = 0;
}

template <typename T>
unified_vector<T>::~unified_vector() {
  clearAndDeallocate();
}

// Operators
template <typename T>
T& unified_vector<T>::operator[](size_t index) {
  return buffer_[index];
}

template <typename T>
const T& unified_vector<T>::operator[](size_t index) const {
  return buffer_[index];
}

// Move assignment
template <typename T>
unified_vector<T>& unified_vector<T>::operator=(unified_vector<T>&& other) {
  clearAndDeallocate();
  buffer_ = other.buffer_;
  buffer_size_ = other.buffer_size_;
  buffer_capacity_ = other.buffer_capacity_;
  memory_type_ = other.memory_type_;
  other.buffer_ = nullptr;
  other.buffer_size_ = 0;
  other.buffer_capacity_ = 0;
  return *this;
}

template <typename T>
template <typename OtherVectorType>
void unified_vector<T>::copyFromAsync(const OtherVectorType& other,
                                      const CudaStream& cuda_stream) {
  copyFromAsync(other.data(), other.size(), cuda_stream);
}

template <typename T>
void unified_vector<T>::copyFromAsync(const T_noextent* const raw_ptr,
                                      const size_t num_elements,
                                      const CudaStream& cuda_stream) {
  resizeAsync(num_elements, cuda_stream);
  if (raw_ptr != nullptr) {
    checkCudaErrors(cudaMemcpyAsync(buffer_, raw_ptr, sizeof(T) * num_elements,
                                    cudaMemcpyDefault, cuda_stream));
  }
}

template <typename T>
void unified_vector<T>::copyFromAsync2D(const T_noextent* const raw_ptr,
                                      const size_t s_stride_num_elements,
                                      const size_t d_stride_num_elements,
                                      const size_t row,
                                      const size_t col,
                                      const CudaStream& cuda_stream) {
  resizeAsync(row * d_stride_num_elements, cuda_stream);
  if (raw_ptr != nullptr) {
    checkCudaErrors(cudaMemcpy2DAsync(buffer_, sizeof(T) * d_stride_num_elements, 
                                      raw_ptr, sizeof(T) * s_stride_num_elements, 
                                      sizeof(T) * col, row,
                                      cudaMemcpyDefault, cuda_stream));
  }
}

template <typename T>
void unified_vector<T>::copyToAsync(T_noextent* raw_ptr,
                                    const CudaStream& cuda_stream) const {
  CHECK(raw_ptr != nullptr);
  checkCudaErrors(cudaMemcpyAsync(raw_ptr, this->data(),
                                  sizeof(T_noextent) * buffer_size_,
                                  cudaMemcpyDefault, cuda_stream));
}

template <typename T>
std::vector<T> unified_vector<T>::toVectorAsync(
    const CudaStream& cuda_stream) const {
  static_assert(!std::is_same<T, bool>::value);

  if (buffer_ == nullptr || buffer_size_ == 0) {
    return std::vector<T>();
  }
  std::vector<T> vect(buffer_size_);
  checkCudaErrors(cudaMemcpyAsync(vect.data(), buffer_,
                                  sizeof(T) * buffer_size_, cudaMemcpyDefault,
                                  cuda_stream));
  return vect;
}

// Specialization for bool
template <>
inline std::vector<bool> unified_vector<bool>::toVectorAsync(
    const CudaStream& cuda_stream) const {
  // The memory layout of std::vector<bool> is different so we have to first
  // copy to an intermediate buffer.
  CHECK(buffer_ != nullptr);
  std::unique_ptr<bool[]> bool_buffer(new bool[buffer_size_]);
  checkCudaErrors(cudaMemcpyAsync(bool_buffer.get(), buffer_,
                                  sizeof(bool) * buffer_size_,
                                  cudaMemcpyDefault, cuda_stream));
  // Now populate the vector
  std::vector<bool> vect(buffer_size_);
  for (size_t i = 0; i < buffer_size_; i++) {
    vect[i] = bool_buffer[i];
  }
  return vect;
}

// Get those raw pointers.
template <typename T>
T* unified_vector<T>::data() {
  return buffer_;
}

template <typename T>
const T* unified_vector<T>::data() const {
  return buffer_;
}

// Hint to move the memory to the GPU.
template <typename T>
void unified_vector<T>::toGPU() {
  if (buffer_ == nullptr || buffer_capacity_ == 0) {
    return;
  }
  int device = 0;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(
      cudaMemPrefetchAsync(buffer_, buffer_capacity_ * sizeof(T), device));
}

template <typename T>
void unified_vector<T>::toCPU() {
  if (buffer_ == nullptr || buffer_capacity_ == 0) {
    return;
  }
  checkCudaErrors(cudaMemPrefetchAsync(buffer_, buffer_capacity_ * sizeof(T),
                                       cudaCpuDeviceId));
}

// Accessors
template <typename T>
size_t unified_vector<T>::capacity() const {
  return buffer_capacity_;
}

template <typename T>
size_t unified_vector<T>::size() const {
  return buffer_size_;
}

template <typename T>
bool unified_vector<T>::empty() const {
  return buffer_size_ == 0;
}

// Changing the size.
template <typename T>
void unified_vector<T>::reserveAsync(size_t capacity,
                                     const CudaStream& cuda_stream) {
  if (buffer_capacity_ < capacity) {
    // Create a new buffer.
    T* new_buffer = nullptr;

    if (memory_type_ == MemoryType::kUnified) {
      checkCudaErrors(cudaMallocManaged(&new_buffer, sizeof(T) * capacity,
                                        cudaMemAttachGlobal));
    } else if (memory_type_ == MemoryType::kDevice) {
      checkCudaErrors(
          cudaMallocAsync(&new_buffer, sizeof(T) * capacity, cuda_stream));
    } else {
      checkCudaErrors(cudaMallocHost(&new_buffer, sizeof(T) * capacity));
    }

    if (buffer_ != nullptr) {
      // Copy the old values to the new buffer.
      CHECK(capacity >= buffer_size_);
      checkCudaErrors(cudaMemcpyAsync(new_buffer, buffer_,
                                      sizeof(T) * buffer_size_,
                                      cudaMemcpyDefault, cuda_stream));

      // Delete the old buffer.
      if (memory_type_ == MemoryType::kHost) {
        checkCudaErrors(cudaFreeHost(reinterpret_cast<void*>(buffer_)));
      } else if (memory_type_ == MemoryType::kUnified) {
        checkCudaErrors(cudaFree(reinterpret_cast<void*>(buffer_)));
      } else {
        checkCudaErrors(
            cudaFreeAsync(reinterpret_cast<void*>(buffer_), cuda_stream));
      }
    }
    buffer_ = new_buffer;
    buffer_capacity_ = capacity;
  }
}

template <typename T>
void unified_vector<T>::resizeAsync(size_t size,
                                    const CudaStream& cuda_stream) {
  // ABSOLUTE no-op.
  if (buffer_size_ == size) {
    return;
  } else {
    reserveAsync(size, cuda_stream);
    buffer_size_ = size;
  }
}

template <typename T>
void unified_vector<T>::clearNoDeallocate() {
  buffer_size_ = 0;
}

template <typename T>
void unified_vector<T>::clearAndDeallocate() {
  if (buffer_ != nullptr) {
    if (memory_type_ == MemoryType::kHost) {
      checkCudaErrors(cudaFreeHost(reinterpret_cast<void*>(buffer_)));
    } else {
      checkCudaErrors(cudaFree(reinterpret_cast<void*>(buffer_)));
    }
  }
  buffer_ = nullptr;
  buffer_size_ = 0;
  buffer_capacity_ = 0;
}

template <typename T>
void unified_vector<T>::push_back(const T& value,
                                  const CudaStream& cuda_stream) {
  CHECK_NE(static_cast<int>(memory_type_),
           static_cast<int>(MemoryType::kDevice))
      << "Can't use push_back on device memory.";

  if (buffer_capacity_ < buffer_size_ + 1) {
    // Do the vector thing and reserve double.
    reserveAsync(std::max(buffer_size_ * 2, buffer_size_ + 1), cuda_stream);
    cuda_stream.synchronize();
  }
  // Add the new value to the end.
  buffer_[buffer_size_] = value;
  buffer_size_++;
}  // namespace nvblox

template <typename T>
typename unified_vector<T>::iterator unified_vector<T>::begin() {
  return iterator(&buffer_[0]);
}

template <typename T>
typename unified_vector<T>::iterator unified_vector<T>::end() {
  return iterator(&buffer_[buffer_size_]);
}

template <typename T>
typename unified_vector<T>::const_iterator unified_vector<T>::cbegin() const {
  return const_iterator(&buffer_[0]);
}

template <typename T>
typename unified_vector<T>::const_iterator unified_vector<T>::cend() const {
  return const_iterator(&buffer_[buffer_size_]);
}

template <typename T>
void unified_vector<T>::setZeroAsync(const CudaStream& cuda_stream) {
  // It is safe to use cudaMemset since the memory is ALWAYS allocated with
  // cudaMalloc.
  CHECK(buffer_ != nullptr);
  checkCudaErrors(
      cudaMemsetAsync(buffer_, 0, buffer_size_ * sizeof(T), cuda_stream));
}

template <class... Args>
void expandBuffersIfRequired(size_t required_min_size,
                             const CudaStream& cuda_stream, Args... args) {
  static_assert((std::is_pointer<Args>::value && ...));
  const bool at_least_one_vector_smaller =
      (... || (args->capacity() < required_min_size));
  if (at_least_one_vector_smaller) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size =
        static_cast<int>(kBufferExpansionFactor * required_min_size);
    (args->reserveAsync(new_size, cuda_stream), ...);
  }
}

}  // namespace nvblox
