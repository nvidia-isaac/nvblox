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
#include <glog/logging.h>

#include "nvblox/core/cuda/error_check.cuh"

namespace nvblox {

// Single object
template <typename T, typename... Args>
typename _Unified_if<T>::_Single_object make_unified(Args&&... args) {
  return make_unified<T>(unified_ptr<T>::kDefaultMemoryType,
                         std::forward<Args>(args)...);
}

template <typename T, typename... Args>
typename _Unified_if<T>::_Single_object make_unified(MemoryType memory_type,
                                                     Args&&... args) {
  T* cuda_ptr = nullptr;
  if (memory_type == MemoryType::kDevice) {
    // No constructor (or destructor, hence the check)
    CHECK(std::is_trivially_destructible<T>::value);
    checkCudaErrors(cudaMalloc(&cuda_ptr, sizeof(T)));
    return unified_ptr<T>(cuda_ptr, memory_type);
  } else if (memory_type == MemoryType::kUnified) {
    // Constructor called
    checkCudaErrors(
        cudaMallocManaged(&cuda_ptr, sizeof(T), cudaMemAttachGlobal));
    return unified_ptr<T>(new (cuda_ptr) T(args...), memory_type);
  } else {
    // Constructor called
    checkCudaErrors(cudaMallocHost(&cuda_ptr, sizeof(T)));
    return unified_ptr<T>(new (cuda_ptr) T(args...), memory_type);
  }
}

// Array
template <typename T>
typename _Unified_if<T>::_Unknown_bound make_unified(std::size_t size) {
  return make_unified<T>(size, unified_ptr<T>::kDefaultMemoryType);
}

template <typename T>
typename _Unified_if<T>::_Unknown_bound make_unified(std::size_t size,
                                                     MemoryType memory_type) {
  typedef typename std::remove_extent<T>::type TNonArray;
  TNonArray* cuda_ptr = nullptr;
  if (memory_type == MemoryType::kDevice) {
    // No constructor
    checkCudaErrors(cudaMalloc(&cuda_ptr, sizeof(TNonArray) * size));
    return unified_ptr<T>(cuda_ptr, memory_type, size);
  } else if (memory_type == MemoryType::kUnified) {
    // Default constructor
    checkCudaErrors(cudaMallocManaged(&cuda_ptr, sizeof(TNonArray) * size,
                                      cudaMemAttachGlobal));
    return unified_ptr<T>(new (cuda_ptr) TNonArray[size], memory_type, size);
  } else {
    // Default constructor
    checkCudaErrors(cudaMallocHost(&cuda_ptr, sizeof(TNonArray) * size));
    return unified_ptr<T>(new (cuda_ptr) TNonArray[size], memory_type, size);
  }
}

// Default constructor.
template <typename T>
unified_ptr<T>::unified_ptr()
    : memory_type_(kDefaultMemoryType), ptr_(nullptr), ref_counter_(nullptr) {}

// Useful constructor.
template <typename T>
unified_ptr<T>::unified_ptr(T_noextent* ptr, MemoryType memory_type,
                            size_t size)
    : memory_type_(memory_type),
      size_(size),
      ptr_(ptr),
      ref_counter_(new std::atomic<int>(1)) {}

// Copy constructor.
template <typename T>
unified_ptr<T>::unified_ptr(const unified_ptr<T>& other)
    : memory_type_(other.memory_type_),
      size_(other.size_),
      ptr_(other.ptr_),
      ref_counter_(other.ref_counter_) {
  if (ptr_ != nullptr) {
    (*ref_counter_)++;
  }
}

// Move constructor.
template <typename T>
unified_ptr<T>::unified_ptr(unified_ptr<T>&& other)
    : memory_type_(other.memory_type_),
      size_(other.size_),
      ptr_(other.ptr_),
      ref_counter_(other.ref_counter_) {
  if (ptr_ != nullptr) {
    (*ref_counter_)++;
  }
}

template <typename T>
struct Deleter {
  static void destroy(T* ptr, MemoryType memory_type) {
    if (memory_type == MemoryType::kUnified) {
      ptr->~T();
      checkCudaErrors(
          cudaFree(const_cast<void*>(reinterpret_cast<void const*>(ptr))));
    } else if (memory_type == MemoryType::kDevice) {
      checkCudaErrors(
          cudaFree(const_cast<void*>(reinterpret_cast<void const*>(ptr))));
    } else {
      ptr->~T();
      checkCudaErrors(
          cudaFreeHost(const_cast<void*>(reinterpret_cast<void const*>(ptr))));
    }
  }
};

template <typename T>
struct Deleter<T[]> {
  static void destroy(T* ptr, MemoryType memory_type) {
    // Do nothing at run-time
    static_assert(
        std::is_trivially_destructible<T>::value,
        "Objects stored in unified_ptr<T[]> must be trivially destructible.");
    if (memory_type == MemoryType::kDevice ||
        memory_type == MemoryType::kUnified) {
      checkCudaErrors(
          cudaFree(const_cast<void*>(reinterpret_cast<void const*>(ptr))));
    } else {
      checkCudaErrors(
          cudaFreeHost(const_cast<void*>(reinterpret_cast<void const*>(ptr))));
    }
  }
};

// Destructor.
template <typename T>
unified_ptr<T>::~unified_ptr() {
  if (ptr_ != nullptr) {
    (*ref_counter_)--;
    if (*ref_counter_ <= 0) {
      Deleter<T>::destroy(ptr_, memory_type_);
      delete ref_counter_;
      ptr_ = nullptr;
      ref_counter_ = nullptr;
    }
  }
}

// Operators
template <typename T>
unified_ptr<T>& unified_ptr<T>::operator=(const unified_ptr<T>& other) {
  reset();
  ptr_ = other.ptr_;
  ref_counter_ = other.ref_counter_;
  memory_type_ = other.memory_type_;
  if (ptr_ != nullptr) {
    (*ref_counter_)++;
  }
  return *this;
}

template <typename T>
unified_ptr<T>& unified_ptr<T>::operator=(unified_ptr<T>&& other) {
  reset();
  ptr_ = other.ptr_;
  ref_counter_ = other.ref_counter_;
  memory_type_ = other.memory_type_;
  if (ptr_ != nullptr) {
    (*ref_counter_)++;
  }
  return *this;
}

template <typename T>
template <typename T2, std::enable_if_t<!std::is_const<T2>::value, bool>>
unified_ptr<T>::operator unified_ptr<const T2>() const {
  unified_ptr<const T2> const_ptr;
  if (ptr_ != nullptr) {
    const_ptr.ptr_ = ptr_;
    const_ptr.ref_counter_ = ref_counter_;
    const_ptr.memory_type_ = memory_type_;
    (*const_ptr.ref_counter_)++;
  }
  return const_ptr;
}

// Operator bool
template <typename T>
unified_ptr<T>::operator bool() const {
  return ptr_ != nullptr;
}

// Operator comparison
template <typename T>
bool unified_ptr<T>::operator==(const unified_ptr<T>& other) const {
  return ptr_ == other.ptr_;
}

template <typename T>
bool unified_ptr<T>::operator!=(const unified_ptr<T>& other) const {
  return ptr_ != other.ptr_;
}

template <typename T>
typename unified_ptr<T>::T_noextent* unified_ptr<T>::operator->() {
  CHECK(memory_type_ != MemoryType::kDevice);
  return ptr_;
}

template <typename T>
T& unified_ptr<T>::operator*() {
  CHECK(memory_type_ != MemoryType::kDevice);
  return *ptr_;
}

template <typename T>
const typename unified_ptr<T>::T_noextent* unified_ptr<T>::operator->() const {
  CHECK(memory_type_ != MemoryType::kDevice);
  return ptr_;
}

template <typename T>
const T& unified_ptr<T>::operator*() const {
  CHECK(memory_type_ != MemoryType::kDevice);
  return *ptr_;
}

// Getters
template <typename T>
typename unified_ptr<T>::T_noextent* unified_ptr<T>::get() {
  return ptr_;
}

template <typename T>
const typename unified_ptr<T>::T_noextent* unified_ptr<T>::get() const {
  return ptr_;
}

// Reset
template <typename T>
void unified_ptr<T>::reset() {
  if (ptr_ != nullptr) {
    (*ref_counter_)--;
    if (*ref_counter_ <= 0) {
      Deleter<T>::destroy(ptr_, memory_type_);
      delete ref_counter_;
    }
    ptr_ = nullptr;
    ref_counter_ = nullptr;
  }
}

// Cloner
template <typename T>
struct Cloner {
  typedef typename std::remove_cv<T>::type T_nonconst;
  static unified_ptr<T_nonconst> clone(const unified_ptr<T>& original,
                                       MemoryType memory_type, size_t) {
    auto other = make_unified<T_nonconst>(memory_type);
    cudaMemcpy(other.get(), original.get(), sizeof(T), cudaMemcpyDefault);
    return other;
  }
};

template <typename T>
struct Cloner<T[]> {
  typedef typename std::remove_extent<T>::type T_noextent;
  typedef typename std::remove_cv<T>::type T_nonconst;
  static unified_ptr<T_nonconst[]> clone(const unified_ptr<T[]>& original,
                                         MemoryType memory_type, size_t size) {
    auto other = make_unified<T_nonconst[]>(size, memory_type);
    cudaMemcpy(other.get(), original.get(), sizeof(T_noextent) * size,
               cudaMemcpyDefault);
    return other;
  }
};

template <typename T>
unified_ptr<typename std::remove_cv<T>::type> unified_ptr<T>::clone() const {
  return Cloner<T>::clone(*this, memory_type_, size_);
}

template <typename T>
unified_ptr<typename std::remove_cv<T>::type> unified_ptr<T>::clone(
    MemoryType memory_type) const {
  return Cloner<T>::clone(*this, memory_type, size_);
}

// Hint to move to GPU or CPU.
template <typename T>
void unified_ptr<T>::toGPU() {
  CHECK(memory_type_ == MemoryType::kUnified);
  if (ptr_ == nullptr) {
    return;
  }
  int device;
  cudaGetDevice(&device);
  checkCudaErrors(cudaMemPrefetchAsync(ptr_, sizeof(T), device));
}

template <typename T>
void unified_ptr<T>::toGPU(cudaStream_t cuda_stream) {
  CHECK(memory_type_ == MemoryType::kUnified);
  if (ptr_ == nullptr) {
    return;
  }
  int device;
  cudaGetDevice(&device);
  checkCudaErrors(cudaMemPrefetchAsync(ptr_, sizeof(T), device, cuda_stream));
}

template <typename T>
void unified_ptr<T>::toCPU() {
  CHECK(memory_type_ == MemoryType::kUnified);
  if (ptr_ == nullptr) {
    return;
  }
  checkCudaErrors(cudaMemPrefetchAsync(ptr_, sizeof(T), cudaCpuDeviceId));
}

template <typename T>
void unified_ptr<T>::preferGPU() {
  CHECK(memory_type_ == MemoryType::kUnified);
  int device;
  cudaGetDevice(&device);
  checkCudaErrors(cudaMemAdvise(ptr_, sizeof(T),
                                cudaMemAdviseSetPreferredLocation, device));
}

template <typename T>
void unified_ptr<T>::setZero() {
  checkCudaErrors(cudaMemset(ptr_, 0, sizeof(T)));
}
}  // namespace nvblox
