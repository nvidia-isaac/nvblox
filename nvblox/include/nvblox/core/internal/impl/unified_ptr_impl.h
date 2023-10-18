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
#include "nvblox/utils/logging.h"

#include "nvblox/core/internal/error_check.h"

namespace nvblox {

// Single object
template <typename T, typename... Args>
typename _Unified_if<T>::_Single_object make_unified(Args&&... args) {
  return make_unified<T>(unified_ptr<T>::kDefaultMemoryType,
                         std::forward<Args>(args)...);
}

template <typename T, typename... Args>
typename _Unified_if<T>::_Single_object make_unified_async(
    MemoryType memory_type, const CudaStream& cuda_stream, Args&&... args) {
  T* cuda_ptr = nullptr;
  if (memory_type == MemoryType::kDevice) {
    // No constructor (or destructor, hence the check)
    CHECK(std::is_trivially_destructible<T>::value);
    checkCudaErrors(cudaMallocAsync(&cuda_ptr, sizeof(T), cuda_stream));
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

template <typename T, typename... Args>
typename _Unified_if<T>::_Single_object make_unified(MemoryType memory_type,
                                                     Args&&... args) {
  return make_unified_async<T>(memory_type, CudaStreamOwning(), args...);
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
  size_ = other.size_;
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
  size_ = other.size_;
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
    const_ptr.size_ = size_;
    (*const_ptr.ref_counter_)++;
  }
  return const_ptr;
}

template <typename T>
template <typename T2, typename std::enable_if<std::is_base_of<T2, T>{} &&
                                                   !std::is_const<T2>{} &&
                                                   !std::is_const<T>{},
                                               bool>::type>
unified_ptr<T>::operator unified_ptr<T2>() const {
  unified_ptr<T2> base_class_ptr;
  if (ptr_ != nullptr) {
    base_class_ptr.ptr_ = dynamic_cast<T2*>(ptr_);
    base_class_ptr.ref_counter_ = ref_counter_;
    base_class_ptr.memory_type_ = memory_type_;
    base_class_ptr.size_ = size_;
    (*base_class_ptr.ref_counter_)++;
  }
  return base_class_ptr;
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
    size_ = 1;
  }
}

// Cloner
template <typename T>
struct Cloner {
  typedef typename std::remove_cv<T>::type T_nonconst;
  static unified_ptr<T_nonconst> cloneAsync(const unified_ptr<T>& original,
                                            MemoryType memory_type, size_t,
                                            const CudaStream cuda_stream) {
    CHECK(original.get() != nullptr);
    CHECK(!(memory_type == MemoryType::kUnified &&
            original.memory_type() == MemoryType::kUnified))
        << "Cloning between two unified memory areas is not allowed since "
           "it's not supported on all devices (need "
           "concurrentManagedAccess=1)";
    auto other = make_unified<T_nonconst>(memory_type);
    checkCudaErrors(cudaMemcpyAsync(other.get(), original.get(), sizeof(T),
                                    cudaMemcpyDefault, cuda_stream));
    return other;
  }
};

template <typename T>
struct Cloner<T[]> {
  typedef typename std::remove_extent<T>::type T_noextent;
  typedef typename std::remove_cv<T>::type T_nonconst;
  static unified_ptr<T_nonconst[]> cloneAsync(const unified_ptr<T[]>& original,
                                              MemoryType memory_type,
                                              size_t size,
                                              const CudaStream cuda_stream) {
    CHECK(original.get() != nullptr);
    auto other = make_unified<T_nonconst[]>(size, memory_type);
    checkCudaErrors(cudaMemcpyAsync(other.get(), original.get(),
                                    sizeof(T_noextent) * size,
                                    cudaMemcpyDefault, cuda_stream));
    return other;
  }
};

template <typename T>
unified_ptr<typename std::remove_cv<T>::type> unified_ptr<T>::clone() const {
  return Cloner<T>::cloneAsync(*this, memory_type_, size_, CudaStreamOwning());
}

template <typename T>
unified_ptr<typename std::remove_cv<T>::type> unified_ptr<T>::clone(
    MemoryType memory_type) const {
  return Cloner<T>::cloneAsync(*this, memory_type, size_, CudaStreamOwning());
}

template <typename T>
unified_ptr<typename std::remove_cv<T>::type> unified_ptr<T>::cloneAsync(
    const CudaStream cuda_stream) const {
  return Cloner<T>::cloneAsync(*this, memory_type_, size_, cuda_stream);
}

template <typename T>
unified_ptr<typename std::remove_cv<T>::type> unified_ptr<T>::cloneAsync(
    MemoryType memory_type, const CudaStream cuda_stream) const {
  return Cloner<T>::cloneAsync(*this, memory_type, size_, cuda_stream);
}

template <typename T>
void unified_ptr<T>::copyToAsync(unified_ptr<T_nonconst>& ptr,
                                 const CudaStream cuda_stream) const {
  CHECK(ptr.get() != nullptr);
  CHECK(ptr.size_ >= size_);
  checkCudaErrors(cudaMemcpyAsync(ptr.get(), this->get(),
                                  sizeof(T_noextent) * size_, cudaMemcpyDefault,
                                  cuda_stream));
}
template <typename T>
void unified_ptr<T>::copyTo(unified_ptr<T_nonconst>& ptr) const {
  copyToAsync(ptr, CudaStreamOwning());
}

template <typename T>
void unified_ptr<T>::copyFromAsync(const T_noextent* const raw_ptr,
                                   const size_t num_elements,
                                   const CudaStream cuda_stream) {
  CHECK(num_elements <= size_);
  checkCudaErrors(cudaMemcpyAsync(this->get(), raw_ptr,
                                  sizeof(T_noextent) * num_elements,
                                  cudaMemcpyDefault, cuda_stream));
}

template <typename T>
void unified_ptr<T>::copyFrom(const T_noextent* const raw_ptr,
                              const size_t num_elements) {
  copyFromAsync(raw_ptr, num_elements, CudaStreamOwning());
}

template <typename T>
void unified_ptr<T>::copyFromAsync(unified_ptr<T_nonconst>& ptr,
                                   const CudaStream cuda_stream) {
  ptr.copyToAsync(*this, cuda_stream);
}

template <typename T>
void unified_ptr<T>::copyFrom(unified_ptr<T_nonconst>& ptr) {
  ptr.copyTo(*this);
}

template <typename T>
void unified_ptr<T>::setZero() {
  setZeroAsync(CudaStreamOwning());
}

template <typename T>
void unified_ptr<T>::setZeroAsync(const CudaStream cuda_stream) {
  CHECK(ptr_ != nullptr);
  checkCudaErrors(
      cudaMemsetAsync(ptr_, 0, sizeof(T_noextent) * size_, cuda_stream));
}

}  // namespace nvblox
