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
#include <atomic>
#include <memory>
#include <type_traits>
#include "nvblox/core/cuda_stream.h"
#include "nvblox/utils/logging.h"

#include "nvblox/core/types.h"

namespace nvblox {

/// shared_ptr for device, unified memory, and pinned host memory
/// Things to be aware of
/// - Single objects
///   - Constructor and Destructor are not called when memory_type==kDevice
///     (these are CPU functions). Therefore unified_ptr generates an error if
///     used in device mode with non-trivially destructible types.
///   - Both Constructor and Destructor are called when memory_type==kUnified ||
///     kHost.
/// - Arrays
///   - Default Constructor called when storing arrays in kUnified || kHost.
///   - No Constructor called in kDevice mode.
///   - No destructor called for ANY memory setting so we make a static check
///     that objects are trivially destructable.
template <typename T>
class unified_ptr {
 public:
  typedef typename std::remove_extent<T>::type T_noextent;
  typedef typename std::remove_cv<T>::type T_nonconst;

  static constexpr MemoryType kDefaultMemoryType = MemoryType::kUnified;

  unified_ptr();
  explicit unified_ptr(T_noextent* ptr, MemoryType memory_type,
                       size_t size = 1);
  unified_ptr(const unified_ptr<T>& other);
  unified_ptr(unified_ptr<T>&& other);
  ~unified_ptr();

  /// Operator =
  unified_ptr<T>& operator=(const unified_ptr<T>& other);
  unified_ptr<T>& operator=(unified_ptr<T>&& other);

  /// Operator bool
  operator bool() const;
  /// Operator comparison
  bool operator==(const unified_ptr<T>& other) const;
  bool operator!=(const unified_ptr<T>& other) const;

  /// Operator dereference
  T_noextent* operator->();
  const T_noextent* operator->() const;
  T& operator*();
  const T& operator*() const;

  /// Operator array access
  /// Only enabled if the underlying type is an array
  template <typename T1 = T,
            std::enable_if_t<std::is_array<T1>::value, bool> = true>
  T_noextent& operator[](size_t i) {
    CHECK(memory_type_ != MemoryType::kDevice);
    return ptr_[i];
  }
  template <typename T1 = T,
            std::enable_if_t<std::is_array<T1>::value, bool> = true>
  const T_noextent& operator[](size_t i) const {
    CHECK(memory_type_ != MemoryType::kDevice);
    return ptr_[i];
  }

  /// Operator convert
  /// Only enabled if the base type is NOT const, otherwise adds a second
  /// trivial converter.
  template <typename T2,
            std::enable_if_t<!std::is_const<T2>::value, bool> = true>
  operator unified_ptr<const T2>() const;

  /// Operator convert to base class.
  template <typename T2, typename std::enable_if<std::is_base_of<T2, T>{} &&
                                                     !std::is_const<T2>{} &&
                                                     !std::is_const<T>{},
                                                 bool>::type = true>
  operator unified_ptr<T2>() const;

  /// Get the raw pointer.
  T_noextent* get();
  const T_noextent* get() const;

  /// Reset the pointer to point to nothing.
  void reset();

  /// Copy the underlying object (potentially to another memory location)
  /// NOTE: This is implemented as a memcpy at the pointed to location.
  unified_ptr<T_nonconst> clone() const;
  unified_ptr<T_nonconst> clone(MemoryType memory_type) const;

  unified_ptr<T_nonconst> cloneAsync(const CudaStream cuda_stream) const;
  unified_ptr<T_nonconst> cloneAsync(MemoryType memory_type,
                                     const CudaStream cuda_stream) const;

  /// Copy memory between two unified ptrs, potentially of different memory
  /// types.
  void copyTo(unified_ptr<T_nonconst>& ptr) const;
  void copyToAsync(unified_ptr<T_nonconst>& ptr,
                   const CudaStream cuda_stream) const;
  void copyFrom(unified_ptr<T_nonconst>& ptr);
  void copyFromAsync(unified_ptr<T_nonconst>& ptr,
                     const CudaStream cuda_stream);

  /// Copy from a raw pointer
  void copyFrom(const T_noextent* const raw_ptr, const size_t num_elements);
  void copyFromAsync(const T_noextent* const raw_ptr, const size_t num_elements,
                     const CudaStream cuda_stream);

  MemoryType memory_type() const { return memory_type_; }

  /// Helper function to memset all the memory to 0.
  void setZeroAsync(const CudaStream cuda_stream);
  void setZero();

  // Unified pointer has heaps of friends.
  friend class unified_ptr<T_nonconst>;
  friend class unified_ptr<const T>;
  template <class U>
  friend class unified_ptr;

 private:
  MemoryType memory_type_;
  size_t size_;

  T_noextent* ptr_;

  mutable std::atomic<int>* ref_counter_;
};

// Comparison with nullptr
template <class T>
bool operator==(const unified_ptr<T>& lhs, std::nullptr_t) noexcept {
  return !lhs;
}
template <class T>
bool operator==(std::nullptr_t, const unified_ptr<T>& rhs) noexcept {
  return !rhs;
}
template <class T>
bool operator!=(const unified_ptr<T>& lhs, std::nullptr_t) noexcept {
  return lhs;
}
template <class T>
bool operator!=(std::nullptr_t, const unified_ptr<T>& rhs) noexcept {
  return rhs;
}

// These are used to specialize make_unique on the return type, which is need to
// generate class templates for non-array and array types. See gcc here:
// https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/unique_ptr.h#L943
template <class T>
struct _Unified_if {
  typedef unified_ptr<T> _Single_object;
};

template <class T>
struct _Unified_if<T[]> {
  typedef unified_ptr<T[]> _Unknown_bound;
};

template <class T, size_t N>
struct _Unified_if<T[N]> {
  typedef void _Known_bound;
};

// Single object, default storage location
template <typename T, typename... Args>
typename _Unified_if<T>::_Single_object make_unified(Args&&... args);

// Specialization, Single object, specified storage location
template <typename T, typename... Args>
typename _Unified_if<T>::_Single_object make_unified(MemoryType memory_type,
                                                     Args&&... args);

// Specialization, Single object, specified storage location, async
template <typename T, typename... Args>
typename _Unified_if<T>::_Single_object make_unified_async(
    MemoryType, const CudaStream& cuda_stream, Args&&... args);

// Specialization for arrays, default storage location
template <typename T>
typename _Unified_if<T>::_Unknown_bound make_unified(std::size_t size);

// Specialization for arrays, specified storage location
template <typename T>
typename _Unified_if<T>::_Unknown_bound make_unified(std::size_t size,
                                                     MemoryType memory_type);

template <typename T, typename... Args>
typename _Unified_if<T>::_Known_bound make_unified(Args&&... args) = delete;

}  // namespace nvblox

#include "nvblox/core/internal/impl/unified_ptr_impl.h"
