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

#include <type_traits>
#include <vector>
#include "nvblox/utils/logging.h"

#include "nvblox/core/iterator.h"
#include "nvblox/core/types.h"

namespace nvblox {

/// Unified-memory CUDA vector that should only be used on trivial types
/// as the constructors and destructors *are NOT called*.
template <typename T>
class unified_vector {
 public:
  typedef RawIterator<T> iterator;
  typedef RawIterator<const T> const_iterator;

  static constexpr MemoryType kDefaultMemoryType = MemoryType::kUnified;

  /// Static asserts on the type.
  static_assert(
      std::is_default_constructible<T>::value,
      "Objects stored in unified vector should be default constructible.");

  unified_vector(MemoryType memory_type = kDefaultMemoryType);
  unified_vector(size_t size, MemoryType memory_type = kDefaultMemoryType);
  unified_vector(size_t size, const T& initial,
                 MemoryType memory_type = kDefaultMemoryType);
  /// Copy constructor.
  unified_vector(const unified_vector<T>& other,
                 MemoryType memory_type = kDefaultMemoryType);
  unified_vector(const std::vector<T>& other,
                 MemoryType memory_type = kDefaultMemoryType);

  /// Move constructor.
  unified_vector(unified_vector<T>&& other);

  /// Destructor
  virtual ~unified_vector();

  /// Operators.
  T& operator[](size_t index);
  const T& operator[](size_t index) const;
  unified_vector<T>& operator=(const unified_vector<T>& other);
  unified_vector<T>& operator=(unified_vector<T>&& other);
  unified_vector<T>& operator=(const std::vector<T>& other);

  /// Convert to an std::vector. Creates a copy.
  std::vector<T> toVector() const;

  /// Get raw pointers. This is also for GPU pointers.
  T* data();
  const T* data() const;

  /// Hint to move the memory to the GPU or CPU.
  void toGPU();
  void toCPU();

  /// Access information.
  size_t capacity() const;
  size_t size() const;
  bool empty() const;

  /// Changing the size.
  void reserve(size_t capacity);
  void resize(size_t size);
  void clear();

  /// Adding elements.
  void push_back(const T& value);

  /// Iterator access.
  iterator begin();
  iterator end();
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  const_iterator cbegin() const;
  const_iterator cend() const;

  /// Get the memory type.
  MemoryType memory_type() const { return memory_type_; }

  /// Set the entire *memory* of the vector to zero.
  void setZero();

 private:
  MemoryType memory_type_;

  T* buffer_;
  size_t buffer_size_;
  size_t buffer_capacity_;
};

/// Specialization for unified_vector on device memory only.
template <typename T>
class device_vector : public unified_vector<T> {
 public:
  device_vector() : unified_vector<T>(MemoryType::kDevice) {}
  device_vector(MemoryType memory_type) = delete;
  device_vector(size_t size) : unified_vector<T>(size, MemoryType::kDevice) {}
  device_vector(size_t size, const T& initial)
      : unified_vector<T>(size, initial, MemoryType::kDevice) {}

  // Conversions from base
  device_vector(const unified_vector<T>& other)
      : unified_vector<T>(other, MemoryType::kDevice){};
  device_vector(unified_vector<T>&& other)
      : unified_vector<T>(std::move(other), MemoryType::kDevice){};
  device_vector& operator=(const unified_vector<T>& other) {
    unified_vector<T>::operator=(other);
    return *this;
  }
  device_vector& operator=(unified_vector<T>&& other) {
    unified_vector<T>::operator=(std::move(other));
    return *this;
  }

  // From std::vector
  device_vector(const std::vector<T>& other)
      : unified_vector<T>(other, MemoryType::kDevice) {}
  device_vector<T>& operator=(const std::vector<T>& other) {
    unified_vector<T>::operator=(other);
    return *this;
  }
};

/// Specialization for unified_vector on host memory only.
template <typename T>
class host_vector : public unified_vector<T> {
 public:
  host_vector() : unified_vector<T>(MemoryType::kHost) {}
  host_vector(MemoryType memory_type) = delete;
  host_vector(size_t size) : unified_vector<T>(size, MemoryType::kHost) {}
  host_vector(size_t size, const T& initial)
      : unified_vector<T>(size, initial, MemoryType::kHost) {}

  // Conversions from base
  host_vector(const unified_vector<T>& other)
      : unified_vector<T>(other, MemoryType::kHost){};
  host_vector(unified_vector<T>&& other)
      : unified_vector<T>(std::move(other), MemoryType::kHost){};
  host_vector& operator=(const unified_vector<T>& other) {
    unified_vector<T>::operator=(other);
    return *this;
  }
  host_vector& operator=(unified_vector<T>&& other) {
    unified_vector<T>::operator=(std::move(other));
    return *this;
  }

  // From std::vector
  host_vector(const std::vector<T>& other)
      : unified_vector<T>(other, MemoryType::kHost) {}
  host_vector<T>& operator=(const std::vector<T>& other) {
    unified_vector<T>::operator=(other);
    return *this;
  }
};

}  // namespace nvblox

#include "nvblox/core/internal/impl/unified_vector_impl.h"
