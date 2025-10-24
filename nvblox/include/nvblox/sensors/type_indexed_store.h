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
#include <memory>
#include <typeindex>
#include <unordered_map>

#include "nvblox/sensors/sensor.h"

namespace nvblox {

/// Generic type-indexed store. Stores one value per IndexedType.
class TypeIndexedStore {
 public:
  /// Set value for IndexedType (copy)
  template <typename IndexedType>
  void set(const IndexedType& value);

  /// Set value for IndexedType (move)
  template <typename IndexedType>
  void set(IndexedType&& value);

  /// Check if a value exists for IndexedType
  template <typename IndexedType>
  bool hasType() const;

  /// Get const pointer to value for IndexedType
  template <typename IndexedType>
  const IndexedType* getPtr() const;

  /// Get const reference to value for IndexedType
  template <typename IndexedType>
  const IndexedType& get() const;

  /// Get mutable pointer to value for IndexedType
  template <typename IndexedType>
  IndexedType* getMutablePtr();

 private:
  struct StorageBase {
    virtual ~StorageBase() = default;
  };

  template <typename IndexedType>
  struct Storage : StorageBase {
    explicit Storage(const IndexedType& v) : value(v) {}
    explicit Storage(IndexedType&& v) : value(std::move(v)) {}
    IndexedType value;
  };

  template <typename IndexedType>
  std::type_index idxFromType() const {
    return std::type_index(typeid(IndexedType));
  }

  std::unordered_map<std::type_index, std::unique_ptr<StorageBase>> store_;
};

}  // namespace nvblox

#include "nvblox/sensors/internal/impl/type_indexed_store_impl.h"
