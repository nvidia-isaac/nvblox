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

namespace nvblox {

template <typename IndexedType>
void TypeIndexedStore::set(const IndexedType& value) {
  store_[idxFromType<IndexedType>()] =
      std::make_unique<Storage<IndexedType>>(value);
}

template <typename IndexedType>
void TypeIndexedStore::set(IndexedType&& value) {
  store_[idxFromType<IndexedType>()] =
      std::make_unique<Storage<IndexedType>>(std::move(value));
}

template <typename IndexedType>
bool TypeIndexedStore::hasType() const {
  return store_.count(idxFromType<IndexedType>()) > 0;
}

template <typename IndexedType>
const IndexedType* TypeIndexedStore::getPtr() const {
  auto it = store_.find(idxFromType<IndexedType>());
  NVBLOX_CHECK(it != store_.end(), "Type not found in store");
  NVBLOX_CHECK(it->second.get() != nullptr, "Nullptr in store");
  const auto* storage =
      dynamic_cast<const Storage<IndexedType>*>(it->second.get());
  NVBLOX_CHECK(storage != nullptr, "Dynamic cast failed in store");
  return &storage->value;
}

template <typename IndexedType>
const IndexedType& TypeIndexedStore::get() const {
  const auto* ptr = getPtr<IndexedType>();
  NVBLOX_CHECK(ptr != nullptr, "Nullptr in store");
  return *ptr;
}

template <typename IndexedType>
IndexedType* TypeIndexedStore::getMutablePtr() {
  // Delegate to const version to avoid duplication (Scott Meyers pattern)
  return const_cast<IndexedType*>(
      static_cast<const TypeIndexedStore*>(this)->getPtr<IndexedType>());
}

}  // namespace nvblox
