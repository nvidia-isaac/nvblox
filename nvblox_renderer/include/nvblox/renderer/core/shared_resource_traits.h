/*
Copyright 2026 NVIDIA CORPORATION

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

namespace nvblox {
namespace renderer {

// Forward declarations
class SharedBuffer;
class SharedTexture;
template <typename Derived>
class SharedResourceBase;

namespace traits {

/// Type trait to check if a type is a shared resource.
/// Use with static_assert for compile-time validation.
///
/// Example:
/// @code
/// template <typename T>
/// void processResource(T& resource) {
///   static_assert(traits::is_shared_resource<T>::value,
///                 "T must be a SharedBuffer or SharedTexture");
///   // ...
/// }
/// @endcode
template <typename T>
struct is_shared_resource : std::false_type {};

/// Specialization for SharedBuffer.
template <>
struct is_shared_resource<SharedBuffer> : std::true_type {};

/// Specialization for SharedTexture.
template <>
struct is_shared_resource<SharedTexture> : std::true_type {};

/// Helper variable template (C++14 style).
template <typename T>
constexpr bool is_shared_resource_v = is_shared_resource<T>::value;

/// Type trait to check if a type derives from SharedResourceBase.
/// This is useful for checking custom derived types.
template <typename T, typename = void>
struct is_shared_resource_derived : std::false_type {};

/// Specialization that checks for inheritance via DerivedType typedef.
template <typename T>
struct is_shared_resource_derived<T, std::void_t<typename T::DerivedType>>
    : std::true_type {};

/// Helper variable template.
template <typename T>
constexpr bool is_shared_resource_derived_v =
    is_shared_resource_derived<T>::value;

/// Type trait to check if a type has a buffer-like interface (has cudaPtr()).
template <typename T, typename = void>
struct has_cuda_ptr : std::false_type {};

template <typename T>
struct has_cuda_ptr<T, std::void_t<decltype(std::declval<T>().cudaPtr())>>
    : std::true_type {};

/// Helper variable template.
template <typename T>
constexpr bool has_cuda_ptr_v = has_cuda_ptr<T>::value;

/// Type trait to check if a type has a texture-like interface (has
/// cudaArray()).
template <typename T, typename = void>
struct has_cuda_array : std::false_type {};

template <typename T>
struct has_cuda_array<T, std::void_t<decltype(std::declval<T>().cudaArray())>>
    : std::true_type {};

/// Helper variable template.
template <typename T>
constexpr bool has_cuda_array_v = has_cuda_array<T>::value;

}  // namespace traits
}  // namespace renderer
}  // namespace nvblox
