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

namespace nvblox {

// Unique type
template <class T1, class T2, class... Trest>
struct unique_types<T1, T2, Trest...> {
  static constexpr bool value = unique_types<T1, T2>::value &&
                                unique_types<T1, Trest...>::value &&
                                unique_types<T2, Trest...>::value;
};

template <class T1, class T2>
struct unique_types<T1, T2> {
  static constexpr bool value = !std::is_same<T1, T2>::value;
};

template <class T1>
struct unique_types<T1> {
  static constexpr bool value = true;
};

template <>
struct unique_types<> {
  static constexpr bool value = true;
};

}  // namespace nvblox
