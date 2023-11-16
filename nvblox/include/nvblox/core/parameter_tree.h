/*
Copyright 2023 NVIDIA CORPORATION

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

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <glog/logging.h>

namespace nvblox {
namespace parameters {

// Making a templated version of std::to_string for the default constructor
// argument below.
template <typename ParameterType>
inline std::string to_string(const ParameterType& val) {
  return std::to_string(val);
}

template <>
inline std::string to_string(const std::string& val) {
  return val;
}
// A class representing a node in a parameter tree. A node is either:
// - A leaf, which is a single parameter, or
// - Not a leaf, which represents a component of nvblox containing parameters.
// NOTE(alexmillane): At the moment we're only using the parameter tree for
// printing parameters. Therefore we only store the value *string*. In the
// future we plan to add functionality for saving and loading the parameter
// string from file.
class ParameterTreeNode {
 public:
  // Initialize a leaf node (a parameter)
  template <typename ParameterType>
  explicit ParameterTreeNode(const std::string& name,
                             const ParameterType& value,
                             std::function<std::string(const ParameterType&)>
                                 _to_string = to_string<ParameterType>)
      : name_(name), value_string_(_to_string(value)) {
    CHECK(!name.empty());
  }

  // Initialize a component.
  explicit ParameterTreeNode(const std::string& name,
                             const std::vector<ParameterTreeNode>& children);

  /// Returns the name of this node in the parameter tree.
  /// @return the name
  const std::string& name() const;

  /// Returns the (optional) value of this node in the parameter tree.
  /// Note that non-leaf nodes, do not have a value
  /// @return The (optional) value string
  const std::optional<std::string>& value_string() const;

  /// Returns the (optional) children of this node in the parameter tree.
  /// Note that leaf nodes do not have children.
  /// @return The (optional) children
  const std::optional<std::vector<ParameterTreeNode>>& children() const;

  /// Returns true if this node is a leaf.
  /// @return true if this node is a leaf
  bool isLeaf() const;

 private:
  // The name of the parameter or component-containing-parameters
  std::string name_;

  // The value of the parameter represented as a string.
  std::optional<std::string> value_string_;

  // Sub-parameters/components
  std::optional<std::vector<ParameterTreeNode>> children_;
};

std::string parameterTreeToString(const ParameterTreeNode& root);

}  // namespace parameters
}  // namespace nvblox
