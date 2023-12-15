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

#include "nvblox/core/parameter_tree.h"

#include <algorithm>
#include <sstream>

#include <glog/logging.h>

namespace nvblox {
namespace parameters {

namespace {

std::string getIndentString(const int indent_level) {
  std::stringstream ss;
  constexpr int kIndentSize = 4;
  for (int i = 0; i < kIndentSize * indent_level; i++) {
    ss << " ";
  }
  return ss.str();
}

void printNodeAndChildren(const ParameterTreeNode& node, const int indent_level,
                          std::stringstream* ss) {
  // Recurse into the children
  if (node.isLeaf()) {
    *ss << getIndentString(indent_level) << node.name() << ": "
        << *node.value_string() << "\n";
  } else {
    *ss << getIndentString(indent_level) << node.name() << "\n";
    const std::vector<ParameterTreeNode> children = *node.children();
    std::for_each(children.begin(), children.end(),
                  [&](const ParameterTreeNode& child) {
                    printNodeAndChildren(child, indent_level + 1, ss);
                  });
  }
}

}  // namespace

ParameterTreeNode::ParameterTreeNode(
    const std::string& name, const std::vector<ParameterTreeNode>& children)
    : name_(name), children_(children) {
  CHECK(!name.empty());
  CHECK_GT(children.size(), 0);
}

const std::string& ParameterTreeNode::name() const { return name_; }

const std::optional<std::string>& ParameterTreeNode::value_string() const {
  return value_string_;
}

const std::optional<std::vector<ParameterTreeNode>>&
ParameterTreeNode::children() const {
  return children_;
}

bool ParameterTreeNode::isLeaf() const { return !children_; }

std::string parameterTreeToString(const ParameterTreeNode& root) {
  std::stringstream ss;
  constexpr int kStartIndentLevel = 0;
  printNodeAndChildren(root, kStartIndentLevel, &ss);
  return ss.str();
}

}  // namespace parameters
}  // namespace nvblox
