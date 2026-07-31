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

#include <filesystem>
#include <fstream>
#include <iomanip>

namespace nvblox {
namespace io {

std::string getIndexedPath(const std::string& dir, const std::string& prefix,
                           const int idx, const int num_digits,
                           const std::string& ext) {
  // Build the path
  std::ostringstream oss;
  oss << prefix << std::setw(num_digits) << std::setfill('0') << idx << ext;
  const std::string filename = oss.str();
  return std::filesystem::path(dir) / filename;
}

template <typename Derived>
void writeToCsv(const std::string& filepath,
                const Eigen::DenseBase<Derived>& eig) {
  std::ofstream file_stream(filepath, std::ofstream::out);
  const Eigen::IOFormat csv_format(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                   ", ", "\n");
  file_stream << eig.format(csv_format);
}

}  // namespace io
}  // namespace nvblox
