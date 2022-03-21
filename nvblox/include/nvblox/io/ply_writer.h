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

#include <fstream>
#include <string>

#include "nvblox/core/color.h"
#include "nvblox/core/types.h"

namespace nvblox {
namespace io {
/**
 * Writes a mesh to a .ply file. For reference on the format, see:
 *  http://paulbourke.net/dataformats/ply/
 */
class PlyWriter {
 public:
  explicit PlyWriter(const std::string& filename) : file_(filename) {}

  ~PlyWriter() { file_.close(); }

  // Add however many of these you want.
  void setPoints(const std::vector<Vector3f>* points) { points_ = points; }
  void setColors(const std::vector<Color>* colors) { colors_ = colors; }
  void setIntensities(const std::vector<float>* intensities) {
    intensities_ = intensities;
  }
  void setNormals(const std::vector<Vector3f>* normals) { normals_ = normals; }
  void setTriangles(const std::vector<int>* triangles) {
    triangles_ = triangles;
  }

  // Call this after points, normals, triangles, etc. have been added to write
  // to file.
  bool write();

 private:
  void writeHeader();
  void writePoints();
  void writeTriangles();

  const std::vector<Vector3f>* points_ = nullptr;
  const std::vector<Vector3f>* normals_ = nullptr;
  const std::vector<float>* intensities_ = nullptr;
  const std::vector<Color>* colors_ = nullptr;
  const std::vector<int>* triangles_ = nullptr;

  std::ofstream file_;
};

}  // namespace io

}  // namespace nvblox
