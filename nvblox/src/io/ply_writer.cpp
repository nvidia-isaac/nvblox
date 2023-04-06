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
#include "nvblox/utils/logging.h"

#include "nvblox/io/ply_writer.h"

namespace nvblox {
namespace io {

bool PlyWriter::write() {
  if (!file_) {
    // Output a warning -- couldn't open file?
    LOG(WARNING) << "Could not open file for PLY output.";
    return false;
  }
  if (points_ == nullptr || points_->size() == 0) {
    LOG(ERROR) << "No points added, nothing to output.";
    return false;
  }

  // Double-check that the rest of the sizes match.
  if (intensities_ != nullptr && points_->size() != intensities_->size()) {
    LOG(ERROR) << "Intensity size does not match point size.";
    return false;
  }

  if (colors_ != nullptr && points_->size() != colors_->size()) {
    LOG(ERROR) << "Color size does not match point size.";
    return false;
  }
  if (normals_ != nullptr && points_->size() != normals_->size()) {
    LOG(ERROR) << "Normal size does not match point size.";
    return false;
  }
  // Triangles can be whatever size they like.

  // Output the header first.
  writeHeader();

  // Then output all the points.
  writePoints();

  // Then triangles.
  writeTriangles();

  return true;
}

void PlyWriter::writeHeader() {
  file_ << "ply" << std::endl;
  file_ << "format ascii 1.0" << std::endl;
  file_ << "element vertex " << points_->size() << std::endl;
  file_ << "property float x" << std::endl;
  file_ << "property float y" << std::endl;
  file_ << "property float z" << std::endl;

  if (normals_) {
    // TODO: should this be normal_x or nx?
    file_ << "property float nx" << std::endl;
    file_ << "property float ny" << std::endl;
    file_ << "property float nz" << std::endl;
  }

  if (intensities_) {
    file_ << "property float intensity" << std::endl;
  }

  if (colors_) {
    file_ << "property uchar red" << std::endl;
    file_ << "property uchar green" << std::endl;
    file_ << "property uchar blue" << std::endl;
  }

  if (triangles_) {
    // TODO: check if "vertex_index" or "vertex_indices"
    file_ << "element face " << triangles_->size() / 3 << std::endl;
    file_ << "property list uchar int vertex_indices"
          << std::endl;  // pcl-1.7(ros::kinetic) breaks ply convention by not
                         // using "vertex_index"
  }
  file_ << "end_header" << std::endl;
}

void PlyWriter::writePoints() {
  // We've already checked that all the ones that exist have a matching size.
  for (size_t i = 0; i < points_->size(); i++) {
    const Vector3f vertex = (*points_)[i];
    file_ << vertex.x() << " " << vertex.y() << " " << vertex.z();

    if (normals_) {
      const Vector3f& normal = (*normals_)[i];
      file_ << " " << normal.x() << " " << normal.y() << " " << normal.z();
    }

    if (intensities_) {
      file_ << " " << (*intensities_)[i];
    }

    if (colors_) {
      const Color& color = (*colors_)[i];
      file_ << " " << std::to_string(color.r) << " " << std::to_string(color.g)
            << " " << std::to_string(color.b);
    }
    file_ << std::endl;
  }
}

void PlyWriter::writeTriangles() {
  constexpr int kTriangleSize = 3;
  if (triangles_) {
    for (size_t i = 0; i < triangles_->size(); i += kTriangleSize) {
      file_ << kTriangleSize << " ";

      for (int j = 0; j < kTriangleSize; j++) {
        file_ << (*triangles_)[i + j] << " ";
      }

      file_ << std::endl;
    }
  }
}

}  // namespace io
}  // namespace nvblox