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
#include "nvblox/io/mesh_io.h"

#include <cmath>
#include <vector>

#include "nvblox/geometry/plane.h"
#include "nvblox/geometry/transforms.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/mesh/mesh_transform.h"

namespace nvblox {
namespace io {

namespace {

bool writeMeshToPly(const std::string& filename,
                    std::vector<Vector3f>* vertices,
                    std::vector<int>* triangles, std::vector<Vector3f>* normals,
                    std::vector<Color>* colors) {
  io::PlyWriter writer(filename);
  writer.setPoints(vertices);
  writer.setTriangles(triangles);
  if (normals->size() > 0) {
    writer.setNormals(normals);
  }
  if (colors->size() > 0) {
    writer.setColors(colors);
  }
  return writer.write();
}

}  // namespace

bool outputColorMeshLayerToPly(const ColorMeshLayer& layer,
                               const std::string& filename) {
  // Create a CUDA stream for the mesh operations
  auto cuda_stream = CudaStreamOwning();

  // NOTE: Intensity mesh output is not supported yet.
  const std::shared_ptr<const ColorMesh> mesh = layer.getMesh(cuda_stream);

  // Convert unified vectors to std vectors for PLY writer
  std::vector<Vector3f> vertices = mesh->vertices.toVectorAsync(cuda_stream);
  std::vector<Vector3f> normals =
      mesh->vertex_normals.toVectorAsync(cuda_stream);
  std::vector<Color> colors =
      mesh->vertex_appearances.toVectorAsync(cuda_stream);
  std::vector<int> triangles = mesh->triangles.toVectorAsync(cuda_stream);

  cuda_stream.synchronize();

  return writeMeshToPly(filename, &vertices, &triangles, &normals, &colors);
}

bool outputColorMeshLayerToPly(const ColorMeshLayer& layer,
                               const char* filename) {
  return outputColorMeshLayerToPly(layer, std::string(filename));
}

bool outputColorMeshLayerToPly(const ColorMeshLayer& layer,
                               const std::string& filename,
                               const Plane& ground_plane) {
  // Create a CUDA stream for the mesh operations
  auto cuda_stream = CudaStreamOwning();

  // TODO: doesn't support intensity yet!!!!
  const std::shared_ptr<const ColorMesh> mesh = layer.getMesh(cuda_stream);

  // Create mutable copies of vertices and normals for GPU transformation
  unified_vector<Vector3f> transformed_vertices(MemoryType::kDevice);
  unified_vector<Vector3f> transformed_normals(MemoryType::kDevice);

  // Compute transform to align ground plane to z=0
  Transform T_plane_to_z0 = computeTransformToAlignPlaneToZ0(ground_plane);

  // Copy mesh data to mutable unified vectors for GPU transformation
  transformed_vertices.copyFromAsync(mesh->vertices, cuda_stream);
  if (!mesh->vertex_normals.empty()) {
    transformed_normals.copyFromAsync(mesh->vertex_normals, cuda_stream);
  }
  cuda_stream.synchronize();

  // Transform vertices and normals on GPU
  transformMeshOnGPU(T_plane_to_z0, &transformed_vertices, &transformed_normals,
                     &cuda_stream);
  cuda_stream.synchronize();

  LOG(INFO) << "Applied ground plane transform to mesh on GPU. "
            << "Plane normal: (" << ground_plane.normal().x() << ", "
            << ground_plane.normal().y() << ", " << ground_plane.normal().z()
            << "), offset: " << ground_plane.offset();

  // Convert unified vectors to std vectors for PLY writer
  std::vector<Vector3f> vertices =
      transformed_vertices.toVectorAsync(cuda_stream);
  std::vector<Vector3f> normals;
  if (!transformed_normals.empty()) {
    normals = transformed_normals.toVectorAsync(cuda_stream);
  }
  std::vector<Color> colors =
      mesh->vertex_appearances.toVectorAsync(cuda_stream);
  std::vector<int> triangles = mesh->triangles.toVectorAsync(cuda_stream);

  cuda_stream.synchronize();

  return writeMeshToPly(filename, &vertices, &triangles, &normals, &colors);
}

}  // namespace io
}  // namespace nvblox
