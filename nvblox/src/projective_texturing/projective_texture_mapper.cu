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

#include "nvblox/projective_texturing/projective_texture_mapper.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include "nvblox/core/internal/error_check.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

namespace {

/// Project a single vertex to atlas-space UV.
/// Steps:
///   1. Transform vertex from world frame to camera frame (T_C_W)
///   2. Project 3D point to 2D pixel using camera intrinsics
///   3. Check occlusion against depth image (if provided)
///   4. Normalize pixel coords to [0,1] and remap to atlas-space UVs
/// @return false if vertex is outside camera frustum or occluded.
__device__ bool projectVertex(const Vector3f& vertex, const CameraView& view,
                              float occlusion_tolerance_m, Vector2f* uv_out) {
  // Step 1: World to camera frame
  Vector3f p_C = view.T_C_W * vertex;

  // Step 2: Project to pixel coordinates using camera intrinsics
  Vector2f u_px;
  if (!view.camera.project(p_C, &u_px)) {
    return false;  // Outside camera frustum or behind camera
  }

  // Step 3: Occlusion check -- compare vertex depth against the depth image.
  // If the vertex is further from the camera than the observed surface
  // (plus tolerance), it is occluded by a foreground object.
  int px_x = static_cast<int>(u_px.x() + 0.5f);
  int px_y = static_cast<int>(u_px.y() + 0.5f);
  if (px_x >= 0 && px_x < view.depth_image.width() && px_y >= 0 &&
      px_y < view.depth_image.height()) {
    float observed_depth = view.depth_image(px_y, px_x);
    if (observed_depth > 0.0f &&
        p_C.z() > observed_depth + occlusion_tolerance_m) {
      return false;  // Occluded by foreground surface
    }
  }

  // Step 4: Normalize pixel coords to [0,1] image-space UVs, then remap
  // to atlas-space. For single camera, offset=(0,0) and scale=(1,1).
  float u = u_px.x() / static_cast<float>(view.color_image.width());
  float v = u_px.y() / static_cast<float>(view.color_image.height());

  *uv_out = Vector2f(view.atlas_uv_offset.x() + u * view.atlas_uv_scale.x(),
                     view.atlas_uv_offset.y() + v * view.atlas_uv_scale.y());
  return true;
}

/// Try to project all 3 triangle vertices into the given camera.
/// @return true if all 3 vertices were successfully projected.
__device__ bool tryProjectTriangle(const Vector3f* vertices, int base,
                                   const CameraView& view,
                                   float occlusion_tolerance_m, Vector2f* uvs) {
  for (int v = 0; v < 3; ++v) {
    if (!projectVertex(vertices[base + v], view, occlusion_tolerance_m,
                       &uvs[v])) {
      return false;
    }
  }
  return true;
}

/// Projective texture mapping kernel: one thread per triangle.
/// For each triangle:
///   1. Score each camera by how front-facing the triangle is (face normal
///      dot viewing direction). Cameras are sorted by score descending.
///   2. Try cameras in order: project all 3 vertices using projectVertex().
///   3. All-or-nothing: if ANY vertex fails, fall back to the next-best
///      camera. If all cameras fail, the triangle gets kInvalidUV.
///
/// Assumes triangle-list layout: triangle i uses vertices [i*3, i*3+1, i*3+2].
/// @note num_cameras is expected to be small (typically 1-8). Per-thread stack
/// arrays are allocated proportional to num_cameras.
__global__ void projectTrianglesKernel(const Vector3f* vertices,
                                       int num_triangles,
                                       const CameraView* views, int num_cameras,
                                       float occlusion_tolerance_m,
                                       Vector2f* vertex_uvs) {
  int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tri_idx >= num_triangles) return;

  const Vector2f kInvalidUV(-1.0f, -1.0f);
  int base = tri_idx * 3;

  // Score each camera by how front-facing the triangle is to that camera.
  // The camera with the highest dot(face_normal, viewing_direction) wins --
  // meaning the triangle most directly faces that camera, giving the least
  // texture distortion.
  //
  // Face normal is computed from triangle edges rather than input vertex
  // normals because: (1) vertex normals may not be provided for all mesh
  // types, and (2) face normal gives the true triangle orientation, while
  // vertex normals may be smoothed across faces.
  //
  // Normalization of face_normal is skipped because it is constant across
  // all cameras for a given triangle, so it does not affect relative ranking.
  Vector3f center =
      (vertices[base] + vertices[base + 1] + vertices[base + 2]) / 3.0f;
  Vector3f edge1 = vertices[base + 1] - vertices[base];
  Vector3f edge2 = vertices[base + 2] - vertices[base];
  Vector3f face_normal = edge1.cross(edge2);

  // Collect per-camera scores and indices, then sort by score descending
  // so we can try the best camera first and fall back to the next-best.
  float scores[kMaxProjectiveTextureCameras];
  int indices[kMaxProjectiveTextureCameras];
  int valid_count = 0;

  for (int cam = 0; cam < num_cameras; ++cam) {
    // Check if the triangle center is visible in this camera
    Vector3f p_C = views[cam].T_C_W * center;
    Vector2f u_px;
    if (!views[cam].camera.project(p_C, &u_px)) continue;

    // Score: dot product of face normal with camera-to-triangle direction.
    // Higher score = triangle faces the camera more directly.
    // camera_position_W is precomputed on host to avoid T_C_W.inverse() here.
    Vector3f viewing_dir = (views[cam].camera_position_W - center).normalized();
    float score = face_normal.dot(viewing_dir);

    // Insertion sort (descending) -- small array, simple and efficient.
    int insert_pos = valid_count;
    while (insert_pos > 0 && scores[insert_pos - 1] < score) {
      scores[insert_pos] = scores[insert_pos - 1];
      indices[insert_pos] = indices[insert_pos - 1];
      --insert_pos;
    }
    scores[insert_pos] = score;
    indices[insert_pos] = cam;
    ++valid_count;
  }

  // Try cameras in order of decreasing score. If the best camera can't
  // project all 3 vertices, fall back to the next-best, and so on.
  for (int i = 0; i < valid_count; ++i) {
    Vector2f uvs[3];
    if (tryProjectTriangle(vertices, base, views[indices[i]],
                           occlusion_tolerance_m, uvs)) {
      vertex_uvs[base + 0] = uvs[0];
      vertex_uvs[base + 1] = uvs[1];
      vertex_uvs[base + 2] = uvs[2];
      return;
    }
  }

  // No camera could project all 3 vertices
  vertex_uvs[base + 0] = kInvalidUV;
  vertex_uvs[base + 1] = kInvalidUV;
  vertex_uvs[base + 2] = kInvalidUV;
}

}  // namespace

void ProjectiveTextureMapper::buildAtlasAsync(std::vector<CameraView> views,
                                              const CudaStream& stream) {
  timing::Timer timer("projective_texture_mapper/set_views");
  views_ = std::move(views);
  atlas_.buildAtlasAsync(views_, stream);

  // Populate atlas UV fields on each view after atlas is built
  for (int i = 0; i < static_cast<int>(views_.size()); ++i) {
    views_[i].atlas_uv_offset = atlas_.uvOffset(i);
    views_[i].atlas_uv_scale = atlas_.uvScale(i);
  }
}

void ProjectiveTextureMapper::mapMesh(ColorMesh* mesh,
                                      const CudaStream& stream) {
  CHECK_NOTNULL(mesh);

  constexpr size_t kVerticesPerTriangle = 3;
  const size_t num_vertices = mesh->vertices.size();

  if (views_.empty()) {
    LOG(WARNING) << "ProjectiveTextureMapper::mapMesh: no views set";
    return;
  }

  if (num_vertices == 0) {
    LOG(WARNING) << "ProjectiveTextureMapper::mapMesh: no vertices";
    return;
  }

  CHECK_EQ(num_vertices % kVerticesPerTriangle, size_t{0})
      << "Mesh vertices must be in triangle-list layout (multiple of 3)";
  const size_t num_triangles = num_vertices / kVerticesPerTriangle;

  timing::Timer timer("projective_texture_mapper/map_mesh");

  CHECK_LE(static_cast<int>(views_.size()), kMaxProjectiveTextureCameras)
      << "Number of cameras exceeds kernel stack array limit ("
      << kMaxProjectiveTextureCameras << ")";

  mesh->vertex_uvs.resizeAsync(num_vertices, stream);

  // Copy view data to persistent device buffer (resizes as needed)
  d_views_.copyFromAsync(views_, stream);

  // Launch kernel: one thread per triangle
  constexpr int kBlockSize = 256;
  const int num_blocks =
      static_cast<int>((num_triangles + kBlockSize - 1) / kBlockSize);
  projectTrianglesKernel<<<num_blocks, kBlockSize, 0, stream>>>(
      mesh->vertices.data(), static_cast<int>(num_triangles), d_views_.data(),
      static_cast<int>(views_.size()), params_.occlusion_tolerance_m,
      mesh->vertex_uvs.data());

  checkCudaErrors(cudaGetLastError());
}

}  // namespace nvblox
