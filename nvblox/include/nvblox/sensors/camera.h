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

#include "nvblox/core/types.h"
#include "nvblox/interpolation/interpolation_2d.h"
#include "nvblox/sensors/distortion.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/sensor.h"

namespace nvblox {

class Frustum;
class BoundingPlane;

/// Class representing the image rectangle
using CameraViewport = Eigen::AlignedBox<float, 2>;

/// Class that describes the parameters and FoV of a camera.
class Camera : public SensorBase {
 public:
  __host__ __device__ inline Camera() = default;
  __host__ __device__ inline ~Camera() = default;

  /// Constructor
  /// @param fu Focal length (in pixels) in the u (x/width) direction.
  /// @param fv Focal length (in pixels) in the v (y/height) direction.
  /// @param cu Principal point position in the u (x/width) direction.
  /// @param cv Principal point position in the v (y/height) direction.
  /// @param width Width (in pixels) of the image plane.
  /// @param height Height (in pixels) of the image plane.
  /// @param distortion_params Distortion parameters.
  __host__ __device__ inline Camera(
      float fu, float fv, float cu, float cv, int width, int height,
      std::optional<RadialTangentialDistortionParams> distortion_params =
          std::nullopt);

  /// Project a 3D point in camera image space to a 2D pixel coordinate.
  /// @param p_C Input 3D point coordinate in image space.
  /// @param u_C Output 2D pixel coordinate.
  /// @param min_depth Min allowed depth
  /// @param check_viewport Whether viewport check should be performed
  /// @return True if the projected depth is larger than min_depth AND the point
  /// is inside the camera viewport
  __host__ __device__ inline bool project(
      const Vector3f& p_C, Vector2f* u_C,
      const float min_depth = kDefaultMinProjectionDepth,
      const bool check_viewport = true) const;

  /// Project p_C into a normalized camera (camera that has identity
  /// calibration matrix).
  /// @param p_C Input 3D point coordinate in image space.
  /// @param u_C Output 2D normalized image coordinate.
  /// @return Whether the pixel is on the image.
  inline static __host__ __device__ bool projectToNormalizedCoordinates(
      const Vector3f& p_C, Vector2f* u_C,
      const float min_depth = kDefaultMinProjectionDepth);

  /// Get the depth of a 3D point (just the z component).
  /// @param p_C The point in the coordinate frame of the camera.
  /// @return The depth.
  __host__ __device__ inline float getDepth(const Vector3f& p_C) const;

  /// Calculates a 3D point from a (floating point) pixel coordinate and a
  /// depth.
  /// @param u_C The point on the image plane.
  /// @param depth The depth of the returned point.
  /// @return The 3D point in the camera frame.
  __host__ __device__ inline Vector3f unprojectFromImagePlaneCoordinates(
      const Vector2f& u_C, const float depth) const;

  /// Calculates a 3D point from a (integer) pixel coordinate and a depth.
  /// @param u_C The point on the image plane.
  /// @param depth The depth of the returned point.
  /// @return The 3D point in the camera frame.
  __host__ __device__ inline Vector3f unprojectFromPixelIndices(
      const Index2D& u_C, const float depth) const;

  /// Get the axis aligned bounding box of the view in the Layer coordinate
  /// frame.
  /// @param T_L_C The pose of the camera in the layer frame.
  /// @param min_depth The minimum depth of view frustum.
  /// @param max_depth The maximum depth of view frustum.
  /// @return The AxisAlignedBoundingBox.
  __host__ AxisAlignedBoundingBox getViewAABB(const Transform& T_L_C,
                                              const float min_depth,
                                              const float max_depth) const;

  /// Get a frustum in the Layer frame representing the camera's view.
  /// @param T_L_C The pose of the camera in the layer frame.
  /// @param min_depth The minimum depth of view frustum.
  /// @param max_depth The maximum depth of view frustum.
  /// @return The Frustum.
  __host__ Frustum getViewFrustum(const Transform& T_L_C, const float min_depth,
                                  const float max_depth) const;

  /// Get the corners of the view frustum in the camera coordinate frame.
  /// @param min_depth The minimum depth of view frustum.
  /// @param max_depth The maximum depth of view frustum.
  /// @return An 8 by 3 matrix describing the 3D location of the 8 corners.
  __host__ Eigen::Matrix<float, 8, 3> getViewCorners(
      const float min_depth, const float max_depth) const;

  /// Get the camera viewport in normalized image coordinates with an
  /// appended margin in pixels.
  ///
  /// A normalized camera has K = I and thus its image coordinates
  /// reside in a plane situated one length-unit in front of the
  /// camera. For more info, see:
  /// https://en.wikipedia.org/wiki/Camera_matrix#Normalized_camera_matrix_and_normalized_image_coordinates
  /// @param margin_pixels The floating point number of pixels added to the
  /// image height and width.
  /// @return The CameraViewport.
  __host__ CameraViewport
  getNormalizedViewport(const float margin_pixels = 0.F) const;

  /// Returns an unnormalized ray direction in the camera frame corresponding
  /// to the passed pixel (in floating point pixels).
  /// @param u_C The pixel coordinates.
  /// @return The vector pointing from the projective center through the pixel.
  __host__ __device__ inline Vector3f vectorFromImagePlaneCoordinates(
      const Vector2f& u_C) const;

  /// Returns an unnormalized ray direction in the camera frame corresponding
  /// to the passed pixel (in integer point pixels).
  /// @param u_C The pixel coordinates.
  /// @return The vector pointing from the projective center through the pixel.
  __host__ __device__ inline Vector3f vectorFromPixelIndices(
      const Index2D& u_C) const;

  /// Define how this sensor interpolates on a depth image.
  __host__ __device__ inline bool static interpolateDepthImage(
      const DepthImageConstView depth_image, const Vector2f& u_px,
      const Vector3f& p_voxel_center_C_unused, const float voxel_size_unused,
      float* value_interpolated_ptr, Index2D* u_px_closest_ptr = nullptr);

  /// Check if camera has distortion parameters
  /// @return True if camera was created with distortion parameters.
  __host__ __device__ inline bool hasDistortion() const;

  /// Accessors
  __host__ __device__ inline float fu() const { return fu_; }
  __host__ __device__ inline float fv() const { return fv_; }
  __host__ __device__ inline float cu() const { return cu_; }
  __host__ __device__ inline float cv() const { return cv_; }
  __host__ __device__ inline int width() const { return width_; }
  __host__ __device__ inline int height() const { return height_; }
  __host__ __device__ inline int cols() const { return width_; }
  __host__ __device__ inline int rows() const { return height_; }

  /// Get the distortion parameters
  /// @return The distortion parameters or std::nullopt if no distortion
  /// parameters are set.
  __host__
      __device__ inline const std::optional<RadialTangentialDistortionParams>&
      distortion_params() const {
    return distortion_params_;
  }

  /// Get the sensor modality identifier
  /// @return The sensor modality (kCamera).
  __host__ __device__ static constexpr SensorModality sensor_modality() {
    return SensorModality::kCamera;
  }

  /// Camera factory.
  /// @param mat Matrix representation of the camera intrinsics.
  /// @param width The width (in pixels) of the image plane.
  /// @param height The height (in pixels) of the image plane.
  /// @param distortion_params Distortion parameters.
  /// @return A Camera object representation of the intrinsics.
  inline static Camera fromIntrinsicsMatrix(
      const Matrix3f& mat, int width, int height,
      std::optional<RadialTangentialDistortionParams> distortion_params =
          std::nullopt);

  /// Equality
  __host__ inline friend bool operator==(const Camera& lhs, const Camera& rhs);

 private:
  float fu_ = 0.F;
  float fv_ = 0.F;
  float cu_ = 0.F;
  float cv_ = 0.F;

  int width_ = 0;
  int height_ = 0;

  std::optional<RadialTangentialDistortionParams> distortion_params_;
};

/// Equality
__host__ inline bool operator==(const Camera& lhs, const Camera& rhs);

// Stream Camera as text
std::ostream& operator<<(std::ostream& os, const Camera& camera);

/// A bounding plane which has one "inside" direction and the other direction is
/// "outside." Allows to query for which side of the plane you are on.
class BoundingPlane {
 public:
  /// Default constructor.
  BoundingPlane() : normal_(Vector3f::Identity()), distance_(0.0f) {}

  /// Set the bounding plane from three points.
  /// @param p1 The first point.
  /// @param p2 The second point.
  /// @param p3 The third point.
  void setFromPoints(const Vector3f& p1, const Vector3f& p2,
                     const Vector3f& p3);

  /// Set the bounding plane from its normal and distance from the origin.
  /// @param normal The normal vector of the plane.
  /// @param distance The distance from the origin along the normal vector.
  void setFromDistanceNormal(const Vector3f& normal, float distance);

  /// Check if a point is on the same side of the plane normal.
  /// @param point The point to check.
  /// @return True if the point is inside the bounding plane, false otherwise.
  bool isPointInside(const Vector3f& point) const;

  /// Get the normal vector of the bounding plane.
  /// @return The normal vector of the bounding plane.
  const Vector3f& normal() const { return normal_; }

  /// Get the distance of the bounding plane from the origin along its normal
  /// vector.
  /// @return The distance of the bounding plane from the origin.
  float distance() const { return distance_; }

 private:
  /// The normal vector of the bounding plane.
  Vector3f normal_;

  /// The distance from the origin along the normal vector.
  float distance_;
};

/// Class that allows checking for whether objects are within the field of
/// view of a camera or not.
class Frustum {
 public:
  /// Constructor to initialize the frustum with camera parameters and depth
  /// range.
  /// @param camera The camera used to define the frustum.
  /// @param T_L_C The transformation from the camera to the Layer frame.
  /// @param min_depth The minimum depth of the frustum.
  /// @param max_depth The maximum depth of the frustum.
  Frustum(const Camera& camera, const Transform& T_L_C, float min_depth,
          float max_depth);

  /// Get the axis-aligned bounding box (AABB) of the frustum.
  /// @return The axis-aligned bounding box of the frustum.
  AxisAlignedBoundingBox getAABB() const { return aabb_; }

  /// Check if a point is inside the frustum's view.
  /// @param point The point to check.
  /// @return True if the point is inside the frustum's view, false otherwise.
  bool isPointInView(const Vector3f& point) const;

  /// Check if an axis-aligned bounding box (AABB) is touched by the frustum.
  /// @param aabb The AABB to check.
  /// @return True if the AABB is touched by the frustum, false otherwise.
  bool isAABBInView(const AxisAlignedBoundingBox& aabb) const;

 private:
  /// Compute the bounding planes of the frustum.
  /// @param corners_C The corners of the frustum in camera coordinates.
  /// @param T_L_C Transformation from the camera to the Layer coordinate
  /// frame.
  void computeBoundingPlanes(const Eigen::Matrix<float, 8, 3>& corners_C,
                             const Transform& T_L_C);

  /// Bounding planes containing around the frustum. Expressed in the layer
  /// coordinate frame.
  std::array<BoundingPlane, 6> bounding_planes_L_;

  /// Cached AABB of the
  AxisAlignedBoundingBox aabb_;
};

}  // namespace nvblox

#include "nvblox/sensors/internal/impl/camera_impl.h"
