/*
Copyright 2025 NVIDIA CORPORATION

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
#include <Eigen/Dense>
#include <cmath>

namespace nvblox {

// Compute correction factor for compensating for distortion arising from the
// spherical shape of lenses. See
// https://en.wikipedia.org/wiki/Distortion_(optics)
template <typename T>
__host__ __device__ T radialDistortionScale(const T r2,
                                            const RadialDistortionParams& rp) {
  const T r4 = r2 * r2;
  const T r6 = r2 * r4;
  const T numerator = 1.0 + rp.k1 * r2 + rp.k2 * r4 + rp.k3 * r6;
  const T denominator = 1.0 + rp.k4 * r2 + rp.k5 * r4 + rp.k6 * r6;
  return numerator / denominator;
}

// Correct a point for tangential distortion, arising when imaging elements in
// the camera are off-axis from each other. See
// https://en.wikipedia.org/wiki/Distortion_(optics)
template <typename T>
__host__ __device__ Eigen::Vector2<T> applyTangentialDistortion(
    const Eigen::Vector2<T>& u, const T r2,
    const TangentialDistortionParams& tp) {
  const T xy = u.x() * u.y();
  const T x_tangential = 2.0 * tp.p1 * xy + tp.p2 * (r2 + 2.0 * u.x() * u.x());
  const T y_tangential = 2.0 * tp.p2 * xy + tp.p1 * (r2 + 2.0 * u.y() * u.y());

  return Eigen::Vector2<T>(x_tangential, y_tangential);
}

Vector2f applyDistortion(
    const Vector2f& u_normalized,
    const RadialTangentialDistortionParams& distortion_params) {
  const float x = u_normalized[0];
  const float y = u_normalized[1];
  const float r2 = x * x + y * y;

  return u_normalized * radialDistortionScale(r2, distortion_params.radial) +
         applyTangentialDistortion(u_normalized, r2,
                                   distortion_params.tangential);
}

__host__ __device__ inline double compute_dR_dr2(
    const double r2, const double k1, const double k2, const double k3,
    const double k4, const double k5, const double k6) {
  // Thanks to sympy for the formula!
  // clang-format off
  //
  // ⎛                    2⎞ ⎛           2       3    ⎞   ⎛                    2⎞ ⎛           2       3    ⎞
  // ⎝k₁ + 2⋅k₂⋅q + 3⋅k₃⋅q ⎠⋅⎝k₄⋅q + k₅⋅q  + k₆⋅q  + 1⎠ - ⎝k₄ + 2⋅k₅⋅q + 3⋅k₆⋅q ⎠⋅⎝k₁⋅q + k₂⋅q  + k₃⋅q  + 1⎠
  // ───────────────────────────────────────────────────────────────────────────────────────────────────────
  //                                                                 2
  //                                       ⎛           2       3    ⎞
  //                                       ⎝k₄⋅q + k₅⋅q  + k₆⋅q  + 1⎠
  //
  // We use notation (a*b - c*d) / b*b  and q == r2.
  //
  // clang-format on

  const double q = r2;
  const double q2 = q * q;
  const double q3 = q2 * q;

  const double k4q = k4 * q;
  const double k5q2 = k5 * q2;
  const double k6q3 = k6 * q3;

  const double a = k1 + 2. * k2 * q + 3. * k3 * q2;
  const double c = k4 + 2. * k5 * q + 3. * k6 * q2;
  const double b = k4q + k5q2 + k6q3 + 1.;
  const double d = k1 * q + k2 * q2 + k3 * q3 + 1.;

  return (a * b - c * d) / (b * b);
}

Vector2f removeDistortion(
    const Vector2f& u_in,
    const RadialTangentialDistortionParams& distortion_params) {
  // We apply an iterative solution (Newton Raphson) to remove distortion.
  // This is since a closed form solution of the inverse distortion function
  // is infeasible.
  //
  // In each iteration, we linearize the distortion function around the
  // current estimate and compute the update step as:
  //
  // delta_xy = -inv(J) * (u_est - u_in)
  //
  // Where:
  //   delta_xy:    Computed update step
  //   u_est:       Current estimate of the undistorted point
  //   u_in:        Input distorted point (in normalized image coordinates)
  //   J:           Jacobian
  //
  // We repeat this process until the update step is small enough. Since the
  // distortion function is well-behaved, this is usually fast.
  //

  constexpr int kMaxIterations = 6;  // Usually converges in 2-3 iterations.

  // Note that we're using double precision to reduce the risk of
  // numerical errors. We therefore convert the distortion coefficients to
  // double. TODO(dtingdahl) investigate impact of using floats.
  const double k1 = distortion_params.radial.k1;
  const double k2 = distortion_params.radial.k2;
  const double k3 = distortion_params.radial.k3;
  const double k4 = distortion_params.radial.k4;
  const double k5 = distortion_params.radial.k5;
  const double k6 = distortion_params.radial.k6;
  const double p1 = distortion_params.tangential.p1;
  const double p2 = distortion_params.tangential.p2;

  // x,y are the current estimates of the undistorted coordinates.
  double x = u_in.x();
  double y = u_in.y();

  for (int i = 0; i < kMaxIterations; i++) {
    // Precompute squared radius.
    const double x2 = x * x;
    const double y2 = y * y;
    const double r2 = x2 + y2;

    // Apply radial and tangential distortion to the current estimate.
    const double R =
        radialDistortionScale<double>(r2, distortion_params.radial);
    const Eigen::Vector2d tangential = applyTangentialDistortion<double>(
        Eigen::Vector2d(x, y), r2, distortion_params.tangential);

    const double x_estimated = x * R + tangential.x();
    const double y_estimated = y * R + tangential.y();

    // Compute residual error.
    const double error_x = x_estimated - u_in.x();
    const double error_y = y_estimated - u_in.y();

    // We precompute derivative of R with respect to x and y
    const double dR_dr2 = compute_dR_dr2(r2, k1, k2, k3, k4, k5, k6);
    const double dR_dx = 2.0 * x * dR_dr2;  // Chain rule
    const double dR_dy = 2.0 * y * dR_dr2;  // Chain rule

    // Then we compute the jacobian.
    const double a = R + x * dR_dx + 2 * p1 * y + 6 * p2 * x;  // derror_x/dx
    const double b = x * dR_dy + 2 * p1 * x + 2 * p2 * y;      // derror_x/dy
    const double c = y * dR_dx + 2 * p2 * y + 2 * p1 * x;      // derror_y/dx
    const double d = R + y * dR_dy + 2 * p2 * x + 6 * p1 * y;  // derror_y/dy

    // Solve for delta step by inverting the 2x2 jacobian.
    const double det = a * d - b * c;
    const double delta_x = (d * error_x - b * error_y) / det;
    const double delta_y = (-c * error_x + a * error_y) / det;

    // Apply step.
    if (std::isfinite(delta_x) && std::isfinite(delta_y)) {
      x = x - delta_x;
      y = y - delta_y;
    }

    // Check termination criteria.
    constexpr double kTerminationCondition = 1e-20;
    if (delta_x * delta_x + delta_y * delta_y < kTerminationCondition) {
      break;
    }
  }
  return Vector2f(static_cast<float>(x), static_cast<float>(y));
}

}  // namespace nvblox
