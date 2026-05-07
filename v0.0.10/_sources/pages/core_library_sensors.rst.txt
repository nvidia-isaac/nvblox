

Visual Sensors
==============

``nvblox`` supports both :ref:`camera` and :ref:`lidar` sensors. It is also possible to extend the library to :ref:`extending-nvblox-to-support-other-sensors`.

.. _camera:

Camera
------

The camera sensor is modeled as a **pinhole camera** operating in the
**computer vision coordinate frame**, defined as:

- **Z-axis**: Forward (optical axis)
- **X-axis**: Right (along the image width)
- **Y-axis**: Down (along the image height)

The corresponding depth images store **Z-values**, i.e. distances along the camera's Z-axis.

A 3D point :math:`P = (X, Y, Z)` in camera coordinates projects to pixel coordinates
:math:`(u, v)` using the pinhole model:

.. math::

   u = f_x \frac{X}{Z} + c_x \\
   v = f_y \frac{Y}{Z} + c_y

where :math:`Z` is the depth value stored in the depth image, and the camera intrinsics are defined in the table below:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Notation
     - Parameter name in :nvblox_code_link:`<nvblox/include/nvblox/sensors/camera.h>`.
     - Description
   * - :math:`f_x`
     - ``fu``
     - Focal length in the u (width) direction [px]
   * - :math:`f_y`
     - ``fv``
     - Focal length in the v (height) direction [px]
   * - :math:`c_x`
     - ``cu``
     - Principal point offset in the u (width) direction [px]
   * - :math:`c_y`
     - ``cv``
     - Principal point offset in the v (height) direction [px]


.. _lidar:

LiDAR
-----

The LiDAR sensor is parameterized using spherical coordinates with **range** :math:`r`,
**azimuth** :math:`\phi` (horizontal angle), and **polar angle** :math:`\alpha` (vertical angle measured from the +Z axis).

The LiDAR sensor operates in the
**standard robotics coordinate frame**, defined as:

- **X-axis**: Forward (:math:`\phi = 0^\circ`, :math:`\alpha = 90^\circ`)
- **Y-axis**: Left
- **Z-axis**: Up

A 3D point :math:`P = (X, Y, Z)` in LiDAR coordinates converts to spherical coordinates using:

.. math::

   r = \sqrt{X^2 + Y^2 + Z^2} \\
   \phi = \operatorname{atan2}(Y, X) \\
   \alpha = \arccos\!\left(\frac{Z}{r}\right)

where :math:`r` is the radial distance stored as depth at image coordinate :math:`(u,v)`. The depth image coordinates are otained by applying the LiDAR intrisics to the spherical coordinates:

.. math::

   u = (\phi - \phi_{\text{start}}) \cdot \frac{N_\phi}{2\pi} \\
   v = (\alpha - \alpha_{\text{start}}) \cdot \frac{N_\alpha - 1}{\text{vfov}}

with intrinsic parameters defined in the table below:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Notation
     - Parameter name in :nvblox_code_link:`<nvblox/include/nvblox/sensors/lidar.h>`.
     - Description
   * - :math:`N_\phi`
     - ``num_azimuth_divisions``
     - Number of azimuth divisions (range image width)
   * - :math:`N_\alpha`
     - ``num_elevation_divisions``
     - Number of elevation divisions (range image height)
   * - :math:`\phi_{\text{start}}`
     - ``start_azimuth_angle_rad``
     - Starting azimuth angle for image mapping [rad]
   * - :math:`\alpha_{\text{start}}`
     - ``start_polar_angle_rad``
     - Starting polar angle (from +Z) for image mapping [rad]
   * - :math:`\text{vfov}`
     - ``vertical_fov_rad``
     - Vertical field of view [rad]

.. _extending-nvblox-to-support-other-sensors:

Extending nvblox to support other sensors
-----------------------------------------
For advanced users, it is possible to extend ``nvblox`` to support other sensors by implementing the interface defined in :nvblox_code_link:`<nvblox/include/nvblox/sensors/sensor.h>`.
For more details, see :nvblox_code_link:`<nvblox/include/nvblox/sensors/camera.h>`
and :nvblox_code_link:`<nvblox/include/nvblox/sensors/lidar.h>`.
Your sensor class must inherit from ``SensorBase`` and implement all methods checked by the ``is_sensor_interface`` trait.
Note that this can be done without modifying the core library.

Template Instantiations
~~~~~~~~~~~~~~~~~~~~~~~
After implementing your sensor class, you must provide explicit template instantiations for certain functions that are using them. Follow the pattern in :nvblox_code_link:`<nvblox/src/integrators/instantiations/camera_inst.cu>` and link with your own instantiations file when building your project.

Some implementation notes:
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Depth Representation**: Implement ``getDepth()`` to return the appropriate distance metric for your sensor (e.g., Euclidean distance for LiDAR, Z-distance for cameras).

- **Interpolation**: The ``interpolateDepthImage()`` method should handle sensor-specific interpolation logic, including discontinuity detection and ray-distance validation.

- **CUDA Compatibility**: All methods must be callable from both host and device code using ``__host__ __device__`` qualifiers.

- **Build Integration**: Add your instantiation file to the appropriate build target to ensure it gets compiled and linked.
