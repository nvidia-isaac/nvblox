API Reference
=============

All renderer types live in the ``nvblox::renderer`` namespace. The primary header is
``nvblox/renderer/renderer.h``.

.. contents:: On This Page
   :local:
   :depth: 2

.. note::

   This page is auto-generated from source-code doc comments via Doxygen + Breathe.


Configuration
-------------

.. doxygenstruct:: nvblox::renderer::RendererConfig
   :members:

.. doxygenenum:: nvblox::renderer::RenderMode


NvbloxRenderer
--------------

.. doxygenclass:: nvblox::renderer::NvbloxRenderer
   :members:

In windowed mode, ``render()`` handles special cases automatically:

- **Minimized window**: Returns ``true`` immediately (no draw, no error).
- **Window resize**: The swapchain is recreated automatically. One frame is skipped
  after the resize. The ``ViewCamera`` aspect ratio updates automatically from the
  new framebuffer size.

The background clear color is dark gray ``(0.1, 0.1, 0.1)``.


Visualizers
-----------

All visualizers share a common ``bool hasData() const`` method that returns ``true``
if the visualizer has data to render.

.. note::

   Visualizers are **not thread-safe**. All calls (init, destroy, render, update) must
   come from a single thread or be externally synchronized.


ImageVisualizer
~~~~~~~~~~~~~~~

.. doxygenstruct:: nvblox::renderer::ImageVisualizerConfig
   :members:

.. doxygenclass:: nvblox::renderer::ImageVisualizer
   :members:

Rendering behavior:

- Depth pixels that are ``≤ 0`` or greater than ``max_depth`` are rendered as
  gray ``(0.2, 0.2, 0.2)``. Depth below ``min_depth`` (but positive) is
  clamped to the low end of the colormap rather than rendered as gray.
- In ``kOverlay`` layout, valid depth pixels are blended on top of the color image at
  alpha ``0.5``; pixels where the depth is invalid (``≤ 0`` or ``> max_depth``) show
  the color image unmodified.
- In ``kSideBySide`` layout, the viewport is split at the horizontal midpoint.

.. note::

   ``ImageVisualizer::setDepthRange()`` (defaults ``0.1`` / ``5.0``) sets the
   colormap normalization range for the depth texture display. This is a **different
   setting** from ``NvbloxRenderer::setDepthRange()`` (defaults ``0.1`` / ``10.0``),
   which controls depth filtering inside ``updatePointCloud()``. The two ranges are
   independent and can be configured separately.


PointCloudVisualizer
~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: nvblox::renderer::PointCloudPoint
   :members:

.. doxygenclass:: nvblox::renderer::PointCloudVisualizer
   :members:

Rendering behavior:

- Points are rendered as **circles** (not squares).
- Vertex colors undergo **sRGB-to-linear** conversion.


MeshVisualizer
~~~~~~~~~~~~~~

.. doxygenclass:: nvblox::renderer::MeshVisualizer
   :members:

Rendering behavior:

- **Backface culling is disabled.** Both sides of every triangle are rendered.
- Vertex colors undergo **sRGB-to-linear** conversion.
- When a texture atlas is uploaded and UVs are non-negative, the texture is sampled.


Low-Level Data Types
~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: nvblox::renderer::MeshVertex
   :members:

.. doxygenfunction:: nvblox::renderer::interleaveMeshVertexData


ViewCamera
----------

.. doxygenclass:: nvblox::renderer::ViewCamera
   :members:

Default projection parameters: FOV 60°, aspect 16:9, near 0.01, far 100.0.
Default distance from target: 3.0 m.


Coordinate Conventions
----------------------

The renderer uses a **right-handed, Y-up** coordinate system for 3D visualization.
The view matrix follows the OpenGL ``lookAt`` convention (forward = ``-Z`` in eye space,
up = ``+Y``). Matrices are stored column-major (Eigen default).

Point Clouds
~~~~~~~~~~~~

``updatePointCloud()`` unprojects depth pixels using nvblox camera intrinsics (CV camera
frame: X-right, Y-down, Z-forward) and writes them into the display frame by **negating
Y** (CV Y-down → Y-up) and **negating X** as a mirror flip for an intuitive
selfie-style view. The Z axis is passed through unchanged. The resulting point cloud is
**camera-centered** -- no world transform is applied.

.. note::

   The X negation is a mirror, not a rigid coordinate-system rotation, so the point
   cloud is not a metrically-correct representation of the scene in any canonical
   world frame. If you need world-frame points, compute them yourself and use
   ``PointCloudVisualizer::updatePoints()`` instead.

When using the low-level ``PointCloudVisualizer::updatePoints()``, coordinates are
passed through as-is and the caller is responsible for axis conventions.

Meshes
~~~~~~

``updateMesh()`` passes vertex positions through **without any axis flip or transform**.
Meshes are rendered in whatever coordinate frame the input data uses (typically the
nvblox world frame for ``ColorMesh``).

.. note::

   Point clouds from ``updatePointCloud()`` are in a **camera-centered** frame, while
   meshes are in the **nvblox world frame**. To display both in the same scene, use
   ``PointCloudVisualizer::updatePoints()`` with world-frame points instead of
   ``updatePointCloud()``.


Initialization Model
--------------------

``NvbloxRenderer`` uses a two-phase lifecycle: a trivial default constructor
followed by an explicit ``init()`` (or ``initWithWindow()`` /
``initHeadless()``) call. There is no constructor that takes a
``RendererConfig`` directly.

State before ``init()``
~~~~~~~~~~~~~~~~~~~~~~~

A default-constructed renderer is **not yet ready to render**. Until
``init()`` succeeds:

- Query methods (``isInitialized()``, ``isHeadless()``, ``renderMode()``,
  ``cameraControlsEnabled()``) and the configuration setters are safe to
  call.
- ``init*()``, ``initVisualizer()``, and ``render()`` return ``false``
  cleanly.
- The update methods (``updateDepth``, ``updateColor``, ``updatePointCloud``,
  ``updateMesh``, ``updateMeshTexture``) log a warning and return ``false``.
- ``pollEvents()`` is a no-op and ``shouldClose()`` returns ``true``.
- ``destroy()`` is a safe no-op.
- The visualizer, camera, and context accessors
  (``imageVisualizer()``, ``pointCloudVisualizer()``, ``meshVisualizer()``,
  ``viewCamera()``, ``context()``) return ``nullptr``. Any chained call
  through those pointers will crash.

Even after ``init()`` succeeds, the per-mode accessors remain ``nullptr``
until ``initVisualizer(mode)`` is called for that mode. ``viewCamera()`` is
created lazily on the first ``initVisualizer(kPointCloud)`` or
``initVisualizer(kMesh)`` call, and stays ``nullptr`` if you only use
``kImage``.

Why initialization is split out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Init can fail.** Vulkan setup may fail because no driver is installed,
   no GPU is compatible, or no display server is available. The renderer
   reports such failures via ``bool`` return so callers can fall back to a
   different configuration.
2. **Re-initialization.** After ``destroy()``, the same renderer can be
   ``init()``-ed again with a different configuration — for example to
   switch between windowed and headless mode, or change the resolution.
   Note that switching between render modes (e.g. ``kPointCloud`` →
   ``kMesh``) does **not** require destroy + re-init: call
   ``initVisualizer()`` once for each mode you want, then flip between them
   at runtime with ``setRenderMode()``.
3. **Use as a class member.** Owning classes can declare
   ``NvbloxRenderer renderer_;`` and initialize it later, once their
   configuration is known (from CLI flags, config files,
   etc.).


Error Handling
--------------

All ``init`` and ``render`` methods return ``bool``:

- **Initialization failures** return ``false`` and log details via ``glog``.
- **Update methods** (``updateDepth``, ``updateColor``, etc.) return ``false`` if
  the corresponding visualizer has not been initialized.
- **``render()``** returns ``false`` only on Vulkan errors. If the active mode's
  visualizer is missing, it logs a warning (rate-limited to once per mode) and
  returns ``true`` with nothing drawn.
- **Double initialization** of the renderer (without ``destroy()``) returns ``false``.
  Re-initializing a visualizer via ``initVisualizer()`` with the same mode destroys
  and recreates it (with a warning).


Thread Safety
-------------

``NvbloxRenderer`` and all visualizers are **not thread-safe**. All calls must originate
from the same thread.

In **windowed mode** that thread must specifically be the program's **main thread**.
GLFW only officially supports window creation, ``glfwPollEvents()``, and window
destruction on the main thread; calling them from a worker thread may appear to
work but can deadlock or misbehave under some X11 / Wayland compositors.

In **headless mode** there is no GLFW and no window, so any single thread works.

The synchronization contract between CUDA and Vulkan is:

1. Call the update methods (``updateDepth``, ``updateColor``,
   ``updatePointCloud``, ``updateMesh``, ``updateMeshTexture``) with a CUDA
   stream.
2. Call ``stream.synchronize()`` to ensure all CUDA writes are complete.
3. Call ``render()`` to submit Vulkan commands.

This CPU-side synchronization is used for simplicity and portability.


Limits
------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Limit
     - Value
   * - Max points / vertices / triangles
     - 50,000,000 each
   * - Max buffer size
     - 1 GB
   * - Texture dimensions
     - 1 to 16,384 per side

Buffers grow automatically as needed. If a data update exceeds the maximum count
or buffer size, the renderer logs an error and silently drops the update (the
previous data remains unchanged).
