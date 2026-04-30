Renderer Overview
=================

``nvblox_renderer`` is a Vulkan-based visualization library for ``nvblox`` data. It renders
depth images, color images, point clouds, and triangle meshes using shared CUDA-Vulkan memory,
avoiding unnecessary copies between the GPU compute and graphics pipelines.

Capabilities
------------

The renderer supports three visualization modes via the ``RenderMode`` enum:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Description
     - Input Data
   * - ``kImage``
     - Renders depth and color images as 2D textured quads with depth colormapping.
     - ``DepthImage``, ``ColorImage``
   * - ``kPointCloud``
     - Renders 3D colored point clouds from RGBD data.
     - ``DepthImage``, ``ColorImage``, ``Camera`` intrinsics
   * - ``kMesh``
     - Renders 3D triangle meshes with per-vertex color or texture atlas.
     - ``ColorMesh`` (or raw vertex/index arrays)

Both **windowed** and **headless** rendering are supported:

- **Windowed mode** opens a GLFW window with interactive arcball camera controls
  (left-drag to rotate, right-drag to pan, scroll to zoom, ``R`` to reset camera).
- **Headless mode** renders offscreen without a window, useful for testing, CI, or
  server-side rendering. Use ``ViewCamera`` methods directly to control the viewpoint.

The renderer uses a **right-handed, Y-up** coordinate system. Meshes are rendered in
their original coordinate frame (typically the nvblox world frame), while point clouds
from ``updatePointCloud()`` are converted from the CV camera frame to a Y-up display
frame (Y negated) and mirrored left/right (X negated) for an intuitive selfie-style
view. See :doc:`renderer_api` for full coordinate convention details.

Architecture
------------

The renderer is designed for **real-time visualization** of nvblox data with minimal
overhead. It achieves this by sharing GPU memory between CUDA and Vulkan -- nvblox
compute outputs flow directly into Vulkan-visible buffers without ever leaving the GPU,
eliminating the costly GPU-to-CPU-to-GPU round-trip of a naive rendering approach.

Vulkan buffers and textures are allocated as shared GPU memory that both CUDA and Vulkan
can access. When nvblox produces a mesh, depth image, or point cloud on the GPU, the
renderer writes that data directly into the shared allocation. Vulkan then renders from
the same memory with no additional copy to host.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Data Flow
   * - Mesh
     - nvblox ``ColorMesh`` vertex data is interleaved by a CUDA kernel that writes
       directly into the shared Vulkan vertex buffer. Index data is copied within VRAM.
   * - Image
     - Depth and color images are copied from nvblox device memory into shared Vulkan
       textures via GPU-to-GPU transfer.
   * - Point Cloud
     - A CUDA kernel converts RGBD data to colored 3D points, then the result is copied
       into the shared Vulkan vertex buffer.

All data stays on the GPU throughout. The only CPU involvement is a
``stream.synchronize()`` call before ``render()`` to ensure CUDA writes are complete
before Vulkan reads. Point-cloud mode additionally performs one device-to-host copy
of an atomic point counter inside ``updatePointCloud()`` (one extra sync per frame)
so the visualizer knows how many valid points the conversion kernel produced.

In a real-time application, the integration pattern is straightforward: run nvblox
mapping and meshing on a CUDA stream, call the renderer's update methods, synchronize
the stream, and render. The renderer is shipped as a single shared library
(``libnvblox_renderer``). Users interact primarily with the ``NvbloxRenderer`` class,
plus the visualizer accessors (``imageVisualizer()``, ``pointCloudVisualizer()``,
``meshVisualizer()``) and ``ViewCamera`` for mode-specific controls; all Vulkan and
CUDA interop details are handled internally.

Dependencies
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dependency
     - Requirement
   * - CUDA
     - >= 12.0 (``nvcc`` 11.x cannot compile C++17 headers used by Vulkan interop)
   * - Vulkan
     - >= 1.2 (required for timeline semaphores)
   * - GLFW
     - >= 3.3 (auto-fetched if not found on system)
   * - glslangValidator
     - Required at build time to compile GLSL shaders to SPIR-V
       (``apt-get install glslang-tools``)
   * - glog
     - Provided by the ``nvblox`` core library

.. note::

   On multi-GPU systems, the renderer automatically selects the Vulkan physical device
   whose UUID matches the GPU that nvblox is running on.

Building
--------

The renderer is built by default (``BUILD_RENDERER`` is ``ON``). To disable it, pass
``-DBUILD_RENDERER=OFF``:

.. code-block:: bash

    cmake -DBUILD_RENDERER=OFF ..

When enabled, CMake compiles the GLSL shaders under ``nvblox_renderer/shaders/`` to SPIR-V
and installs them alongside the library. The compiled shaders are loaded at runtime from
the install path. If the install prefix does not match the runtime path, set the
``NVBLOX_SHADER_DIR`` environment variable to the directory containing the compiled
``.spv`` files:

.. code-block:: bash

    export NVBLOX_SHADER_DIR=/path/to/share/nvblox_renderer/shaders

.. note::

   On systems without a display server (e.g. headless machines, Docker containers),
   the library still builds and links normally. Use headless mode
   (``RendererConfig::headless = true``) at runtime; windowed mode will fail to create
   a window on displayless systems.
