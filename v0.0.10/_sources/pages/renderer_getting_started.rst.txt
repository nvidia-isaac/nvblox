Getting Started
===============

This page shows how to set up and run the ``nvblox`` renderer in both windowed and
headless modes.

Prerequisites
-------------

1. Build ``nvblox`` with the renderer enabled:

   .. code-block:: bash

       cmake -DBUILD_RENDERER=ON ..
       make -j6

2. Ensure a Vulkan-capable GPU driver is installed. You can verify with:

   .. code-block:: bash

       vulkaninfo --summary

3. For windowed mode, a display server (X11 or Wayland) must be available.
   In Docker containers, either use headless mode or forward the display.


Lifecycle Overview
------------------

``NvbloxRenderer`` uses an explicit, multi-stage lifecycle:

1. **Construct** — default-construct the object. No Vulkan, GLFW, or GPU
   resources are touched yet.
2. **Initialize** — call ``init()`` (or ``initWithWindow()`` /
   ``initHeadless()``). This creates the Vulkan instance, picks a physical GPU
   device, creates the render target, etc., and returns ``false`` on failure.
3. **Initialize visualizers** — call ``initVisualizer(mode)`` once per
   render mode you intend to use.
4. **Render loop** — for each frame: push new data via the update methods
   (``updateDepth()``, ``updateColor()``, ``updatePointCloud()``,
   ``updateMesh()``, ``updateMeshTexture()``) on a CUDA stream, then
   ``stream.synchronize()``, ``render()``, and ``pollEvents()``.
5. **Destroy** — the destructor calls ``destroy()`` automatically. You may
   also call ``destroy()`` explicitly to release Vulkan resources early, and
   you may re-initialize the same object afterward.

Why initialization is split out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vulkan setup can fail for many reasons (missing driver, no compatible GPU,
surface or swapchain creation, …) and the renderer reports those via a
``bool`` return plus glog rather than exceptions. Splitting ``init()``
from the constructor also lets you hold a renderer as a member before its
config is known, and tear down and re-initialize the same object — for
example, to change resolution.

Behavior before ``init()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before ``init()`` succeeds, the object is constructed but inert. All
mutating methods are guarded and safe to call:

- ``render()``, ``initVisualizer()``, and the update methods return
  ``false`` cleanly.
- ``pollEvents()`` is a no-op.
- ``shouldClose()`` returns ``true``.
- ``destroy()`` is a no-op.
- Query (``isInitialized()``, ``isHeadless()``, ``renderMode()``,
  ``cameraControlsEnabled()``) and setter methods behave normally.

Accessor pointers
~~~~~~~~~~~~~~~~~

The pointer accessors each become non-null only after a specific
initialization step has succeeded. Always null-check before dereferencing —
a chained call like ``renderer.imageVisualizer()->resizeDepthTexture(...)``
will crash if the accessor is ``nullptr``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Accessor
     - Becomes non-null after
   * - ``context()``
     - ``init()`` / ``initWithWindow()`` / ``initHeadless()`` succeeds.
   * - ``imageVisualizer()``
     - ``initVisualizer(RenderMode::kImage)`` succeeds.
   * - ``pointCloudVisualizer()``
     - ``initVisualizer(RenderMode::kPointCloud)`` succeeds.
   * - ``meshVisualizer()``
     - ``initVisualizer(RenderMode::kMesh)`` succeeds.
   * - ``viewCamera()``
     - first ``initVisualizer(kPointCloud)`` or ``initVisualizer(kMesh)``
       call. Stays ``nullptr`` if only ``kImage`` is used.

.. warning::

   In **windowed mode**, all renderer calls — construction, ``init*()``,
   ``render()``, ``pollEvents()``, ``shouldClose()``, ``destroy()`` — must
   come from the program's **main thread**. GLFW only officially supports
   window creation and event polling on the main thread; calling them from
   a worker thread may appear to work but can deadlock or misbehave under
   some X11 / Wayland compositors.

   In **headless mode** there is no window and no GLFW, so any single
   thread works.


The Render Loop
---------------

Every frame follows the same three steps:

1. **Update** — push new data with ``updateDepth()``, ``updateColor()``,
   ``updatePointCloud()``, ``updateMesh()``, or ``updateMeshTexture()``.
   These enqueue asynchronous CUDA work on the stream you pass in.
2. **Synchronize** — call ``stream.synchronize()`` (or
   ``cudaStreamSynchronize``).
3. **Render** — call ``render()`` to draw the frame, then ``pollEvents()``
   to keep the window responsive.

.. warning::

   Step 2 is **not optional**. The renderer shares memory between CUDA and
   Vulkan: the update methods enqueue async CUDA writes into the same
   allocation that ``render()`` reads on the Vulkan side, and Vulkan does
   not see the CUDA stream's ordering. Without the sync, ``render()`` may
   sample texels that CUDA is still writing — no error is reported, you
   will just see garbage or torn frames.

   CPU sync is used here for simplicity and portability. If you need to
   overlap CPU work with rendering, GPU-side sync via Vulkan timeline
   semaphores is possible by modifying ``VkContext::endFrame()``.


Windowed Example
----------------

A complete windowed application using the configuration struct:

.. code-block:: cpp

    #include "nvblox/renderer/renderer.h"

    using namespace nvblox::renderer;

    // 1. Configure and initialize
    NvbloxRenderer renderer;
    RendererConfig config;
    config.width = 1280;
    config.height = 720;
    config.title = "My Viewer";
    renderer.init(config);

    // 2. Initialize the image visualizer and resize textures
    renderer.initVisualizer(RenderMode::kImage);
    renderer.imageVisualizer()->resizeDepthTexture(640, 480);
    renderer.imageVisualizer()->resizeColorTexture(1280, 720);

    // 3. Render loop
    CudaStreamOwning stream;
    while (!renderer.shouldClose()) {
        renderer.updateDepth(depth_image, stream);
        renderer.updateColor(color_image, stream);

        // IMPORTANT: synchronize CUDA before rendering
        stream.synchronize();

        renderer.render();
        renderer.pollEvents();
    }

You can also use the convenience method ``initWithWindow()`` to skip the config
struct:

.. code-block:: cpp

    renderer.initWithWindow(1280, 720, "My Viewer");


Point Cloud Example
-------------------

To visualize RGBD data as a 3D point cloud with interactive camera controls:

.. code-block:: cpp

    renderer.init(config);
    renderer.initVisualizer(RenderMode::kPointCloud);

    // Optional: narrow the depth range (defaults are 0.1 m / 10.0 m).
    renderer.setDepthRange(0.1f, 5.0f);

    CudaStreamOwning stream;
    while (!renderer.shouldClose()) {
        renderer.updatePointCloud(depth_image, color_image,
                                  depth_cam, color_cam, stream);
        stream.synchronize();
        renderer.render();
        renderer.pollEvents();
    }


Mesh Example
------------

To render a ``ColorMesh`` produced by ``nvblox``:

.. code-block:: cpp

    renderer.init(config);
    renderer.initVisualizer(RenderMode::kMesh);

    CudaStreamOwning stream;
    while (!renderer.shouldClose()) {
        renderer.updateMesh(color_mesh, stream);
        stream.synchronize();
        renderer.render();
        renderer.pollEvents();
    }

For textured meshes, also upload the texture atlas:

.. code-block:: cpp

    renderer.updateMeshTexture(atlas_image, stream);


Headless Example
----------------

Headless mode renders offscreen without creating a window. This is useful for
automated testing, CI pipelines, or server-side rendering.

.. code-block:: cpp

    RendererConfig config;
    config.width = 1920;
    config.height = 1080;
    config.headless = true;
    renderer.init(config);

    renderer.initVisualizer(RenderMode::kMesh);

    CudaStreamOwning stream;

    // Control the camera programmatically
    ViewCamera* cam = renderer.viewCamera();
    cam->setTarget(0.0f, 0.0f, 0.0f);
    cam->setDistance(3.0f);
    cam->setOrbitAngles(0.5f, 0.3f);

    // Render a single frame
    renderer.updateMesh(mesh, stream);
    stream.synchronize();
    renderer.render();

You can also use the convenience method ``initHeadless()`` to skip the config struct:

.. code-block:: cpp

    renderer.initHeadless(1920, 1080);

.. note::

   In headless mode, ``shouldClose()`` always returns false (there is no window close
   event). Use your own termination condition for render loops.
   ``pollEvents()`` is a no-op and camera controls have no effect since there is no input.
   Use ``ViewCamera`` methods directly to position the camera.


Input Controls
--------------

In windowed mode (3D modes: point cloud and mesh), the renderer provides interactive
camera controls:

- **Left-drag**: Arcball rotation
- **Right-drag**: Pan
- **Scroll**: Zoom
- **R key**: Reset camera to default position

Camera controls can be disabled at runtime:

.. code-block:: cpp

    renderer.setCameraControlsEnabled(false);

    // Re-enable later
    renderer.setCameraControlsEnabled(true);

.. note::

   When camera controls are disabled, mouse and scroll input is ignored.
   The ``R`` key reset and user key callbacks still fire regardless of this setting.


Key Callbacks
-------------

Register a custom key callback to handle application-specific keyboard input.
The callback receives GLFW key constants:

.. code-block:: cpp

    renderer.setKeyCallback([](int key, int action, int mods) {
        if (action == GLFW_PRESS) {
            switch (key) {
                case GLFW_KEY_W:
                    // Toggle wireframe on mesh visualizer
                    break;
                case GLFW_KEY_ESCAPE:
                    // Request close
                    break;
            }
        }
    });

The callback receives **all** key events, including the built-in ``R`` key for
camera reset. Built-in handling runs first, then the user callback is invoked.

.. note::

   ``setKeyCallback`` has no effect in headless mode (there is no window to receive
   key events).


Switching Render Modes
----------------------

You can switch between visualizers at runtime. Initialize all the modes you need
upfront, then switch with ``setRenderMode()``:

.. code-block:: cpp

    // Initialize all visualizers
    renderer.initVisualizer(RenderMode::kImage);
    renderer.imageVisualizer()->resizeDepthTexture(640, 480);
    renderer.imageVisualizer()->resizeColorTexture(1280, 720);
    renderer.initVisualizer(RenderMode::kPointCloud);
    renderer.initVisualizer(RenderMode::kMesh);

    CudaStreamOwning stream;
    while (!renderer.shouldClose()) {
        // Update data for each mode as needed...

        stream.synchronize();

        // Switch modes at any time
        if (show_mesh) {
            renderer.setRenderMode(RenderMode::kMesh);
        } else {
            renderer.setRenderMode(RenderMode::kPointCloud);
        }

        renderer.render();
        renderer.pollEvents();
    }

``setRenderMode()`` is instant -- it only changes which visualizer ``render()``
dispatches to, with no teardown or re-initialization.

If ``render()`` is called with a mode whose visualizer has not been initialized, it
logs a warning (once) and renders nothing, but still returns ``true``.


Lifecycle and Cleanup
---------------------

The destructor calls ``destroy()`` automatically, but you can also call it explicitly
to release Vulkan resources early:

.. code-block:: cpp

    renderer.destroy();

After ``destroy()``, you can re-initialize the renderer:

.. code-block:: cpp

    renderer.destroy();
    renderer.initHeadless(1024, 768);  // Re-init with new dimensions
    renderer.initVisualizer(RenderMode::kMesh);

.. note::

   Calling ``init*()`` on an already-initialized renderer returns ``false``.
   Call ``destroy()`` first to re-initialize.
