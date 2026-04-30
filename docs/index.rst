``nvblox`` Documentation
===========================

``nvblox`` is a library for real-time 3D reconstruction, designed for robotic applications.

With ``nvblox`` you can build, manipulate, and query reconstructions on the GPU.
The library is highly optimized for runtime performance, taking full advantage of ``CUDA`` and ``NVIDIA`` hardware.

Depending on your use-case, you may interact with ``nvblox`` through either Python, C++ or
`ROS2 <https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/index.html>`_.


.. _quickstart:

Quickstart
==========


:nvblox_torch_pip_install_code_block:

See :doc:`pages/installation` for more options.

With ``nvblox`` installed, you're ready to build your first reconstruction.
Follow our :doc:`pages/torch_examples_reconstruction` Example.
After that try out some more functionality below.

Examples
========

Below are some examples of building reconstructions using ``nvblox``.


.. list-table::
    :class: gallery
    :widths: auto

    * - .. image:: images/3dmatch.gif
         :height: 200px
         :target: pages/torch_examples_reconstruction.html
      - .. image:: images/desk_radio_x2_600px.gif
         :height: 200px
         :target: pages/torch_examples_deep_features.html
    * - :doc:`pages/torch_examples_reconstruction`
      - :doc:`pages/torch_examples_deep_features`
    * - .. image:: images/trajectory_optimization.gif
         :height: 200px
         :target: pages/torch_examples_gradients.html
      - .. image:: images/esdf_example.gif
         :height: 200px
         :target: pages/torch_examples_esdf.html
    * - :doc:`pages/torch_examples_gradients`
      - :doc:`pages/torch_examples_esdf`
    * - .. image:: images/voxels_example.gif
         :height: 200px
         :target: pages/torch_examples_voxel_access.html
      - .. image:: images/nvblox_torch_realsense_live.gif
         :height: 200px
         :target: pages/torch_examples_realsense.html
    * - :doc:`pages/torch_examples_voxel_access`
      - :doc:`pages/torch_examples_realsense`
    * - .. image:: images/nvblox_texture_mapping_replica.gif
         :height: 200px
         :target: pages/core_library_run_an_example.html
      -
    * - :doc:`Texture mapped reconstruction <pages/core_library_run_an_example>`
      -


License
-------
This code is under an `open-source license <https://github.com/nvidia-isaac/nvblox/blob/public/LICENSE.md>`_ (Apache 2.0).


Papers
------
If you find this library useful for your research, please consider citing our papers:

* Alexander Millane, Helen Oleynikova, Emilie Wirbel, Remo Steiner, Vikram Ramasamy, David Tingdahl, and Roland Siegwart.
  "**nvblox: GPU-Accelerated Incremental Signed Distance Field Mapping**".
  `arXiv preprint arXiv:2311.00626 (2024). <https://arxiv.org/abs/2311.00626>`_

* Sundaralingam, Balakumar, Siva Kumar Sastry Hari, Adam Fishman, Caelan Garrett, Karl Van Wyk, Valts Blukis, Alexander Millane et al.
  "**curobo: Parallelized collision-free minimum-jerk robot motion generation.**".
  `arXiv preprint arXiv:2310.17274 (2023). <https://arxiv.org/abs/2310.17274>`_


.. toctree::
   :maxdepth: 1
   :caption: User's Guide

   pages/installation

.. toctree::
   :maxdepth: 1
   :caption: python

   pages/torch_examples_reconstruction
   pages/torch_examples_deep_features
   pages/torch_examples_gradients
   pages/torch_examples_esdf
   pages/torch_examples_voxel_access
   pages/torch_examples_realsense

.. toctree::
   :maxdepth: 1
   :caption: c++

   pages/core_library_run_an_example
   pages/core_library_more_examples
   pages/core_library_interface
   pages/core_library_sensors

.. toctree::
   :maxdepth: 1
   :caption: Renderer

   pages/renderer_overview
   pages/renderer_getting_started
   pages/renderer_api

.. toctree::
   :maxdepth: 1
   :caption: Other

   pages/contributing
   pages/technical_details
   pages/limitations
