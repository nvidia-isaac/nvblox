Run an Example
==============

.. note::
    This example assumes that ``nvblox`` has been built from source — see :ref:`nvblox_torch_source_installation`.

The example demonstrates real-time mesh reconstruction with projective texture mapping. We use `Replica
<https://github.com/facebookresearch/Replica-Dataset>`_ sequences from the `NICE-SLAM
<https://github.com/cvg/nice-slam>`_ project.

:download_replica_test_dataset:

From the nvblox base folder run:

.. code-block:: bash

    build/executables/fuse_replica Replica/room0 --voxel_size=0.02

.. note::

    The ``--voxel_size=0.02`` flag sets the voxel edge length to 2 cm. If reconstruction runs slowly (e.g. on a less capable GPU) try a coarser resolution, for example ``0.05`` (5 cm).

A visualizer window will open and you should see the mesh being incrementally reconstructed in real time:

.. image:: ../images/nvblox_texture_mapping_replica.gif
   :align: center
   :width: 600px


.. admonition:: Camera Controls

    - **Left-drag**: Rotate the view (arcball rotation)
    - **Right-drag**: Pan
    - **Scroll**: Zoom in / out
    - **F**: Toggle camera follow on / off
    - **T**: Toggle texture mapping on / off

For more details on the visualizer, see :doc:`renderer_overview`.

More examples of running nvblox on datasets are given in :doc:`core_library_more_examples`

.. note::

    Ensure that the DISPLAY environment variable is set to the display you want to use.
    This is typically the case when logged in through a graphical desktop environment, but might not be set when running through a containerized environment.
    Depending on your system, it might be necessary to explicitly set the variable:

    .. code-block:: bash

        export DISPLAY=:0

Visualizing without renderer support
---------------------------------------------------------

If nvblox was built without the renderer (``-DBUILD_RENDERER=OFF``), no visualizer
window will appear. You may append ``mesh.ply`` to the fuse command in order to save the resulting mesh:

.. code-block:: bash

    build/executables/fuse_replica Replica/office0 mesh.ply

The mesh can be opened in the Open3D viewer:


Install Open3D:

.. code-block:: bash

    sudo apt-get install libglib2.0-0 libgl1
    pip3 install open3d

Then visualize the mesh:

.. code-block:: bash

    open3d draw mesh.ply

.. note::

   If Open3D is not available for your platform, any tool that can visualize PLY files (such as
   `CloudCompare <https://www.cloudcompare.org>`_ and `Meshlab <https://www.meshlab.net/>`_) may be used instead.
