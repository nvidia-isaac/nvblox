Technical Details
=================

This page gives some technical details about the operation of ``nvblox`` under the hood.

This page is a poor substitute for the scientific papers which describe the underpinnings
of the methods we rely on.
The most important papers to read in chronological order are:

- **KinectFusion** [1]_ Describes TSDF mapping from depth images.
- **Real-time 3D reconstruction at scale using voxel hashing** [2]_ Introduces voxel hashing (on the CPU) which is the map structure used by ``nvblox``.
- **Voxblox** [3]_ Introduces the use of a voxel-hashed TSDF map for robotics (on the CPU)
- **nvblox** [4]_ Describes ``nvblox``, which combines the preceeding papers but moves the algorithms to the GPU.


Map Representation
------------------

By default ``nvblox`` builds the reconstructed map in the form of a Truncated Signed
Distance Function (TSDF) stored in a 3D voxel grid. We also support occupancy
grids.

The TSDF approach is similar to 3D occupancy grid mapping approaches in which
occupancy probabilities are stored at each voxel. In contrast however, TSDF-based
approaches (like nvblox) store the (signed) distance to the closest surface at each voxel.
The surface of the environment can then be extracted as the zero-level set of this
voxelized function.

Typically TSDF-based reconstructions provide higher quality surface reconstructions.
In addition, distance fields are also useful for path planning because they provide
an immediate means of checking whether potential future robot positions are in collision.
This fact, the utility of distance functions for both reconstruction and planning,
motivates their use in nvblox (a reconstruction library for path planning).

Map Structure
-------------

.. image:: ../images/map_structure.png
   :width: 400px
   :align: right
   :alt: Map structure

We implement a hierarchical sparse voxel grid for storing data.

- ``LayerCake``: At the top level we have the ``LayerCake``, which contains several ``Layers``,
  which are colocated voxel grids.
  Each ``Layer`` (voxel-grid) contains a different type of mapped quantity (eg TSDF and ESDF).

- ``Layer``: A ``Layer`` is a sparse voxel grid that contains a single type of mapped quantity (eg TSDF).
  Inside a ``Layer`` we have a sparse collection of ``(Voxel)Blocks``.

- ``(Voxel)Blocks``: A grid element in a layer. Each ``Block`` block defines the map in
  a small cubical region of space. With a ``Block`` voxels are densely allocated in an
  8x8x8 grid.

- ``Voxels``: The smallest unit of resolution. Holds a single value of the mapped quantity (eg the TSDF).

Details of how to interact with the map can be found in :doc:`core_library_interface` (c++),
and :doc:`torch_examples_voxel_access` (python).

This hierarchy allows us several useful properties:

- **Sparsity:** We only allocate memory for the voxels that are mapped. The map
  can grow and shrink dynamically as needed.

- **Data Locality:** A ``Block`` is stored in a contiguous chunk of memory. Our GPU kernels
  are often implemented such that CUDA *ThreadBlocks* process single ``VoxelBlocks``. Within
  such kernels we benefit from coalesced access to global GPU memory.

- **Extensibility:** New mapped quantities can be added to the map by new ``Layer``\s.
  Users can define their own ``Layers`` to map custom quantities alongside those
  built into the ``nvblox`` library.



References
----------

.. [1] Newcombe, Richard A., Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim,
    Andrew J. Davison, Pushmeet Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgibbon.
    "Kinectfusion: Real-time dense surface mapping and tracking." In 2011 10th IEEE
    international symposium on mixed and augmented reality, pp. 127-136. Ieee, 2011.

.. [2] Nießner, Matthias, Michael Zollhöfer, Shahram Izadi, and Marc Stamminger.
    "Real-time 3D reconstruction at scale using voxel hashing." ACM Transactions
    on Graphics (ToG) 32, no. 6 (2013): 1-11.

.. [3] Oleynikova, Helen, Zachary Taylor, Marius Fehr, Roland Siegwart, and Juan Nieto.
    "Voxblox: Incremental 3d euclidean signed distance fields for on-board mav planning."
    In 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),
    pp. 1366-1373. IEEE, 2017.

.. [4] Millane, Alexander, Helen Oleynikova, Emilie Wirbel, Remo Steiner, Vikram Ramasamy,
    David Tingdahl, and Roland Siegwart. "nvblox: Gpu-accelerated incremental signed distance
    field mapping." In 2024 IEEE International Conference on Robotics and Automation (ICRA),
    pp. 2698-2705. IEEE, 2024.
