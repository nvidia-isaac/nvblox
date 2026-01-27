Limitations
===========

This is the first official release of ``nvblox_torch``, and therefore
some features are still under development.

- **Reconstruction quality:** ``nvblox`` is dependent on the quality of the input data.
  Poor depth maps will result in a poor reconstruction.
- **Compute performance:** The ``nvblox`` core library and ROS wrapper have been
  optimized for performance over several years. ``nvblox_torch`` is intended to provide an
  easy to use interface and hasn't been optimized to the same degree (although we've taken
  care to provide zero-copy interfaces).
- **Memory usage:** The memory consumed by ``nvblox`` scales with the volume of mapped space,
  and cubicly with increasing resolution.
  The library will happily run out of of memory and crash if you map a larger volume/higher
  resolution than you have GPU memory for.
- **Memory usage in deep feature reconstruction:** Our Deep Feature Reconstruction is particularly
  memory intensive, exacerbating the issues above.This is a fundamental limitation of storing
  long channel length features in 3D voxels.
- **Mapping with dynamic scene elements:** The ``nvblox`` core library and our ROS wrapper
  supports mapping in the presence of moving elements in the scene. See for example in
  `isaac_ros_nvblox people segmentation example <https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/tutorials/tutorial_realsense.html#reconstruction-with-people-segmentation>`_
  or `isaac_ros_nvblox segmentation-free example <https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/tutorials/tutorial_realsense.html#reconstruction-with-dynamic-scene-elements>`_
  This is not yet supported in ``nvblox_torch``.
- **Incremental visualization:** The ``nvblox`` core library and our ROS wrapper supports
  incremental visualization, that is: only streaming parts of the visualization, for example
  the mesh, to the visualization pipeline. This is not yet supported in ``nvblox_torch``.

If you need a feature for your project, please leave a feature request on
`github issues <https://github.com/nvidia-isaac/nvblox/issues>`_.
We are very interested in external contributions.
If you add a missing feature to ``nvblox_torch``, please consider contributing
your code back to the project.
