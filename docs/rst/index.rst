=======
Introduction to nvblox
=======

Nvblox is a package for building a 3D reconstruction of the environment around your robot from sensor observations in real-time. The reconstruction is intended to be used by path planners to generate collision-free paths. Under the hood, nvblox uses NVIDIA CUDA to accelerate this task to allow operation at real-time rates. This repository contains ROS2 integration for the nvblox core library.

|pic1| |pic2|

.. |pic1| image:: ./images/reconstruction_in_docker_trim.png
   :width: 45%

.. |pic2| image:: /images/nvblox_navigation_trim.gif
   :width: 45%

**Left**: nvblox used for reconstruction on a scan from the `Sun3D Dataset <http://sun3d.cs.princeton.edu/>`_.
**Right**: the nvblox ROS2 wrapper used to construct a costmap for `ROS2 Nav2 <https://navigation.ros.org/>`_ for navigating of a robot inside `Isaac Sim <https://developer.nvidia.com/isaac-sim>`_.

Nvblox is composed of two packages

* `nvblox Core Library <https://gitlab-master.nvidia.com/nvblox/nvblox>`_ Contains the core C++/CUDA reconstruction library.
* `nvblox ROS2 Interface <https://gitlab-master.nvidia.com/isaac_ros/isaac_ros_nvblox>`_ Contains a ROS2 wrapper and integrations for simulation and path planning. Internally builds the core library.




.. .. figure:: ./images/reconstruction_in_docker_trim.png
..     :width: 50 %
..     :align: center

..     nvblox used for reconstruction on a scan from the `Sun3D Dataset http://sun3d.cs.princeton.edu/`_
