3D Reconstruction
==================

This example demonstrates using ``nvblox_torch`` to reconstruct a scene from the `SUN3D
dataset <https://sun3d.cs.princeton.edu/>`_.

:download_sun3d_test_dataset:

Launch the example by running:

.. code-block:: bash

    python3 -m nvblox_torch.examples.reconstruction.sun3d \
        --dataset_path <PATH>/sun3d-mit_76_studyroom-76-1studyroom2/

.. image:: ../images/3dmatch.gif
   :width: 600px
   :alt: Reconstruction Example

The code for this example can be found at
:nvblox_code_link:`<nvblox_torch/nvblox_torch/examples/reconstruction/sun3d.py>`


Details
-------

We first create a ``torch`` dataloader to read the image data off the disk:

.. code-block:: python

    dataloader = Sun3dDataset.create_dataloader(root_dir=args.dataset_path,
                                                sequence_name=args.sequence_name)


We then create a ``Mapper``, and specify a couple of parameters, in particular:

- a voxel size from the command line, and
- the parameter ``projective_integrator_max_integration_distance_m`` is set to ``5`` meters.
  This defines the maximum depth from the camera that depth data is integrated into the reconstruction.

.. code-block:: python

    # Create some parameters
    projective_integrator_params = ProjectiveIntegratorParams()
    projective_integrator_params.projective_integrator_max_integration_distance_m = 5.0
    mapper_params = MapperParams()
    mapper_params.set_projective_integrator_params(projective_integrator_params)

    # Create the mapper
    mapper = Mapper(
        voxel_sizes_m=args.voxel_size_m,
        mapper_parameters=mapper_params,
    )

The ``Mapper`` is the main interface for ``nvblox_torch``.
Internally the ``Mapper`` holds the map, which has several voxel-``Layers`` and provides functions for:

- adding data to the map (for example adding a depth image ``mapper.add_depth_frame()``),
- generating dependant layers (for example generating a mesh ``mapper.update_color_mesh()``),  and
- getting access to voxel data (for example with ``mapper.tsdf_layer_view()``).

We then loop through each frame in the dataset calling ``process_frame()`` on each sample

.. code-block:: python

    for idx, data in enumerate(dataloader):
        print(f'Integrating frame: {idx}')
        process_frame(mapper, data, feature_extractor, visualizer)

In ``process_frame()`` we add depth and color frames to the reconstruction

.. code-block:: python

    mapper.add_depth_frame(depth, pose, intrinsics)
    mapper.add_color_frame(rgba, pose, intrinsics)

Periodically we update the mesh and visualize it:

.. code-block:: python

    mapper.update_color_mesh()
    visualizer.visualize(mapper=mapper)

The mapper can be queried for map data for use in downstream applications.
See, for example, the ESDF query example :doc:`ESDF Example <torch_examples_esdf>`.
