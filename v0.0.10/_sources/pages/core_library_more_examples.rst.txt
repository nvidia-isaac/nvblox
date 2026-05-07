More Examples
=============

.. note::
    This example assumes that ``nvblox`` has been built from source — see :ref:`nvblox_torch_source_installation`.

If you would like to run nvblox on public datasets, we include some executables for
fusing `3DMatch <https://3dmatch.cs.princeton.edu/>`_,
`Replica <https://github.com/facebookresearch/Replica-Dataset>`_,
and `Redwood <http://redwood-data.org/indoor_lidar_rgbd/index.html>`_ datasets.

The executables are run by pointing the respective binary to a folder
containing the dataset. We give details for each dataset below.


Replica
-------

Instructions to run Replica are given in :doc:`core_library_run_an_example`.

3DMatch
-------

In this example we fuse data from the `3DMatch dataset <https://3dmatch.cs.princeton.edu/>`_.

:download_sun3d_test_dataset:

From the nvblox base folder run

.. code-block:: bash

    build/executables/fuse_3dmatch sun3d-mit_76_studyroom-76-1studyroom2

A visualizer window will open and you should see the mesh being incrementally reconstructed in real time. See :doc:`core_library_run_an_example` for more details.
When reconstruction is complete, the mesh will look similar to the image below:

.. image:: ../images/reconstruction_in_docker_trim.png


Redwood
-------

The Redwood RGB-D datasets are available `here <http://redwood-data.org/indoor_lidar_rgbd/download.html>`_.

Download the "RGB-D sequence" and "Our camera poses" at the link above.

Extract the data into a common folder. For example for the apartment sequence
the resultant folder structure looks like (here we assume datasets are stored in ``~/datasets``):

.. code-block:: bash

    ~/datasets/redwood/apartment
    ~/datasets/redwood/apartment/pose_apartment/...
    ~/datasets/redwood/apartment/rgbd_apartment/...

From the nvblox base folder run:

.. code-block:: bash

    build/executables/fuse_redwood ~/datasets/redwood/apartment --voxel_size=0.03

Note this dataset is large (~30000 images) so the reconstruction can take a couple of minutes.

A visualizer window will open and you should see the mesh being incrementally reconstructed in real time. See :doc:`core_library_run_an_example` for more details.
When reconstruction is complete, the mesh will look similar to the image below:

.. image:: ../images/redwood_apartment.png
   :align: center
   :width: 600px
