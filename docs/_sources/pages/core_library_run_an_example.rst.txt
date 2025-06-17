Run an Example
==============

In this example we fuse data from the `3DMatch dataset <https://3dmatch.cs.princeton.edu/>`_.

First let's grab the dataset.
Here I'm downloading it to the dataset folder in the docker, which is at ``/datasets/3dmatch``.

.. code-block:: bash

    wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-mit_76_studyroom-76-1studyroom2.zip -P /datasets/3dmatch
    unzip /datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2.zip -d /datasets/3dmatch

Next we run the ``fuse_3dmatch`` binary.
From the nvblox base folder run

.. code-block:: bash

    cd build/nvblox/executables
    ./fuse_3dmatch /datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2/ \
      mesh.ply

This produces an output mesh file ``mesh.ply`` in the current directory.

We can view the mesh using the Open3D viewer.
First we need to install Open3D.

.. code-block:: bash

    sudo apt-get install libglib2.0-0 libgl1
    pip3 install open3d

Then visualize the mesh

.. code-block:: bash

    open3d draw mesh.ply

You should see a mesh of a room:

.. image:: ../images/reconstruction_in_docker_trim.png

More examples of running nvblox on datasets are given in :doc:`core_library_more_examples`.
