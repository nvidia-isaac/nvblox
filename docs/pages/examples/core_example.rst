====================
Core Library Example
====================

In this example we fuse data from the `3DMatch dataset <https://3dmatch.cs.princeton.edu/>`_. The commands to run the example are slightly different depending on if you've installed :ref:`natively <Native Installation>` or in a :ref:`docker container <Docker Installation>`.

Core Library Example - Native
=============================

In this example we fuse data from the `3DMatch dataset <https://3dmatch.cs.princeton.edu/>`_. First let's grab the dataset. Here I'm downloading it to my dataset folder ``~/dataset/3dmatch``. ::

    wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets//datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2.zip -P ~/datasets/3dmatch
    unzip ~/datasets/3dmatch//datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2.zip -d ~/datasets/3dmatch

Navigate to and run the ``fuse_3dmatch`` binary. From the nvblox base folder run::

    cd nvblox/build/experiments
    ./fuse_3dmatch ~/datasets/3dmatch//datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2/ --esdf_frame_subsampling 3000 --mesh_output_path mesh.ply

Once it's done we can view the output mesh using the Open3D viewer. ::

    pip3 install open3d
    python3 ../../visualization/visualize_mesh.py mesh.ply

you should see a mesh of a room:

.. _example result:
.. figure:: ../../images/reconstruction_in_docker_trim.png
    :align: center

    The result of running the core library example.




Core Library Example - Docker
=============================

Now let's run the 3DMatch example inside the docker. Note there's some additional complexity in the ``docker run`` command such that we can forward X11 to the host (we're going to be viewing a reconstruction in a GUI). Run the container using::

    xhost local:docker
    docker run -it --net=host --env="DISPLAY" -v $HOME/.Xauthority:/root/.Xauthority:rw -v /tmp/.X11-unix:/tmp/.X11-unix:rw nvblox

Let's download a dataset and run the example::

    apt-get update
    apt-get install unzip
    wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-mit_76_studyroom-76-1studyroom2.zip -P ~/datasets/3dmatch
    unzip ~/datasets/3dmatch//datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2.zip -d ~/datasets/3dmatch
    cd nvblox/nvblox/build/experiments/
    ./fuse_3dmatch ~/datasets/3dmatch//datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2/ --esdf_frame_subsampling 3000 --mesh_output_path mesh.ply

Now let's visualize. From the same experiments folder run::

    apt-get install python3-pip libgl1-mesa-glx
    pip3 install open3d
    python3 ../../visualization/visualize_mesh.py mesh.ply

You should see the :ref:`image above <example result>`.




