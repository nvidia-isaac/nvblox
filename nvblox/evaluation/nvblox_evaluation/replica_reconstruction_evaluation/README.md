# Replica Dataset Reconstruction / Integration Test

We use the replica dataset to test the reconstruction quality nvblox core library.

The test the reconstruction pipeline and test against ground-truth is packaged in a two python scripts, one for evaluating the surface (mesh) reconstruction accuracy, and one to evaluate the reconstructed ESDF.

To obtain the dataset, navigate to you dataset directory (we'll call this `DATASET_DIR`), and run:
```
cd DATASET_DIR
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip replica.zip
```

From the `nvblox` folder navigate to the `evaluation` folder. Run one of the two evaluation scripts (`replica_surface_evaluation.py` and `replica_esdf_evaluation.py`) on the particular replica dataset you want. In the example below we run the surface evaluation on the `office0` dataset:
```
cd nvblox/evaluation/replica
./replica_surface_evaluation.py DATASET_DIR/office0_mesh.ply
```
The test creates the reconstruction, and then an error mesh (which will be displayed), as well as error statistics which will be saved in a subfolder of the script directory called `output/office0`.

Note that the above assumes that you have nvblox built in the usual place: `nvblox/build`. If this is not the case you can specify the location of the `fuse_replica` executable.

In fact there are a few options. For the surface evaluation:

```
usage: replica_surface_evaluation.py [-h] [--output_root_path OUTPUT_ROOT_PATH]
                                     [--dont_visualize_error_mesh] [--do_coverage_visualization]
                                     [--fuse_replica_binary_path FUSE_REPLICA_BINARY_PATH]
                                     groundtruth_mesh_path [reconstructed_mesh_path]

Reconstruct a mesh from the replica dataset and test it against ground-truth geometry.

positional arguments:
  groundtruth_mesh_path
                        Path to the groundtruth mesh.
  reconstructed_mesh_path
                        Path to the mesh to evaluate.

optional arguments:
  -h, --help            show this help message and exit
  --output_root_path OUTPUT_ROOT_PATH
                        Path to the directory in which to save results.
  --dont_visualize_error_mesh
                        Flag indicating if we should visualize the error mesh.
  --do_coverage_visualization
                        Flag indicating if we should display the coverage mesh.
  --fuse_replica_binary_path FUSE_REPLICA_BINARY_PATH
                        Path to the fuse_replica binary. If not passed we search the standard build
                        folder location.
```

and for the ESDF evaluation:
```
usage: replica_esdf_evaluation.py [-h] [--output_root_path OUTPUT_ROOT_PATH] [--dont_visualize_slice]
                                  [--dont_animate_slice]
                                  [--fuse_replica_binary_path FUSE_REPLICA_BINARY_PATH]
                                  groundtruth_mesh_path [reconstructed_esdf_path]
                                  [reconstructed_mesh_path]

Evaluates a reconstructed ESDF.

positional arguments:
  groundtruth_mesh_path
                        Path to the groundtruth mesh.
  reconstructed_esdf_path
                        Path to the esdf to evaluate.
  reconstructed_mesh_path
                        Path to the reconstructed mesh (for visualization).

optional arguments:
  -h, --help            show this help message and exit
  --output_root_path OUTPUT_ROOT_PATH
                        Path to the directory in which to save results.
  --dont_visualize_slice
                        Flag indicating if we should visualize an ESDF slice in 3D.
  --dont_animate_slice  Flag indicating if we should animate an ESDF slice.
  --fuse_replica_binary_path FUSE_REPLICA_BINARY_PATH
                        Path to the fuse_replica binary. If not passed we search the standard build
                        folder location.
```
