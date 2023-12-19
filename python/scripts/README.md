## Nvblox python scripts

This package contains useful python scripts for working with nvblox.

Install the python package with:
```bash
pip3 install -e <NVBLOX_ROOT>/python/common/
pip3 install -e <NVBLOX_ROOT>/python/scripts/
```

After that, the modules can be run system-wide with:
```bash
python -m <module_name> <module_arguments>
```

The following modules are available:

| Module Name                         | Description                                                                        |
|-------------------------------------|------------------------------------------------------------------------------------|
| `nvblox_visualize_mesh`             | `Visualize a mesh stored as ply in Open3D.`                                        |
| `nvblox_visualize_pointcloud`       | `Visualize a pointcloud stored as ply in Open3D.`                                  |
| `nvblox_visualize_voxel_grid`       | `Visualize an esdf/occupancy voxel grid stored as ply or npz in Open3D.`           |
| `nvblox_convert_mesh_to_voxel_grid` | `Convert a ply mesh to an esdf/occupancy voxel grid and store it as ply/npz file.` |

To check the usage of each module run:
```bash
python -m <module_name> -h
```
