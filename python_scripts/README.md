## Nvblox python scripts

This package contains useful python scripts for working with nvblox.

Install the python package with:
```bash
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_nvblox/nvblox/python_scripts/
pip3 install -e .
```

After that, the modules can be run system-wide with:
```bash
python -m <module_name> <module_arguments>
```

The following modules are available:

| Module Name                        | Arguments  | Description                          |
|------------------------------------|------------|--------------------------------------|
| `nvblox_visualize_mesh`            | `ply_path` | `Path to the ply file to visualize.` |
