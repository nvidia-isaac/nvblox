This directory contains symlinks to the shared libraries shipped with the PyTorch wrapper.
It mirrors the build directory layout so relative runtime linking works.

`nvblox_torch/cpp/torch_version.txt` links to the build-version record written
only after `py_nvblox` links successfully. Packaging uses this record to pin
the Torch release without importing Torch inside the isolated build backend.
