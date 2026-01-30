Contributing to nvblox
======================

Thank you for your interest in contributing to ``nvblox``! This guide explains our expectations for code style and quality,
and the checks you should run locally before opening a pull request.

Quick checklist (before you open a PR)
--------------------------------------

- **Keep PRs focused** and reasonably small.
- **Document the code** with comments and docstrings. Legibility is preferred over conciseness.
- **Add tests** for new functionality.
- **Pre-commit:** Run all formatters/linters locally with :ref:`pre-commit <precommit>`.
- Build and run tests:
   - C++/CUDA tests with CMake/CTest.
   - Python/torch tests with ``pytest``.
- **Sign your work**: See below on how to :ref:`sign-off your work <sign_off_your_work>`.
- **Docs**: If needed, build the docs (Sphinx) and fix warnings.
- **Update the Changelog**: for user-facing changes.

.. _precommit:

--------------------------------------
Formatting, linting, and type checking
--------------------------------------

We use ``pre-commit`` to run all formatters and linters consistently. Install and run it locally before committing.

.. code-block:: bash

   # From the repo root:
   pip install --upgrade pip pre-commit
   pre-commit install
   # Format and lint everything
   pre-commit run --all-files


----------------------------------
C++ rules and policies
----------------------------------

Repository layout
~~~~~~~~~~~~~~~~~~

- The C++ code is located under ``nvblox/`` and is organized into modules for which we separate public/internal/implementation/cuda headers. Below is an example of the organization of the ``integrators`` module.

.. code-block:: bash

   nvblox/
   ├── include/
   │   ├── nvblox/
   │   │   ├── integrators/
   │   │   │   ├── tsdf_integrator.h                   # Public header
   │   │   │   ├── internal/
   │   │   │   │   ├── tsdf_integrator_params.h        # Internal header
   │   │   │   │   ├── impl/
   │   │   │   │   │   ├── tsdf_integrator_impl.h      # Implementation header
   |   │   │   │   ├── cuda/
   |   │   │   │   │   ├── tsdf_integrator.cuh         # CUDA header
   |   │   │   │   |   ├── impl/
   |   │   │   │   |   │   ├── tsdf_integrator_impl.cu # CUDA implementation header
   |   src/
   |   ├── integrators/
   |   │   ├── tsdf_integrator.cpp  # Source file
   |   │   ├── tsdf_integrator.cu   # CUDA source file
   |   tests/
   |   ├── test_tsdf_integrator.cpp # Unit tests

- **Public:**
    - All public headers should be included from ``nvblox/include/nvblox/nvblox.h`` (pre-commit enforces this).
    - Public headers should not include any cuda headers (indirect or direct). This is to ensure that a compiled library can be included in projects not using the ``nvcc`` compiler.
- **Internal:** Headers that are not part of the public API should be located under ``internal/`` directories.
- **Implementation:** Any definitions that must reside in a header file (such as templated code) should be placed in the ``impl/`` directories.
- **CUDA:** All headers that requires the ``nvcc`` compiler (i.e. CUDA code) should be located under ``cuda/`` directories.
- **Source:** Prefer placing implementations in the source files whenever possible.

C++/CUDA style and conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We generally follow the `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_.
In addition, we exhibit a few conventions, in particular for CUDA code. When in doubt, match the prevailing patterns in the files you
touch.

- We use ``#pragma once`` in headers.
- Use ``...Kernel`` suffix for ``__global__`` functions, e.g., ``insertAllKernel``
- Prefer to use ``unified_ptr`` or ``unified_vector`` for memory management instead of raw pointers.
- Prefer passing ``ImageView`` to kernels rather than raw pointer to image data.
- All CUDA operations should be asynchronous and use the ``CudaStream`` class for synchronization.
- All native cuda operations should be wrapped in the ``checkCudaErrors`` macro.
- In particular, all kernel launches should be followed by ``checkCudaErrors(cudaPeekAtLastError());`` to check for launch-time errors.



Exceptions, logging, and errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- We use ``glog`` macros like ``CHECK`` / ``CHECK_EQ`` / ``VLOG`` for assertions and diagnostics. Prefer these over exceptions and asserts.
- For device code, use ``NVBLOX_CHECK`` / ``NVBLOX_DCHECK`` for assertions and diagnostics.
- These macros should be used generously to ensure that inputs and results are as expected.


-------------------------
Python rules and policies
-------------------------

- The ``nvblox_torch`` Python wrapper is organized as follows:

.. code-block:: bash

  nvblox_torch/
  ├── cpp/            # C++ code for wrapping
  |   ├── include/
  |   ├── tests/
  |   ├── src/
  |   |   ├── py_nvblox.cu # Wrapper definitions.
  |   |   |   ...
  ├── nvblox_torch/   # Python package base dir.
  |   ├── mapper.py   # Main mapper class.
  |   |   ...
  |   ├── lib/        # Symlinks to the .so files we are shipping with the wrapper.
  |   ├── datasets/   # Datasets for testing and evaluation.
  |   ├── examples/   # Example scripts
  |   ├── tests/      # Python unit tests.

Coding style and docs
~~~~~~~~~~~~~~~~~~~~~

- Use type hints throughout (including return types). Prefer ``Optional[T]``/``T | None`` and modern unions.
- Docstrings follow Google style (``pydocstyle`` with ``convention = google``). Include shapes, dtypes, and device expectations in Args/Returns where helpful.
- Prefer ``torch`` over ``numpy`` whenever possible.
- Allocate new tensors on ``'cuda'`` by default when working with GPU-backed data.
- Use ``assert`` for programmer errors and precondition checks in the Python wrappers.
- Raise ``ValueError`` for runtime issues (e.g., failed queries) and ``NotImplementedError`` for unimplemented features.

Interfacing with C++
~~~~~~~~~~~~~~~~~~~~
- Access C++ bindings via ``nvblox_torch.lib.utils.get_nvblox_torch_class(...)``. Do not import private C++ symbols directly.
- Provide high-level Python wrappers (e.g., ``Mapper``, ``Layer`` variants) and avoid exposing low-level C++ details in the public API.
- Many accessors return zero-copy tensor views into C++ memory; document that modifying the underlying layer invalidates previous views.


.. _sign_off_your_work:

-----------------
Signing Your Work
-----------------

* We require that all contributors "sign-off" on their commits according to the DCO below. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not signed-off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

.. code-block:: bash

  $ git commit -s -m "Add cool feature."

This will append the following to your commit message:

.. code-block:: bash

  Signed-off-by: Your Name <your@email.com>


.. admonition:: Developer's Certificate of Origin (DCO)

    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
