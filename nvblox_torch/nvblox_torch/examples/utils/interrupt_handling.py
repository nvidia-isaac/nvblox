#
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import sys
from typing import Any, Callable


def run_with_graceful_interrupt(fn: Callable[..., int], *args: Any, **kwargs: Any) -> int:
    """Run ``fn(*args, **kwargs)`` and convert KeyboardInterrupt into a clean exit.

    Returns 130 (POSIX SIGINT exit code) on Ctrl+C with a single-line message
    on stderr, instead of a Python traceback. Otherwise returns whatever
    ``fn`` returns.
    """
    try:
        return fn(*args, **kwargs)
    except KeyboardInterrupt:
        print('\nInterrupted by user, exiting.', file=sys.stderr)
        return 130
