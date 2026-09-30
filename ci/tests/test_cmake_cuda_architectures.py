"""Exercise the real CMake architecture conversion without CUDA or a GPU."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('cmake'), 'CMake is required for script tests')
class CudaArchitectureTest(unittest.TestCase):
    """Keep Torch architecture flags aligned with the requested CMake targets."""

    def check_architectures(self, architectures: str, expected: str) -> None:
        """Run the shipped CMake module and assert its resulting Torch list."""
        module = ROOT / 'cmake/cuda/setup_compute_capability.cmake'
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
            script = Path(directory) / 'check.cmake'
            script.write_text(
                'cmake_minimum_required(VERSION 3.22.1)\n'
                'set(CMAKE_CUDA_ARCHITECTURES_SET_EXTERNALLY TRUE)\n'
                f'set(CMAKE_CUDA_ARCHITECTURES "{architectures}")\n'
                f'include("{module.as_posix()}")\n'
                f'if(NOT TORCH_CUDA_ARCH_LIST STREQUAL "{expected}")\n'
                '  message(FATAL_ERROR "Wrong Torch architectures: ${TORCH_CUDA_ARCH_LIST}")\n'
                'endif()\n',
                encoding='utf-8')
            result = subprocess.run(['cmake', '-P', str(script)],
                                    capture_output=True,
                                    text=True,
                                    check=False,
                                    timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_two_digit_architectures(self) -> None:
        """Preserve the older architecture mappings."""
        self.check_architectures('75;80;86;89;90', '7.5;8.0;8.6;8.9;9.0')

    def test_blackwell_architectures(self) -> None:
        """Do not replace Blackwell targets with Hopper 9.0."""
        self.check_architectures('100;120;121', '10.0;12.0;12.1')


if __name__ == '__main__':
    unittest.main()
