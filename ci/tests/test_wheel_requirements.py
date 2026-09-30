"""Test ABI-specific wheel dependencies without importing PyTorch or using a GPU."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


class WheelRequirementsTest(unittest.TestCase):
    """Use the real packaging helper with recorded build-version inputs."""

    def setUp(self) -> None:
        """Load setup helpers without requiring a packaging backend for CPU tests."""
        spec = importlib.util.spec_from_file_location('nvblox_package_setup',
                                                      ROOT / 'nvblox_torch/setup.py')
        assert spec is not None and spec.loader is not None
        self.module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {'setuptools': SimpleNamespace(setup=mock.Mock())}):
            spec.loader.exec_module(self.module)
        # unittest.addCleanup guarantees cleanup even when setUp fails.
        self.directory = tempfile.TemporaryDirectory(    # pylint: disable=consider-using-with
            dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.module.__file__ = str(root / 'setup.py')
        (root / 'requirements.txt').write_text('# Shared requirements\nnumpy>=1.24,<1.27\n',
                                               encoding='utf-8')
        self.version_file = root / 'nvblox_torch/lib/nvblox_torch/cpp/torch_version.txt'
        self.version_file.parent.mkdir(parents=True)

    def test_wheel_pins_the_torch_release_used_to_build_it(self) -> None:
        """A Torch14 binary must not advertise compatibility with a Torch9 install."""
        self.version_file.write_text('2.14.1\n', encoding='utf-8')
        self.assertEqual(self.module.get_install_requires(), ['numpy>=1.24,<1.27', 'torch==2.14.1'])

    def test_empty_build_version_is_rejected(self) -> None:
        """Never emit an unpinned requirement when a build record is invalid."""
        self.version_file.write_text('\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'Torch build version'):
            self.module.get_install_requires()

    def test_legacy_and_jetson_versions_are_preserved(self) -> None:
        """Do not raise a CUDA12 or Jetson wheel's Torch dependency to Torch14."""
        for version in ('2.9.1', '2.3.0', '2.3.0a0'):
            with self.subTest(version=version):
                self.version_file.write_text(version + '\n', encoding='utf-8')
                self.assertIn(f'torch=={version}', self.module.get_install_requires())

    def test_missing_native_build_record_is_rejected(self) -> None:
        """Packaging cannot guess an ABI before the native library was built."""
        with self.assertRaises(FileNotFoundError):
            self.module.get_install_requires()


if __name__ == '__main__':
    unittest.main()
