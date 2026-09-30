"""Offline regression checks for the Bazel dependency configuration."""

import ast
from pathlib import Path
import shlex
import unittest

ROOT = Path(__file__).resolve().parents[2]


class BazelBuildConfigurationTest(unittest.TestCase):
    """Guard configuration regressions without installing Bazel or using a GPU."""

    def test_stdgpu_keeps_headers_namespaced(self) -> None:
        """Do not expose stdgpu/limits.h as the system's <limits.h>."""
        build = ROOT / 'bazel/third_party/stdgpu/stdgpu.BUILD'
        tree = ast.parse(build.read_text(encoding='utf-8'))
        rules = [
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == 'cuda_library' and any(
                keyword.arg == 'name' and ast.literal_eval(keyword.value) == 'stdgpu'
                for keyword in node.keywords)
        ]
        self.assertEqual(len(rules), 1)
        includes = next(
            ast.literal_eval(keyword.value) for keyword in rules[0].keywords
            if keyword.arg == 'includes')
        self.assertIn('src', includes)
        self.assertNotIn('src/stdgpu', includes,
                         'This shadows limits.h and breaks C++20 <semaphore>')

    def test_installer_uses_official_release_key(self) -> None:
        """The old bazel.build signing-key endpoint returns HTTP 404."""
        installer = ROOT / 'ci/install_bazel.sh'
        tokens = shlex.split(installer.read_text(encoding='utf-8'), comments=True)
        key_urls = [
            token for token in tokens
            if token.startswith('https://') and token.endswith('/bazel-release.pub.gpg')
        ]
        self.assertEqual(key_urls, ['https://releases.bazel.build/bazel-release.pub.gpg'])


if __name__ == '__main__':
    unittest.main()
