"""Offline regression checks for the Bazel dependency configuration."""

import ast
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import List
import unittest

ROOT = Path(__file__).resolve().parents[2]


def build_flags(config: str = '') -> List[str]:
    """Read default build flags followed by a named config's options."""
    lines = [
        shlex.split(line, comments=True)
        for line in (ROOT / '.bazelrc').read_text(encoding='utf-8').splitlines()
    ]
    sections = ['build'] + ([f'build:{config}'] if config else [])
    return [
        option for section in sections for line in lines if line and line[0] == section
        for option in line[1:]
    ]


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

    def test_cxx20_requires_an_explicit_config(self) -> None:
        """Keep the default compatible and select C++20 for host and device together."""
        for config, standard in (('', '17'), ('cxx20', '20')):
            with self.subTest(config=config or 'default'):
                flags = build_flags(config)
                host_standards = [flag for flag in flags if flag.startswith('--cxxopt=-std=')]
                self.assertTrue(host_standards)
                self.assertEqual(host_standards[-1], f'--cxxopt=-std=c++{standard}')
                cuda_standards = [
                    flag for flag in flags if flag.startswith('--@rules_cuda//cuda:copts=-std=')
                ]
                # Default CUDA uses the supported GCC 11-13 host compiler's
                # C++17 default, as it did before the C++20 source profile.
                expected_cuda = ['--@rules_cuda//cuda:copts=-std=c++20'] if config else []
                self.assertEqual(cuda_standards, expected_cuda)

    def test_toolchain_does_not_override_selected_standard(self) -> None:
        """A toolchain -std can silently override the user's .bazelrc selection."""
        toolchain = ROOT / 'bazel/toolchains/cc_toolchain_config.bzl'
        tree = ast.parse(toolchain.read_text(encoding='utf-8'))
        standards = [
            node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
            and isinstance(node.value, str) and node.value.startswith('-std=')
        ]
        self.assertEqual(standards, [])

    @unittest.skipUnless(shutil.which('c++'), 'A host C++ compiler is required')
    def test_host_compiler_uses_selected_standard(self) -> None:
        """Compile with the actual rc flags to catch incorrect option ordering."""
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
            output = str(Path(directory) / 'probe.o')
            for config, expected in (('', '201703L'), ('cxx20', '202002L')):
                with self.subTest(config=config or 'default'):
                    flags = [
                        flag.split('=', 1)[1] for flag in build_flags(config)
                        if flag.startswith('--cxxopt=')
                    ]
                    source = f'static_assert(__cplusplus == {expected});\n'
                    result = subprocess.run(['c++', *flags, '-x', 'c++', '-c', '-', '-o', output],
                                            input=source,
                                            capture_output=True,
                                            text=True,
                                            check=False,
                                            timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
