from __future__ import annotations

import os
import sys
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    import pybind11
    pybind_include = pybind11.get_include()
except ImportError:
    pybind_include = None

ROOT = Path(__file__).parent

class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/O2", "/std:c++20"],
        "unix": ["-O3", "-std=c++20", "-fvisibility=hidden"],
    }

    def build_extensions(self):
        compiler = self.compiler.compiler_type
        opts = list(self.c_opts.get(compiler, []))
        if compiler == "unix":
            if sys.platform == "darwin":
                opts += ["-stdlib=libc++"]
            elif os.environ.get("PUFFERFORGE_OPENMP", "1") == "1":
                opts += ["-fopenmp"]
                for ext in self.extensions:
                    ext.extra_link_args = ["-fopenmp"]
        for ext in self.extensions:
            ext.extra_compile_args = opts
        super().build_extensions()

include_dirs = [str(ROOT / "cpp" / "include")]
if pybind_include:
    include_dirs.append(pybind_include)

ext_modules = [
    Extension(
        "pufferforge._core",
        [str(ROOT / "cpp" / "src" / "bindings.cpp")],
        include_dirs=include_dirs,
        language="c++",
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExt})
