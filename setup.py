from __future__ import annotations

import os
import sys
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def discover_pybind_include() -> str | None:
    """Find standalone pybind11 or PyTorch's bundled compatible headers."""
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        pass

    try:
        import torch
        candidate = Path(torch.__file__).resolve().parent / "include"
        if (candidate / "pybind11" / "pybind11.h").exists():
            return str(candidate)
    except Exception:
        pass
    return None


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


build_native = os.environ.get("PUFFERFORGE_BUILD_NATIVE", "1") != "0"
pybind_include = discover_pybind_include() if build_native else None

if build_native and pybind_include is None:
    raise RuntimeError(
        "PufferForge native build requested but pybind11 headers were not found. "
        "Install pybind11, install PyTorch, or set PUFFERFORGE_BUILD_NATIVE=0 "
        "for the NumPy/PyTorch fallback runtime."
    )

include_dirs = ["cpp/include"]
if pybind_include:
    include_dirs.append(pybind_include)

ext_modules = []
if build_native:
    ext_modules.append(
        Extension(
            "pufferforge._core",
            ["cpp/src/bindings.cpp"],
            include_dirs=include_dirs,
            language="c++",
        )
    )

setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExt})
