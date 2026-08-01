from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import ClassVar

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
    except (ImportError, OSError):
        return None
    return None


class BuildExt(build_ext):
    c_opts: ClassVar[dict[str, list[str]]] = {
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


native_mode = os.environ.get("PUFFERFORGE_BUILD_NATIVE", "auto").strip().lower()
if native_mode not in {"0", "1", "auto"}:
    raise RuntimeError("PUFFERFORGE_BUILD_NATIVE must be '0', '1', or 'auto'")
pybind_include = discover_pybind_include() if native_mode != "0" else None
build_native = pybind_include is not None

if native_mode == "1" and not build_native:
    raise RuntimeError(
        "PufferForge native build was explicitly requested but pybind11 headers "
        "were not found. Install pybind11 or PyTorch, or set "
        "PUFFERFORGE_BUILD_NATIVE=0 for the fallback runtime."
    )
if native_mode == "auto" and not build_native:
    warnings.warn(
        "pybind11 headers were not found; building the pure-Python PufferForge "
        "runtime. Set PUFFERFORGE_BUILD_NATIVE=1 to require native support.",
        stacklevel=1,
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
