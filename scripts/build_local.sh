#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CXX="${CXX:-c++}"
EXT_SUFFIX="$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
PY_INCLUDES="$($PYTHON-config --includes 2>/dev/null || $PYTHON -c 'import sysconfig; print("-I" + sysconfig.get_path("include"))')"

if $PYTHON -c 'import pybind11' >/dev/null 2>&1; then
  PYBIND_INCLUDE="$($PYTHON -c 'import pybind11; print(pybind11.get_include())')"
else
  PYBIND_INCLUDE="$($PYTHON - <<'PY'
import pathlib, torch
print(pathlib.Path(torch.__file__).parent / "include")
PY
)"
fi

mkdir -p "$ROOT/python/pufferforge"
OPENMP_FLAGS=()
if [[ "${PUFFERFORGE_OPENMP:-1}" == "1" ]] && [[ "$(uname -s)" != "Darwin" ]]; then
  OPENMP_FLAGS=(-fopenmp)
fi

set -x
$CXX -O3 -std=c++20 -shared -fPIC -fvisibility=hidden \
  $PY_INCLUDES \
  -I"$PYBIND_INCLUDE" \
  -I"$ROOT/cpp/include" \
  "${OPENMP_FLAGS[@]}" \
  "$ROOT/cpp/src/bindings.cpp" \
  -o "$ROOT/python/pufferforge/_core$EXT_SUFFIX"
