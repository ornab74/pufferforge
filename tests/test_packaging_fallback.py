from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pure_python_editable_install_metadata_path_is_supported() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "PUFFERFORGE_BUILD_NATIVE" in setup_text
    assert "discover_pybind_include" in setup_text
    assert 'Path(torch.__file__).resolve().parent / "include"' in setup_text


def test_pep517_pure_python_wheel_builds(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PUFFERFORGE_BUILD_NATIVE"] = "0"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT),
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(tmp_path),
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wheels = list(tmp_path.glob("pufferforge-*.whl"))
    assert len(wheels) == 1
    assert "py3-none-any" in wheels[0].name
