from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pure_python_editable_install_metadata_path_is_supported() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "PUFFERFORGE_BUILD_NATIVE" in setup_text
    assert "discover_pybind_include" in setup_text
    assert 'Path(torch.__file__).resolve().parent / "include"' in setup_text
