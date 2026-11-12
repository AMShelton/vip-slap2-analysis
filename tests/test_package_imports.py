"""Every module in the package must import (optional heavy deps may be absent in CI)."""
import importlib
import pkgutil

import pytest

import vip_slap2_analysis

OPTIONAL_DEPS = ("sklearn", "cv2", "imageio", "tifffile", "skimage", "statsmodels")


def _modules():
    return [m.name for m in pkgutil.walk_packages(vip_slap2_analysis.__path__, "vip_slap2_analysis.")]


@pytest.mark.parametrize("name", _modules())
def test_module_imports(name):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = str(exc).split("'")[1] if "'" in str(exc) else ""
        if missing.split(".")[0] in OPTIONAL_DEPS:
            pytest.skip(f"optional dependency missing: {missing}")
        raise
