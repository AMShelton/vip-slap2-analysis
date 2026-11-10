import re

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from vip_slap2_analysis.plotting import style


def test_categorical_palette_is_valid_hex():
    for hexcode in list(style.CATEGORICAL.values()) + list(style.PASTEL_7):
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", hexcode)


def test_categorical_colors_cycles():
    pool = len(style.CATEGORICAL) + len(style.PASTEL_7)
    cols = style.categorical_colors(pool + 2)
    assert len(cols) == pool + 2
    assert cols[0] == cols[pool] and cols[1] == cols[pool + 1]
    assert style.categorical_colors(0) == []
    with pytest.raises(ValueError):
        style.categorical_colors(-1)


def test_continuous_colors_length():
    assert len(style.continuous_colors(5)) == 5
