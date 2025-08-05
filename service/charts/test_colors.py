import pytest

from .colors import Custom, Tab20, COLORS_TABLEAU20


def test_full_color_start_rotation():
    pal = Tab20("SteelBlue")
    colors = pal._get_colors()
    # First color should match starting full color
    assert colors[0] == COLORS_TABLEAU20["SteelBlue"]
    # The paired light color should be at the same relative position in the secondary list
    assert COLORS_TABLEAU20["SkyBlue"] in pal._secondary
    # Number of colors should match palette size
    assert len(colors) == len(COLORS_TABLEAU20)


def test_light_color_start_rotation():
    pal = Tab20("SkyBlue")
    colors = pal._get_colors()
    # First color should match starting light color
    assert colors[0] == COLORS_TABLEAU20["SkyBlue"]
    # The paired full color should be in the secondary list
    assert COLORS_TABLEAU20["SteelBlue"] in pal._secondary
    assert len(colors) == len(COLORS_TABLEAU20)


def test_invert_swaps_primary_and_secondary():
    pal = Tab20("SteelBlue")
    inv = pal.invert()
    assert inv._primary == pal._secondary
    assert inv._secondary == pal._primary
    # Colors after inversion should still be the same set
    assert sorted(inv._get_colors()) == sorted(pal._get_colors())


def test_first_n_cycles_colors():
    pal = Tab20("SteelBlue")
    n = len(COLORS_TABLEAU20) + 5  # force wraparound
    colors = pal.first_n(n)
    assert len(colors) == n
    # Ensure the sequence repeats after the palette length
    assert colors[0] == colors[len(COLORS_TABLEAU20)]


def test_invalid_color_raises():
    with pytest.raises(ValueError):
        Tab20("NotAColor")


def test_custom_single_color():
    pal = Custom("#123456")
    colors = pal._get_colors()
    assert colors == ["#123456"]
    assert pal.first_n(3) == ["#123456", "#123456", "#123456"]


def test_custom_multiple_colors():
    pal = Custom(["#111111", "#222222"])
    colors = pal._get_colors()
    assert colors == ["#111111", "#222222"]
    # first_n should cycle
    assert pal.first_n(5) == ["#111111", "#222222", "#111111", "#222222", "#111111"]


def test_custom_tab20_constructor():
    pal = Custom.tab20("SteelBlue", "Tangerine")
    colors = pal._get_colors()
    assert colors == [COLORS_TABLEAU20["SteelBlue"], COLORS_TABLEAU20["Tangerine"]]


def test_custom_invert():
    pal = Custom(["#111111", "#222222", "#333333"])
    inv = pal.invert()
    assert inv._get_colors() == ["#333333", "#222222", "#111111"]
    # original palette unchanged
    assert pal._get_colors() == ["#111111", "#222222", "#333333"]


def test_custom_str_representation():
    pal = Custom(["#111111", "#222222"])
    s = str(pal)
    assert "Custom" in s
    assert "#111111" in s and "#222222" in s


def test_custom_tab20_invalid_color_raises():
    with pytest.raises(KeyError):
        Custom.tab20("NotAColor")
