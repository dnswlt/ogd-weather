import pytest

from . import prefix


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("Zürich, Weiß, Łódź — São Tomé", "Zurich, Weiss, Lodz - Sao Tome"),
        ("naïve café", "naive cafe"),
        ("Ærøskøbing & œuf", "Aeroskobing & oeuf"),
        (
            "Iceland: Þingvellir, Eyjafjallajökull",
            "Iceland: Thingvellir, Eyjafjallajokull",
        ),
        (
            "Istanbul – “İzmir”",
            'Istanbul - "Izmir"',
        ),  # İ→i, curly quotes/dash normalized
        ("A B", "A B"),  # NBSP -> space
        ("ASCII 123", "ASCII 123"),  # unchanged ASCII
        ("A🍕B Москва", "A?B ??????"),  # non-ASCII that isn't diacritics -> '?'
        ("e\u0301clair", "eclair"),  # combining accent stripped
    ],
)
def test_ascii_fold_examples(inp, expected):
    assert prefix._ascii_fold(inp) == expected


def test_ascii_fold_specific_letters():
    # Spot-check explicit replacements that don't come from simple diacritic stripping
    pairs = {
        "ß": "ss",
        "ẞ": "SS",
        "æ": "ae",
        "Æ": "Ae",
        "œ": "oe",
        "Œ": "Oe",
        "ø": "o",
        "Łódź": "Lodz",
        "ğ": "g",
        "İzmir": "Izmir",
        "ı": "i",
    }
    for src, want in pairs.items():
        assert prefix._ascii_fold(src) == want


def test_prefix_lookup():
    lookup = prefix.PrefixLookup(["Zürich", "Straßenbahn", "Bå", "zuffenhausen"])

    assert not lookup.find_prefix("ich")
    assert lookup.find_prefix("zur") == ["Zürich"]
    assert lookup.find_prefix("ba") == ["Bå"]
    assert lookup.find_prefix("zu") == ["zuffenhausen", "Zürich"]
    assert lookup.find_prefix("zu", limit=1) == ["zuffenhausen"]
