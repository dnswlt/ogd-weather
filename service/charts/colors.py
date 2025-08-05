import altair as alt
from itertools import cycle, islice

import pandas as pd


# tableau20 colors from
# https://vega.github.io/vega/docs/schemes/
# (names are AI-generated)
COLORS_TABLEAU20 = {
    # Blue Pair
    "SteelBlue": "#4c78a8",
    "SkyBlue": "#9ecae9",
    # Orange Pair
    "Tangerine": "#f58518",
    "Apricot": "#ffbf79",
    # Green Pair
    "LeafGreen": "#54a24b",
    "PastelGreen": "#88d27a",
    # Yellow/Gold Pair
    "MustardYellow": "#b79a20",
    "PaleGold": "#f2cf5b",
    # Teal Pair
    "Teal": "#439894",
    "Aqua": "#83bcb6",
    # Red Pair
    "CoralRed": "#e45756",
    "SalmonPink": "#ff9d98",
    # Gray Pair
    "WarmGray": "#79706e",
    "AshGray": "#bab0ac",
    # Pink/Rose Pair
    "DustyRose": "#d67195",
    "PastelPink": "#fcbfd2",
    # Purple/Mauve Pair
    "Mauve": "#b279a2",
    "Lavender": "#d6a5c9",
    # Brown Pair
    "CocoaBrown": "#9e765f",
    "Tan": "#d8b5a5",
}


COLORS_COMMON_GRAYS = {
    "White": "#ffffff",
    "VeryLightGray": "#f0f0f0",
    "LightGray": "#d9d9d9",
    "SoftGray": "#bfbfbf",
    "MediumLightGray": "#999999",
    "MediumGray": "#7f7f7f",
    "MediumDarkGray": "#595959",
    "DarkGray": "#404040",
    "VeryDarkGray": "#262626",
    "NearBlack": "#0d0d0d",
    "Black": "#000000",
}


class Palette:
    """Represents a possibly rotated version of a given color palette.

    Useful to create color scales for nominal measurements.
    """

    def _get_colors(self) -> list[str]:
        """Returns a list of the hex colors defined in this palette."""
        raise NotImplemented(f"_get_rotated not implemented by {self.__class__}")

    def first_n(self, n: int) -> list[str]:
        return list(islice(cycle(self._get_colors()), n))

    def scale(self, data: pd.Series):
        domain = list(data.unique())
        range = self.first_n(len(domain))
        return alt.Scale(domain=domain, range=range)

    def invert(self) -> "Tab20":
        raise NotImplemented(f"invert is not implemented for {self.__class__}")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._get_colors()})"


class Custom(Palette):
    """A custom color palette that uses the given hex color(s)."""

    def __init__(self, colors: str | list[str]):
        if isinstance(colors, str):
            colors = [colors]
        self._colors = colors

    @classmethod
    def tab20(cls, color, *args) -> "Custom":
        """Creates a custom palette from Tableau 20 color names."""
        return cls([COLORS_TABLEAU20[c] for c in (color, *args)])

    def _get_colors(self) -> list[str]:
        return self._colors

    def invert(self) -> "Custom":
        obj = Custom.__new__(Custom)  # bypass __init__
        obj._colors = list(reversed(self._colors))
        return obj


class Tab20(Palette):

    def __init__(self, start_color: str):
        primary, secondary = self._from_color_pairs(COLORS_TABLEAU20, start_color)
        self._primary = primary
        self._secondary = secondary

    def _get_colors(self) -> list[str]:
        return self._primary + self._secondary

    def invert(self) -> "Tab20":
        """Returns a copy of self with primary and secondary colors swapped."""
        obj = Tab20.__new__(Tab20)  # bypass __init__
        obj._primary = list(self._secondary)
        obj._secondary = list(self._primary)
        return obj

    def _from_color_pairs(
        self, palette: dict[str, str], start_color: str
    ) -> tuple[list[str], list[str]]:
        """Specialised version to rotate full and light colors in tandem."""
        if start_color not in palette:
            raise ValueError(f"{start_color} not found in palette keys.")

        keys = list(palette.keys())
        # Split into full and light colors.
        full_keys = keys[0::2]
        light_keys = keys[1::2]

        if start_color in full_keys:
            # Full colors first, rotated so start_color is first
            idx = full_keys.index(start_color)
            full_rot = full_keys[idx:] + full_keys[:idx]
            # Light colors rotated by same offset
            light_rot = light_keys[idx:] + light_keys[:idx]
        else:
            # Light colors first, rotated so start_color is first
            idx = light_keys.index(start_color)
            light_rot = light_keys[idx:] + light_keys[:idx]
            # Full colors rotated by same offset
            full_rot = full_keys[idx:] + full_keys[:idx]

        full_colors = [palette[name] for name in full_rot]
        light_colors = [palette[name] for name in light_rot]

        if start_color in light_keys:
            return light_colors, full_colors

        return full_colors, light_colors
