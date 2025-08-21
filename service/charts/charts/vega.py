"""Vanilla Vega charts.

For all chart types that cannot be expressed in Altair/Vega.
"""

import numpy as np
import pandas as pd

from service.charts.db import constants as dc

# Also used for HTML exports.
# Keep in sync with versions in the "head.html" template!
VEGA_VERSION = "5.33.0"
VEGA_LITE_VERSION = "5.23.0"
VEGA_EMBED_VERSION = "6.29.0"


def _tooltip(params):
    """Helper to build a Vega tooltip from the given `params` dict."""
    ks = []
    for k, v in params.items():
        ks.append(f"'{k}': {v}")
    return {"signal": f"{{ {', '.join(ks)} }}"}


def annual_windrose_chart(df: pd.DataFrame, year: int):
    title = f"Wind rose • {year} • hourly avg. wind directions and speeds"
    return windrose_chart(df, title=title)


def windrose_chart(
    df: pd.DataFrame,
    step_angle=360 / 32,
    font_size=12,
    title: str = "Untitled chart",
):
    """Returns a Vega wind rose chart for the given wind directions and speeds.

    NOTE: The returned chart is a vanilla Vega chart, NOT a Vega-Lite chart.

    Args:
        df: a DataFrame with columns ["deg", "speed"] of average wind directions
            (in 0..360 degree angles, not radians) and wind speeds (in m/s).
        step_angle: the step size (in degrees) for angular histogram segments.
            Prefer to leave it at its default value; 32 segments are commonly used.
        font_size: the font size used for the compass direction markers (N, E, S, W).
    """
    # Rename DB columns for better code readability.
    df = df.rename(
        columns={
            dc.WIND_DIRECTION_HOURLY_MEAN: "deg",
            dc.WIND_SPEED_HOURLY_MEAN: "speed",
        }
    )

    deg_bins = np.arange(0, 361, step_angle)
    # Wind speed categories: [0-3), [3-6), [6, inf)
    speed_bins = [0, 3, 6, np.inf]
    # the "0 degrees" bucket covers the -5 .. 5 degrees range.
    deg_shifted = (df["deg"] + 5.0) % 360

    df["deg_bin"] = pd.cut(deg_shifted, bins=deg_bins, right=False)
    df["speed_bin"] = pd.cut(
        df["speed"],
        bins=speed_bins,
        right=False,
        labels=["speed_cat1", "speed_cat2", "speed_cat3"],
    )

    # Count valid measurements in each (deg_bin, speed_bin)
    histogram = (
        df.groupby(["deg_bin", "speed_bin"], observed=True)["deg"]
        .count()
        .unstack(level="speed_bin", fill_value=0)
    )
    # Normalize values so they sum to 1.
    histogram = histogram / histogram.to_numpy().sum()

    # Compute radius length of the longest arc segment to define
    # grid circle radii to draw.
    max_radius = histogram.sum(axis=1).max()
    grid_circle_radii = [
        {"value": x} for x in [0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1] if x <= max_radius
    ]

    # Use the lower bound of each (shifted) angular bucket as the wind rose angle.
    histogram = histogram.reset_index(names="deg")
    histogram["deg"] = histogram["deg"].map(lambda d: d.left)

    hist_values = histogram.to_dict(orient="records")

    return {
        "$schema": f"https://vega.github.io/schema/vega/v{VEGA_VERSION}.json",
        "title": {
            "text": title,
            "anchor": "middle",
            "frame": "group",
            "offset": 20,
        },
        "autosize": {"type": "pad", "contains": "padding"},
        "signals": [
            {
                "name": "width",
                "init": "containerSize()[0]",
                "on": [{"events": "window:resize", "update": "containerSize()[0]"}],
            },
            {
                "name": "height",
                "init": "containerSize()[1]",
                "on": [{"events": "window:resize", "update": "containerSize()[1]"}],
            },
            {
                "name": "max_pixel_radius",
                "update": "min(width, height) / 2 * 0.9",
            },
            # Needed for scaling the grid_circles:
            {
                "name": "chart_radius",
                "update": "data('wind_data')[0].max_total_radius",
            },
        ],
        "data": [
            {
                "name": "wind_data",
                "values": hist_values,
                "transform": [
                    {"type": "formula", "as": "r1", "expr": "datum.speed_cat1"},
                    {
                        "type": "formula",
                        "as": "r2",
                        "expr": "datum.speed_cat1 + datum.speed_cat2",
                    },
                    {
                        "type": "formula",
                        "as": "r3",
                        "expr": "datum.speed_cat1 + datum.speed_cat2 + datum.speed_cat3",
                    },
                    {
                        "type": "joinaggregate",
                        "ops": ["max"],
                        "fields": ["r3"],
                        "as": ["max_total_radius"],
                    },
                ],
            },
            {
                "name": "grid_circles",
                "values": grid_circle_radii,
            },
            {
                "name": "radial_lines",
                "transform": [
                    {"type": "sequence", "start": 0, "stop": 360, "step": 360 / 16}
                ],
            },
            {
                "name": "compass_labels",
                "values": [
                    {"angle": 0, "label": "N"},
                    {"angle": 90, "label": "E"},
                    {"angle": 180, "label": "S"},
                    {"angle": 270, "label": "W"},
                ],
            },
            {
                "name": "legend_data",
                "values": [
                    {
                        "label": "Light (< 3 m/s)",
                        "color": "#a3d6d2",
                    },
                    {
                        "label": "Gentle (3 ≤ x < 6 m/s)",
                        "color": "#45a2b9",
                    },
                    {
                        "label": "Stronger (≥ 6 m/s)",
                        "color": "#347da0",
                    },
                ],
            },
        ],
        "scales": [
            {
                "name": "color_scale",
                "type": "ordinal",
                "domain": {"data": "legend_data", "field": "label"},
                "range": {"data": "legend_data", "field": "color"},
            }
        ],
        "marks": [
            {
                "type": "arc",
                "from": {"data": "wind_data"},
                "encode": {
                    "update": {
                        "x": {"signal": "width / 2"},
                        "y": {"signal": "height / 2"},
                        "startAngle": {
                            "signal": f"(datum.deg - {step_angle}/2) * PI / 180 "
                        },
                        "endAngle": {
                            "signal": f"(datum.deg + {step_angle}/2) * PI / 180"
                        },
                        "innerRadius": {
                            "signal": "(datum.r2 / datum.max_total_radius) * max_pixel_radius"
                        },
                        "outerRadius": {
                            "signal": "(datum.r3 / datum.max_total_radius) * max_pixel_radius"
                        },
                        "fill": {"scale": "color_scale", "value": "Stronger (≥ 6 m/s)"},
                        "tooltip": _tooltip(
                            {
                                "Wind direction": "datum.deg + '°'",
                                "Stronger (≥ 6 m/s)": "format(datum.speed_cat3 * 100, '.1f') + '%'",
                            }
                        ),
                    }
                },
            },
            {
                "type": "arc",
                "from": {"data": "wind_data"},
                "encode": {
                    "update": {
                        "x": {"signal": "width / 2"},
                        "y": {"signal": "height / 2"},
                        "startAngle": {
                            "signal": f"(datum.deg - {step_angle}/2) * PI / 180 "
                        },
                        "endAngle": {
                            "signal": f"(datum.deg + {step_angle}/2) * PI / 180"
                        },
                        "innerRadius": {
                            "signal": "(datum.r1 / datum.max_total_radius) * max_pixel_radius"
                        },
                        "outerRadius": {
                            "signal": "(datum.r2 / datum.max_total_radius) * max_pixel_radius"
                        },
                        "fill": {
                            "scale": "color_scale",
                            "value": "Gentle (3 ≤ x < 6 m/s)",
                        },
                        "tooltip": _tooltip(
                            {
                                "Wind direction": "datum.deg + '°'",
                                "Gentle (3 ≤ x < 6 m/s)": "format(datum.speed_cat2 * 100, '.1f') + '%'",
                            }
                        ),
                    }
                },
            },
            {
                "type": "arc",
                "from": {"data": "wind_data"},
                "encode": {
                    "update": {
                        "x": {"signal": "width / 2"},
                        "y": {"signal": "height / 2"},
                        "startAngle": {
                            "signal": f"(datum.deg - {step_angle}/2) * PI / 180 "
                        },
                        "endAngle": {
                            "signal": f"(datum.deg + {step_angle}/2) * PI / 180"
                        },
                        "innerRadius": {"value": 0},
                        "outerRadius": {
                            "signal": "(datum.r1 / datum.max_total_radius) * max_pixel_radius"
                        },
                        "fill": {"scale": "color_scale", "value": "Light (< 3 m/s)"},
                        "tooltip": _tooltip(
                            {
                                "Wind direction": "datum.deg + '°'",
                                "Light (< 3 m/s)": "format(datum.speed_cat1 * 100, '.1f') + '%'",
                            }
                        ),
                    }
                },
            },
            {
                "type": "rule",
                "from": {"data": "radial_lines"},
                "encode": {
                    "update": {
                        "x": {"signal": "width / 2"},
                        "y": {"signal": "height / 2"},
                        "x2": {
                            "signal": "width / 2 + max_pixel_radius * sin(datum.data * PI / 180)"
                        },
                        "y2": {
                            "signal": "height / 2 - max_pixel_radius * cos(datum.data * PI / 180)"
                        },
                        "stroke": {"value": "#ccc"},
                        "strokeWidth": {"value": 1},
                        "strokeDash": {"value": [1, 1]},
                    }
                },
            },
            {
                "type": "arc",
                "from": {"data": "grid_circles"},
                "encode": {
                    "update": {
                        "x": {"signal": "width / 2"},
                        "y": {"signal": "height / 2"},
                        "startAngle": {"value": 0},
                        "endAngle": {"signal": "2 * PI"},
                        "innerRadius": {
                            "signal": "(datum.value / chart_radius) * max_pixel_radius"
                        },
                        "outerRadius": {
                            "signal": "(datum.value / chart_radius) * max_pixel_radius"
                        },
                        "stroke": {"value": "#ccc"},
                        "strokeDash": {"value": [3, 3]},
                        "strokeWidth": {"value": 1},
                        "fill": {"value": None},
                    }
                },
            },
            {
                "type": "text",
                "from": {"data": "compass_labels"},
                "encode": {
                    "update": {
                        "x": {
                            "signal": f"width / 2 + (max_pixel_radius + {font_size}) * sin(datum.angle * PI / 180)"
                        },
                        "y": {
                            "signal": f"height / 2 - (max_pixel_radius + {font_size}) * cos(datum.angle * PI / 180)"
                        },
                        "text": {"field": "label"},
                        "align": {"value": "center"},
                        "baseline": {"value": "middle"},
                        "fontSize": {"value": font_size},
                        "fontWeight": {"value": "normal"},
                        "fill": {"value": "#888"},
                    }
                },
            },
        ],
        "legends": [
            {
                "fill": "color_scale",
                "title": "Wind Speed",
                "labelFont": "sans-serif",
                "titleFont": "sans-serif",
                "symbolStrokeWidth": 1,
                "symbolSize": 100,
                "symbolType": "square",
                "direction": "vertical",
                "orient": "top-right",
            }
        ],
    }
