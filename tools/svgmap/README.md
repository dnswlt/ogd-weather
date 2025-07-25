# svgmap

This directory contains a helper script [svg_transform.py](./svg_transform.py)
and HTML page [suisse.html](./suisse.html) which can be used to
determine the affine transform that is needed to approximately map
WGS84 coordinates of locations in Switzerland onto the SVG
coordinates used by the Swiss map in [result.svg](./result.svg).

You can click on points on the map on the HTML page to obtain
the point's SVG coordinates. Then you can use OpenStreetMap or
any other tool to find the WGS84 coordinates for that location.
Enter those in the table at the bottom on the HTML page.
You need at least three positions (that must not be colinear).

Once you have those positions, save the JSON shown at the bottom
of the HTML page to a file and run

```bash
python3 svg_transform.py {your_file.json}
```

The script will output the best (least squares) fit of the
parameters of an affine transform that maps WGS84 coordinates
to SVG coordinates.

Using the example coordinates from [coords.json](./coords.json),
the output is as follows (irrelevant output omitted):

```bash
$ python3 svg_transform.py ./coords.json

--- Affine Transform Coeffs ---
SVG_X = 222.745610 * Longitude + -2.682887 * Latitude + -1167.482181
SVG_Y = -0.979986 * Longitude + -325.449415 * Latitude + 15580.771038

--- Transform Residuals (Errors for Calibration Points) ---
Lac Leman East      : Err_X=-0.28, Err_Y=0.51
Poschiavo S         : Err_X=-0.72, Err_Y=-1.29
Basel NE            : Err_X=-0.69, Err_Y=-2.35
VS/TI S             : Err_X=0.92, Err_Y=0.07
ZH/SG Rapperswil    : Err_X=0.76, Err_Y=3.06
Avg Abs Error X: 0.68
Avg Abs Error Y: 1.46

Resulting transform:
 {
  "a": 222.7456099377238,
  "b": -2.6828866160026887,
  "c": -1167.4821813828262,
  "d": -0.9799860475869895,
  "e": -325.44941520771715,
  "f": 15580.77103793717
}
```

## Credit

The map outline SVG is based on
<https://commons.wikimedia.org/wiki/File:Suisse_cantons.svg>
by [Pymouss44](https://commons.wikimedia.org/wiki/User:Pymouss44),
licensed under
[CC BY‑SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Only the map graphic is covered by this license;
all other parts of this application remain under their respective licenses.
