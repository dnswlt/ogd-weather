import json
import os
import sys
import numpy as np
from pydantic import BaseModel


class CalibrationPoint(BaseModel):
    svg_x: float
    svg_y: float
    wsg84_lon: float
    wsg84_lat: float
    notes: str = ""


class AffineTransform(BaseModel):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def apply(self, lat: float, lon: float) -> tuple[float, float]:
        """Transforms WSG84 coordinates to SVG coordinates using affine coefficients."""
        svg_x = self.a * lon + self.b * lat + self.c
        svg_y = self.d * lon + self.e * lat + self.f
        return svg_x, svg_y


def json_to_points(json_str: str) -> list[CalibrationPoint]:
    """Parses a JSON input into a list of CalPoint objects."""
    try:
        raw_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

    points: list[CalibrationPoint] = []
    for item in raw_data:
        try:
            # We assume svg_x and svg_y are always present and valid floats from the JS side
            point = CalibrationPoint(
                svg_x=float(item["svg_x"]),
                svg_y=float(item["svg_y"]),
                wsg84_lon=float(item["wsg84_lon"]),
                wsg84_lat=float(item["wsg84_lat"]),
                notes=item.get("notes", ""),
            )
            points.append(point)
        except (ValueError, TypeError, KeyError) as e:
            print(f"Warning: Skipping malformed point data: {item} - {e}")
    return points


def calc_transform(points: list[CalibrationPoint]) -> AffineTransform:
    """
    Calculates the 2D affine transformation coefficients from WSG84 to SVG coordinates.
    Filters out points that do not have complete SVG and WSG84 coordinates.
    """
    # Filter for points that have complete SVG AND WSG84 coordinates
    if len(points) < 3:
        raise ValueError("Need at least 3 complete calibration points")

    # Prepare the matrices for solving
    wgs84_matrix = np.array([[p.wsg84_lon, p.wsg84_lat] for p in points])
    svg_x_vec = np.array([p.svg_x for p in points])
    svg_y_vec = np.array([p.svg_y for p in points])

    # Augment the WSG84 matrix with a column of ones for the constant term
    A = np.hstack((wgs84_matrix, np.ones((wgs84_matrix.shape[0], 1))))

    # Solve for a, b, c (for X transformation) using least squares
    coeffs_x, _, _, _ = np.linalg.lstsq(A, svg_x_vec, rcond=None)
    a, b, c = coeffs_x.tolist()

    # Solve for d, e, f (for Y transformation) using least squares
    coeffs_y, _, _, _ = np.linalg.lstsq(A, svg_y_vec, rcond=None)
    d, e, f = coeffs_y.tolist()

    print("\n--- Affine Transform Coeffs ---")
    print(f"SVG_X = {a:.6f} * Longitude + {b:.6f} * Latitude + {c:.6f}")
    print(f"SVG_Y = {d:.6f} * Longitude + {e:.6f} * Latitude + {f:.6f}")
    print("-" * 40)

    # Calculate and print residuals (error)
    print("\n--- Transform Residuals (Errors for Calibration Points) ---")
    total_abs_err_x = 0
    total_abs_err_y = 0
    for p in points:
        pred_svg_x = a * p.wsg84_lon + b * p.wsg84_lat + c
        pred_svg_y = d * p.wsg84_lon + e * p.wsg84_lat + f
        error_x = p.svg_x - pred_svg_x
        error_y = p.svg_y - pred_svg_y
        point_name = (
            p.notes if p.notes else f"Lon:{p.wsg84_lon:.4f}, Lat:{p.wsg84_lat:.4f}"
        )
        print(f"{point_name:<20}: Err_X={error_x:.2f}, Err_Y={error_y:.2f}")
        total_abs_err_x += abs(error_x)
        total_abs_err_y += abs(error_y)
    print(f"Avg Abs Error X: {total_abs_err_x / len(points):.2f}")
    print(f"Avg Abs Error Y: {total_abs_err_y / len(points):.2f}")

    return AffineTransform(a=a, b=b, c=c, d=d, e=e, f=f)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <json-string-or-filename>")
        sys.exit(1)

    if os.path.isfile(sys.argv[1]):
        json_str = open(sys.argv[1]).read()
    else:
        json_str = sys.argv[1]

    points = json_to_points(json_str)
    print(f"Parsed {len(points)} points from JSON.")

    if not points:
        sys.exit(1)

    transf = calc_transform(points)

    if not transf:
        print("Failed to calculate transform")
        sys.exit(1)

    print("\nResulting transform:\n", transf.model_dump_json(indent=2))

    print("\n--- Transforming New Points ---")
    # Example: Transform Zurich HB
    zurich_lon = 8.5359
    zurich_lat = 47.3783
    svg_x_lausanne, svg_y_lausanne = transf.apply(zurich_lon, zurich_lat)
    print(
        f"Zurich HB / Europaallee (WSG84: {zurich_lon}, {zurich_lat}) -> SVG: ({svg_x_lausanne:.2f}, {svg_y_lausanne:.2f})"
    )

    # Example: Transform Interlaken Ost
    interlaken_lon = 7.8691
    interlaken_lat = 46.6908
    svg_x_interlaken, svg_y_interlaken = transf.apply(interlaken_lon, interlaken_lat)
    print(
        f"Interlaken Ost (WSG84: {interlaken_lon}, {interlaken_lat}) -> SVG: ({svg_x_interlaken:.2f}, {svg_y_interlaken:.2f})"
    )


if __name__ == "__main__":
    main()
