"""Helper functions for common pandas operations."""


def pctl(p: int):
    """Returns a function that calculates the given percentile p of a series.

    :param p:
    The percentile to compute; must be in [0, 100].

    The returned function will have the name f"p{p}", e.g. "p50" for pctl(50).
    """

    if p < 0 or p > 100 or int(p) != p:
        raise ValueError(f"Invalid percentile {p}")

    def f(s):
        return s.quantile(p / 100.0)

    f.__name__ = f"p{p}"  # Get columns names p25, p50, etc.
    return f
