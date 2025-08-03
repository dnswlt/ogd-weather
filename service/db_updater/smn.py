import re
from pydantic import BaseModel


class HrefMatch(BaseModel):
    interval: str  # one of ("h", "d", "m")
    frequency: str  # one of ("historical", "recent", "now")
    years: tuple[int, int] | None = None


def match_csv_resource(href: str, filter_re: str | None = None) -> HrefMatch | None:
    if filter_re is not None and not re.search(filter_re, href):
        return None
    # TODO: Re-enable _now_ data at some point.
    # It had a few missing points recently,
    # we don't reconcile with recent data yet, and we're currently
    # focusing on historical data anyway.

    mo = re.search(
        r".*_(?P<interval>d|h|m)(_(?P<freq>historical|historical_(?P<years>\d+-\d+)|recent|now__DISABLED__))?.csv$",
        href,
    )
    if mo is None:
        return None

    interval = mo.group("interval")
    freq = mo.group("freq")
    if not freq:
        # Interpret missing suffix as historical data (happens e.g. for "m")
        freq = "historical"
    elif freq.startswith("historical"):
        freq = "historical"  # Trim suffix years.
    years_str = mo.group("years")
    years = tuple(map(int, years_str.split("-"))) if years_str else None

    return HrefMatch(
        interval=interval,
        frequency=freq,
        years=years,
    )
