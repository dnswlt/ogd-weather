import datetime

from . import constants as bc


def utc_timestr(d: datetime.datetime) -> str:
    """Returns the given datetime as a UTC time string in ISO format.

    Example: "2025-03-31 23:59:59Z"

    If d is a naive datetime (no tzinfo), it is assumed to be in UTC.
    """
    if d.tzinfo is not None:
        # datetime.UTC is only available since Python 3.11
        d = d.astimezone(datetime.timezone.utc)
    return d.strftime("%Y-%m-%d %H:%M:%SZ")


def utc_datestr(d: datetime.datetime | datetime.date) -> str:
    """Returns the given date or datetime as a UTC date string in ISO format.

    Example: "2025-03-31"

    If d is a naive datetime (no tzinfo), it is assumed to be in UTC.
    """
    if isinstance(d, datetime.datetime) and d.tzinfo is not None:
        # datetime.UTC is only available since Python 3.11
        d = d.astimezone(datetime.timezone.utc)
    return d.strftime("%Y-%m-%d")


MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

SEASON_NAMES = {
    "spring": "Spring (Mar-May)",
    "summer": "Summer (Jun-Aug)",
    "autumn": "Autumn (Sep-Nov)",
    "winter": "Winter (Jan-Feb,Dec)",
}

VALID_PERIODS = set(
    [str(i) for i in MONTH_NAMES] + list(SEASON_NAMES.keys()) + [bc.PERIOD_ALL]
)


def period_to_title(period: str) -> str:
    if period.isdigit():
        return MONTH_NAMES[int(period)]
    elif period in SEASON_NAMES:
        return SEASON_NAMES[period]
    elif period == bc.PERIOD_ALL:
        return "Whole Year"
    return "Unknown Period"
