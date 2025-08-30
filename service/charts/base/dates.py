import datetime


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
