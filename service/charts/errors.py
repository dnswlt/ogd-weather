class StationNotFoundError(ValueError):
    """Raised when a requested station doesn't exist."""


class NoDataError(ValueError):
    """Raised when a request is valid, but no data is available."""
