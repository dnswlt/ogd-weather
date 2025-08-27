from pydantic import BaseModel
from typing import Iterable


class StationNotFoundError(ValueError):
    """Raised when a requested station doesn't exist."""


class NoDataError(ValueError):
    """Raised when a request is valid, but no data is available."""


class SchemaColumnMismatchInfo(BaseModel):
    table: str
    column: str
    is_missing: bool = False
    info: str = ""


class SchemaValidationError(ValueError):
    """Custom exception for sqlalchemy schema mismatch."""

    def __init__(
        self,
        message: str,
        *,
        missing_tables: Iterable[str] = (),
        mismatched_columns: Iterable[SchemaColumnMismatchInfo] = (),
    ) -> None:
        """Creates a new SchemaValidationError instance.

        Args:
            message: the exception message
            missing_tables: tables missing in the DB entirely
            missing_columns: columns missing in the DB (format: "{table_name}.{column_name}")
            mismatched_columns: columns that exist in the DB with different types or constraints.
        """
        super().__init__(message)
        self.missing_tables = list(missing_tables)
        self.mismatched_columns = list(mismatched_columns)

    def __str__(self) -> str:
        base = super().__str__()
        parts = []
        if self.missing_tables:
            parts.append(f"missing_tables={self.missing_tables}")
        if self.mismatched_columns:
            parts.append(f"mismatched_columns={self.mismatched_columns}")
        return f"{base} ({', '.join(parts)})" if parts else base
