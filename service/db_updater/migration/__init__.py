from .migration import (
    DBMigration,
    run_migration,
)

# Make sure to import all migrations here, so their classes register themselves in DBMigration.
from .m0001_update_status import UpdateStatusAddDestinationTable

__all__ = [
    # migration
    "DBMigration",
    "run_migration",
    # m0001_update_status
    "UpdateStatusAddDestinationTable",
]
