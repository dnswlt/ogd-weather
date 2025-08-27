import sqlalchemy as sa


class DBMigration:
    """Parent class for all DB migration classes."""

    # Holds all migration subclasses
    ALL_MIGRATIONS: dict[str, type["DBMigration"]] = {}

    def __init_subclass__(cls, migration_id: str, **kwargs):
        """Registers the subclass in ALL_MIGRATIONS.

        Raises:
            TypeError if a migration with the same ID already exists.
        """
        super().__init_subclass__(**kwargs)
        if migration_id in DBMigration.ALL_MIGRATIONS:
            raise TypeError(
                f"Migration with id '{migration_id}' is already registered."
            )

        DBMigration.ALL_MIGRATIONS[migration_id] = cls

    def execute(self, engine: sa.Engine):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement 'execute'"
        )


def run_migration(engine: sa.Engine, migration_id: str) -> None:
    cls = DBMigration.ALL_MIGRATIONS.get(migration_id)
    if cls is None:
        raise ValueError(f"No migration is registered under ID {migration_id}")

    inst = cls()
    inst.execute(engine)
