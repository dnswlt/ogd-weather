"""pytest settings, fixtures, flags for db_updater tests."""

import pytest
import re
import sqlalchemy as sa

from service.charts.db import schema as ds


def pytest_addoption(parser):
    """
    Adds the --test-postgres-url command-line option to pytest.
    """
    parser.addoption(
        "--test-postgres-url",
        action="store",
        default=None,
        help="PostgreSQL URL to use for integration tests",
    )


@pytest.fixture(scope="session")
def db_engine(request):
    """
    A session-scoped fixture that creates a database engine.

    It checks for the --test-postgres-url option. If provided, it connects
    to PostgreSQL. Otherwise, it uses an in-memory SQLite database.

    The fixture handles the creation of the schema and teardown.
    """
    postgres_url = request.config.getoption("--test-postgres-url")

    engine = None

    if postgres_url:
        print(f"\n--- Using PostgreSQL for integration tests: {postgres_url} ---")
        engine = sa.create_engine(postgres_url)
        with engine.connect() as conn:
            current_schema = conn.scalar(sa.text("select current_schema"))
            if re.search(r"(?<![a-z])test(?![a-z])", current_schema.lower()) is None:
                raise ValueError(
                    f"Must run DB integration tests on Postgres in a test schema, found {current_schema}"
                )
        # Make sure tables get generated in the current_schema, not in 'public'.
        # https://docs.sqlalchemy.org/en/20/core/connections.html#translation-of-schema-names
        engine = engine.execution_options(schema_translate_map={None: current_schema})

    else:
        print("\n--- Using in-memory SQLite for integration tests ---")
        engine = sa.create_engine("sqlite:///:memory:")

    # Setup: create all tables
    ds.metadata.create_all(engine)

    # Yield the engine to the tests
    yield engine

    ds.metadata.drop_all(engine)
    engine.dispose()
