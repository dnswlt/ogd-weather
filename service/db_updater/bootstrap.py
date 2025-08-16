import json
import logging
import sqlalchemy as sa
from typing import Any

from service.charts import env


logger = logging.getLogger("bootstrap")

_BOOTSTRAP_SQL = """
-- Read-write access role for all apps: db-updater and charts.
CREATE ROLE {role} LOGIN PASSWORD '{password}';

-- Make role own the database and the public schema (so it can do DDL/DML)
ALTER DATABASE {dbname} OWNER TO {role};
ALTER SCHEMA public OWNER TO {role};

-- Hardening: remove all other permissions.
REVOKE ALL ON DATABASE {dbname} FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
"""


def _create_engine(
    host: str, port: int, dbname: str, user: str, password: str
) -> sa.Engine:
    postgres_url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
    engine = sa.create_engine(postgres_url, echo=False)  # Don't log passwords
    return engine


def bootstrap_postgres(
    pgconn: env.PgConnectionInfo, master_secret: str | dict[str, Any]
):
    if isinstance(master_secret, str):
        master_secret = json.loads(master_secret)

    # Create an engine that uses the master credentials.
    engine = _create_engine(
        host=pgconn.host,
        port=pgconn.port,
        dbname=pgconn.dbname,
        user=master_secret["username"],
        password=master_secret["password"],
    )

    # Execute the bootstrap script.
    with engine.begin() as conn:
        bootstrap_sql = _BOOTSTRAP_SQL.format(
            role=pgconn.user,
            password=pgconn.password,
            dbname=pgconn.dbname,
        )
        conn.execute(sa.text(bootstrap_sql))

    logger.info(
        f"Bootstrapped DB {pgconn.dbname} on host {pgconn.host} for role {pgconn.user}."
    )
