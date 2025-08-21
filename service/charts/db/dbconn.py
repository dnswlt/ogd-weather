"""Helpers for dealing with database connections.

Especially relevant for Cloud deployments, where most parameters
will be provided as env vars.
"""

import json
import os
from urllib.parse import urlparse
from pydantic import BaseModel


class PgConnectionInfo(BaseModel):
    user: str | None
    password: str | None
    host: str | None
    port: int | None
    dbname: str | None

    @classmethod
    def from_env(cls, secret_var: str | None = None) -> "PgConnectionInfo":
        """Builds a PgConnectionInfo from environment variables.

        Args:
            secret_var: the name of an environment variable to retrieve
                username and password from. If set, it must be a JSON string
                '{"username": "...", "password": "..."}' and OGD_DB_USER and
                OGD_DB_PASSWORD are ignored.

        If OGD_POSTGRES_URL is set, it is used and all OGD_DB_* vars are ignored.
        Otherwise, OGD_DB_{USER, PASSWORD, HOST, PORT, DBNAME} variables are read.
        """

        url = os.getenv("OGD_POSTGRES_URL")
        if url:
            parsed_url = urlparse(url)
            user = parsed_url.username
            password = parsed_url.password
            host = parsed_url.hostname
            port = parsed_url.port
            dbname = parsed_url.path.removeprefix("/")
        else:
            user = os.getenv("OGD_DB_USER")
            password = os.getenv("OGD_DB_PASSWORD")
            host = os.getenv("OGD_DB_HOST")
            port = os.getenv("OGD_DB_PORT")
            dbname = os.getenv("OGD_DB_DBNAME")

        if secret_var:
            secret_val = os.getenv(secret_var)
            if not secret_val:
                raise ValueError(f"{secret_var} is specified but not set")
            try:
                secret = json.loads(secret_val)
                user = secret["username"]
                password = secret["password"]
            except ValueError:
                raise ValueError(f"Invalid JSON value for {secret_var}: '{secret_val}'")

        return cls(
            user=user,
            password=password,
            host=host,
            port=int(port) if port else None,
            dbname=dbname,
        )

    @classmethod
    def from_url(cls, url: str) -> "PgConnectionInfo":
        parsed_url = urlparse(url)
        user = parsed_url.username
        password = parsed_url.password
        host = parsed_url.hostname
        port = parsed_url.port
        dbname = parsed_url.path.removeprefix("/")

        return cls(
            user=user,
            password=password,
            host=host,
            port=port if port else None,
            dbname=dbname,
        )

    def url(self):
        pw_suffix = f":{self.password}" if self.password else ""
        port_suffix = f":{self.port}" if self.port else ""

        return f"postgresql+psycopg://{self.user}{pw_suffix}@{self.host}{port_suffix}/{self.dbname}"

    def sanitized_url(self):
        """Returns the connection string as a URL without the password.

        Use this method for logging to avoid leaking passwords.
        """
        port_suffix = f":{self.port}" if self.port else ""

        return (
            f"postgresql+psycopg://{self.user}@{self.host}{port_suffix}/{self.dbname}"
        )
