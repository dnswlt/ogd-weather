"""Helpers for dealing with environment variables.

Especially relevant for Cloud deployments, where most parameters
will be provided as env vars.
"""

import json
import os
from pydantic import BaseModel


class PgConnectionInfo(BaseModel):
    user: str | None
    password: str | None
    host: str | None
    port: int | None
    dbname: str | None
    url: str | None

    @classmethod
    def from_env(cls, secret_var: str | None = None):

        if secret_var:
            secret_val = os.getenv(secret_var)
            if not secret_val:
                raise ValueError(f"{secret_var} is specified but not set")
            secret = json.loads(secret_val)
            user = secret["username"]
            password = secret["password"]
        else:
            user = os.getenv("OGD_DB_USER")
            password = os.getenv("OGD_DB_PASSWORD")

        host = os.getenv("OGD_DB_HOST")
        port = os.getenv("OGD_DB_PORT")
        dbname = os.getenv("OGD_DB_DBNAME")

        url = os.getenv("OGD_POSTGRES_URL")

        return cls(
            user=user,
            password=password,
            host=host,
            port=int(port) if port else None,
            dbname=dbname,
            url=url,
        )

    def get_url(self):
        if self.url:
            return self.url

        pw_suffix = f":{self.password}" if self.password else ""
        port_suffix = f":{self.port}" if self.port else ""

        return f"postgresql+psycopg://{self.user}{pw_suffix}@{self.host}{port_suffix}/{self.dbname}"
