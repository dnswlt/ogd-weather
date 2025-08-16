import json
import os
import pytest

from .env import PgConnectionInfo


# Optional helper to start from a clean slate
@pytest.fixture
def clear_env(monkeypatch):
    for k in [
        "OGD_DB_USER",
        "OGD_DB_PASSWORD",
        "OGD_DB_HOST",
        "OGD_DB_PORT",
        "OGD_DB_DBNAME",
        "OGD_POSTGRES_URL",
        "RDS_SECRET",  # example name used below
    ]:
        monkeypatch.delenv(k, raising=False)


def test_from_env_with_secret(monkeypatch, clear_env):
    # Simulate RDS-managed secret (username/password only)
    secret = {"username": "masteruser", "password": "s3cr3t"}
    monkeypatch.setenv("RDS_SECRET", json.dumps(secret))
    # Other connection bits come from normal env vars
    monkeypatch.setenv("OGD_DB_HOST", "db.example.com")
    monkeypatch.setenv("OGD_DB_PORT", "5432")
    monkeypatch.setenv("OGD_DB_DBNAME", "postgres")

    info = PgConnectionInfo.from_env(secret_var="RDS_SECRET")

    assert info.user == "masteruser"
    assert info.password == "s3cr3t"
    assert info.host == "db.example.com"
    assert info.port == 5432  # cast to int
    assert info.dbname == "postgres"
    assert info.url is None

    # And the constructed DSN:
    assert info.get_url() == (
        "postgresql+psycopg://masteruser:s3cr3t@db.example.com:5432/postgres"
    )


def test_from_env_missing_secret_raises(monkeypatch, clear_env):
    # secret_var provided but not set in env
    with pytest.raises(ValueError) as exc:
        PgConnectionInfo.from_env(secret_var="RDS_SECRET")
    assert "RDS_SECRET" in str(exc.value)


def test_from_env_plain_env_vars(monkeypatch, clear_env):
    # No secret_var => read all from standard env
    monkeypatch.setenv("OGD_DB_USER", "alice")
    monkeypatch.setenv("OGD_DB_PASSWORD", "pw")
    monkeypatch.setenv("OGD_DB_HOST", "localhost")
    monkeypatch.setenv("OGD_DB_PORT", "5433")
    monkeypatch.setenv("OGD_DB_DBNAME", "mydb")

    info = PgConnectionInfo.from_env()

    assert (info.user, info.password, info.host, info.port, info.dbname) == (
        "alice",
        "pw",
        "localhost",
        5433,
        "mydb",
    )
    assert info.get_url() == "postgresql+psycopg://alice:pw@localhost:5433/mydb"


def test_get_url_prefers_explicit_url(monkeypatch, clear_env):
    # If OGD_POSTGRES_URL is set, PgConnectionInfo should return it verbatim
    monkeypatch.setenv("OGD_POSTGRES_URL", "postgresql+psycopg://u:p@h:5432/d")
    info = PgConnectionInfo.from_env()
    assert info.url == "postgresql+psycopg://u:p@h:5432/d"
    assert info.get_url() == "postgresql+psycopg://u:p@h:5432/d"


def test_get_url_without_password(monkeypatch, clear_env):
    # Password optional: DSN must omit the colon if password is missing
    monkeypatch.setenv("OGD_DB_USER", "bob")
    monkeypatch.setenv("OGD_DB_HOST", "h")
    monkeypatch.setenv("OGD_DB_PORT", "5432")
    monkeypatch.setenv("OGD_DB_DBNAME", "d")

    info = PgConnectionInfo.from_env()
    assert info.password is None
    assert info.get_url() == "postgresql+psycopg://bob@h:5432/d"


def test_port_is_none(monkeypatch, clear_env):
    # Port missing -> None in model; (calling get_url would render 'None' in string)
    monkeypatch.setenv("OGD_DB_USER", "u")
    monkeypatch.setenv("OGD_DB_HOST", "h")
    monkeypatch.setenv("OGD_DB_DBNAME", "d")

    info = PgConnectionInfo.from_env()
    assert info.port is None
    assert info.get_url() == "postgresql+psycopg://u@h/d"
