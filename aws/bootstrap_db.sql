-- psql script to set up the initial role.
-- Run with: psql -v DB_PASSWORD='the-password' -f bootstrap_db.sql

-- Read-write access role for all apps: db-updater and charts.
CREATE ROLE weather LOGIN PASSWORD :'DB_PASSWORD';

-- Make role own the database and the public schema (so it can do DDL/DML)
ALTER DATABASE weather OWNER TO weather;
ALTER SCHEMA public OWNER TO weather;

-- Hardening: remove all other permissions.
REVOKE ALL ON DATABASE weather FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
