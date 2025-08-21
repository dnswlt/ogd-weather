"""Contains SQL queries for various use cases."""

import sqlalchemy as sa


def psql_total_bytes(user: str) -> sa.TextClause:
    sql = sa.text(
        f"""
        SELECT
            n.nspname AS schema,
            c.relname AS table,
            pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
            pg_total_relation_size(c.oid) AS total_bytes
        FROM
            pg_class c
        JOIN
            pg_namespace n ON n.oid = c.relnamespace
        WHERE
            c.relkind = 'r'  -- regular table
            AND pg_catalog.pg_get_userbyid(c.relowner) = :user
        ORDER BY
            pg_total_relation_size(c.oid) DESC;
        """
    )
    return sql.bindparams(user=user)
