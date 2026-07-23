"""Database access layer.

Thin psycopg2 helpers around the demo Postgres database:

    get_conn()                     -> context manager yielding a connection
    query(sql, params=None)        -> pandas.DataFrame
    list_tables()                  -> list[str] of public table names
"""

import warnings
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence

import pandas as pd
import psycopg2

from src.config import get_pgurl


@contextmanager
def get_conn() -> Iterator["psycopg2.extensions.connection"]:
    """Yield a psycopg2 connection, committing on success and closing always.

    Usage:
        with get_conn() as conn:
            ...
    """
    conn = psycopg2.connect(get_pgurl())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query(sql: str, params: Optional[Sequence] = None) -> pd.DataFrame:
    """Run a read query and return the results as a pandas DataFrame.

    Args:
        sql: SQL statement, optionally with psycopg2 placeholders.
        params: Sequence/tuple/dict of parameters for the placeholders.

    Returns:
        A pandas DataFrame of the result set (empty if no rows returned).
    """
    with get_conn() as conn:
        # pandas warns that it only "officially" supports SQLAlchemy
        # connectables; the raw psycopg2 DBAPI connection works fine here
        # and avoids adding SQLAlchemy as a dependency.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.read_sql_query(sql, conn, params=params)


def list_tables() -> list:
    """Return the names of all base tables in the public schema."""
    sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """
    df = query(sql)
    return df["table_name"].tolist()
