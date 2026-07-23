"""Connectivity + sanity check.

Connects to the demo database and prints the row count of every table.
Exits non-zero if no tables are found or any table is empty.

Usage:
    python scripts/check_db.py
"""

import sys
from pathlib import Path

# Allow running as a plain script: `python scripts/check_db.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import list_tables, query  # noqa: E402


def main() -> int:
    tables = list_tables()
    if not tables:
        print("No tables found in the public schema.")
        return 1

    print(f"Found {len(tables)} table(s):\n")
    counts = {}
    for t in tables:
        # Table names come from information_schema, not user input, so safe
        # to interpolate. Use a quoted identifier to be safe.
        df = query(f'SELECT COUNT(*) AS n FROM "{t}"')
        n = int(df["n"].iloc[0])
        counts[t] = n
        print(f"  {t:<20} {n:>12,} rows")

    empty = [t for t, n in counts.items() if n == 0]
    print()
    if empty:
        print(f"WARNING: {len(empty)} empty table(s): {', '.join(empty)}")
        return 1

    print(f"OK: all {len(tables)} table(s) have non-zero row counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
