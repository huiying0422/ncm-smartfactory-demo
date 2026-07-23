"""Build the batch feature table and persist it.

Usage:
    python scripts/build_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import build_batch_features, write_batch_features  # noqa: E402


def main() -> int:
    df = build_batch_features()
    print(f"build_batch_features() -> shape {df.shape}")

    n = write_batch_features(df)
    print(f"write_batch_features() -> wrote {n} rows to batch_features")

    print("\nColumns:")
    for c in df.columns:
        print(f"  {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
