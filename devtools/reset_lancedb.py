"""Remove the local LanceDB database after explicit confirmation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete the local data/lancedb directory."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion. Required because this operation is destructive.",
    )
    args = parser.parse_args()

    target = Path(__file__).resolve().parents[1] / "data" / "lancedb"
    if not args.yes:
        parser.error(f"Refusing to delete {target}; re-run with --yes.")

    if target.exists():
        shutil.rmtree(target)
        print(f"Removed {target}")
    else:
        print(f"Nothing to remove; {target} does not exist.")


if __name__ == "__main__":
    main()
