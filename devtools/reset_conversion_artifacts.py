"""Remove generated conversion, classification, query, and session artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TARGETS = (
    "marker/conversion_results",
    "outputs",
    "xoutputs",
    "resoutputs",
    "ressessions",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete generated paper-conversion and query artifacts."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion. Required because this operation is destructive.",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    targets = [repository_root / relative_path for relative_path in TARGETS]
    if not args.yes:
        parser.error(
            "Refusing to delete generated artifacts; re-run with --yes.\n"
            + "Targets:\n"
            + "\n".join(f"- {target}" for target in targets)
        )

    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed {target}")
        else:
            print(f"Nothing to remove; {target} does not exist.")


if __name__ == "__main__":
    main()
