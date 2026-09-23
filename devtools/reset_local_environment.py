"""Reset all generated local pipeline state after explicit confirmation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TARGETS = (
    "data/lancedb",
    "marker/conversion_results",
    "outputs",
    "xoutputs",
    "resoutputs",
    "ressessions",
    "data/doi_metadata_cache",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete all generated local pipeline state, including DOI cache."
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
            "Refusing to reset local state; re-run with --yes.\n"
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
