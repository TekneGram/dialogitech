from __future__ import annotations

import argparse

from chunker.llm_metadata_extractor_helpers.doi_metadata_client import (
    DoiMetadataClient,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a live Crossref DOI metadata lookup."
    )
    parser.add_argument("doi", help="DOI to look up, with or without a DOI URL.")
    args = parser.parse_args()

    client = DoiMetadataClient()
    normalized_doi = client.normalize_doi(args.doi)
    metadata = client.lookup(normalized_doi)

    if metadata.get("doi") != normalized_doi:
        raise RuntimeError("Crossref response DOI does not match the requested DOI.")
    if not any(metadata.get(field) for field in ("title", "journal", "year")):
        raise RuntimeError("Crossref returned no usable bibliographic metadata.")

    print(f"DOI: {normalized_doi}")
    print("Crossref metadata:", metadata)


if __name__ == "__main__":
    main()
