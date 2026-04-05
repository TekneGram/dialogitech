from __future__ import annotations

import argparse

import lancedb

from .lancedb_schema import DEFAULT_TABLE_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a LanceDB chunk table.")
    parser.add_argument("--db-path", required=True, help="Path to the LanceDB database directory.")
    parser.add_argument(
        "--table-name",
        default=DEFAULT_TABLE_NAME,
        help="Name of the LanceDB table to inspect.",
    )
    parser.add_argument("--paper-id", help="Optional paper_id filter.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of sample rows to print.",
    )
    args = parser.parse_args()

    db = lancedb.connect(args.db_path)
    table = db.open_table(args.table_name)

    count = table.count_rows()
    print(f"table={args.table_name}")
    print(f"total_rows={count}")

    query = table.search()
    if args.paper_id:
        escaped_paper_id = args.paper_id.replace("'", "''")
        query = query.where(f"paper_id = '{escaped_paper_id}'")

    rows = query.limit(args.limit).to_list()
    print(f"sample_rows={len(rows)}")

    for index, row in enumerate(rows):
        print(f"-- row {index + 1} --")
        print(f"paper_id={row['paper_id']}")
        print(f"paper_title={row['paper_title']}")
        print(f"section_title={row['section_title']}")
        print(f"chunk_index={row['chunk_index']}")
        print(f"classification_label={row['classification_label']}")
        print(f"classification_source={row['classification_source']}")
        print(f"embedding_dim={len(row['embedding'])}")
        print(f"text_preview={row['text'][:160].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
