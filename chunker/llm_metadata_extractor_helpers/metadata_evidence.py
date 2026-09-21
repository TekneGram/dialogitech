from __future__ import annotations

import re
from typing import Any


class MetadataEvidence:
  """Select and compact Marker pages for metadata prompts."""

  def select_pages(
      self,
      document: dict[str, Any],
      page_numbers: list[int],
  ) -> list[dict[str, Any]]:
    if not isinstance(document, dict):
      raise TypeError("Marker document must be a dictionary.")
    if not isinstance(page_numbers, list):
      raise TypeError("page_numbers must be a list of integers.")

    requested_pages: list[int] = []
    for page_number in page_numbers:
      if isinstance(page_number, bool) or not isinstance(page_number, int):
        raise TypeError("Each page number must be a non-negative integer.")
      if page_number < 0:
        raise ValueError("Page numbers must be non-negative.")
      if page_number not in requested_pages:
        requested_pages.append(page_number)

    children = document.get("children")
    if not isinstance(children, list):
      raise ValueError("Marker document does not contain a valid 'children' list.")

    pages_by_number: dict[int, dict[str, Any]] = {}
    for page in children:
      if not isinstance(page, dict):
        continue
      page_id = page.get("id")
      if not isinstance(page_id, str):
        continue
      match = re.search(r"/page/(\d+)/", page_id)
      if match is not None:
        pages_by_number.setdefault(int(match.group(1)), page)

    missing_pages = [
        page_number
        for page_number in requested_pages
        if page_number not in pages_by_number
    ]
    if missing_pages:
      raise KeyError(f"Marker pages not found: {missing_pages}")

    return [pages_by_number[page_number] for page_number in requested_pages]

  def available_page_numbers(self, document: dict[str, Any]) -> set[int]:
    children = document.get("children")
    if not isinstance(children, list):
      return set()

    page_numbers: set[int] = set()
    for page in children:
      if not isinstance(page, dict) or not isinstance(page.get("id"), str):
        continue
      match = re.search(r"/page/(\d+)/", page["id"])
      if match is not None:
        page_numbers.add(int(match.group(1)))
    return page_numbers

  def select_available_pages(
      self,
      document: dict[str, Any],
      page_numbers: list[int],
  ) -> list[dict[str, Any]]:
    available = self.available_page_numbers(document)
    requested = [page for page in page_numbers if page in available]
    return self.select_pages(document, requested) if requested else []

  def compact_pages(
      self,
      pages: list[dict[str, Any]],
  ) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(pages, list):
      raise TypeError("pages must be a list of page dictionaries.")

    compact_pages: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages):
      if not isinstance(page, dict):
        raise TypeError("Each page must be a dictionary.")

      page_id = page.get("id")
      if not isinstance(page_id, str):
        raise ValueError(f"Page at index {page_index} does not have a valid 'id'.")

      page_match = re.search(r"/page/(\d+)/", page_id)
      if page_match is None:
        raise ValueError(f"Could not determine page number from page ID: {page_id!r}")
      page_number = int(page_match.group(1))

      blocks = page.get("children")
      if not isinstance(blocks, list):
        raise ValueError(
            f"Page at index {page_index} does not contain a valid 'children' list."
        )

      compact_blocks = [
          {
              "block_type": block["block_type"],
              "html": block["html"],
          }
          for block in blocks
          if isinstance(block, dict)
          and isinstance(block.get("block_type"), str)
          and isinstance(block.get("html"), str)
      ]
      compact_pages.append({"page_number": page_number, "blocks": compact_blocks})

    return {"pages": compact_pages}
