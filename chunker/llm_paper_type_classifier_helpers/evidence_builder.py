from typing import Any
from chunker.llm_paper_type_classifier_helpers.paper_type_models import PaperTypeEvidence
from chunker.markdown_section_chunker import MarkdownSectionChunker

class PaperTypeEvidenceBuilder:

  def build(self, filtered_markdown: str, metadata: dict[str, Any] | None = None) -> PaperTypeEvidence:
    metadata = metadata or {}
    headings = [heading.title for heading in MarkdownSectionChunker().extract_headings(filtered_markdown)]
    abstract = self._abstract_from_markdown(filtered_markdown)
    return PaperTypeEvidence(
      title=self._optional_string(metadata.get("title")),
      abstract=abstract,
      headings=headings,
      opening_excerpt=filtered_markdown[:1800].strip(),
      closing_excerpt=filtered_markdown[-900:].strip()
    )

  def _abstract_from_markdown(self, markdown: str) -> str | None:
    splits = MarkdownSectionChunker().split_by_headings(markdown)
    for split in splits:
      if split.title.strip().lower() == "abstract":
        return split.content[:1800].strip() or None

    return None

  def _optional_string(self, value: object) -> str | None:
    if not isinstance(value, str):
      return None

    return value.strip() or None
