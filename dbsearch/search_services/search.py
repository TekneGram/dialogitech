
import lancedb
from .embedding_service import OllamaEmbeddingService
from collections.abc import Collection
from chunker.llm_section_type_classifier_helpers.section_taxonomy import SectionLabel

class SearchLanceDB:
  def __init__(self) -> None:
    return

  def basic_search(self, text_to_search: str, limit: int = 10) -> list:
    db = lancedb.connect("data/lancedb")
    table = db.open_table("paper_chunks")

    # Embedd the question_to_embed and retrieve top 10 results.
    embedder = OllamaEmbeddingService(model="qwen3-embedding:0.6b")
    query_vector = embedder.embed_texts(
      [text_to_search]
    )[0]

    results = (
      table.search(
        query_vector,
        query_type="vector",
        vector_column_name="embedding",
      )
      .distance_type("cosine")
      .limit(limit)
      .to_list()
    )

    return results

  def basic_search_exclude_chunk_ids(self, text_to_search: str, limit: int = 10, excluded_chunk_ids: list[str] | None = None) -> list:
    db = lancedb.connect("data/lancedb")
    table = db.open_table("paper_chunks")
    embedder = OllamaEmbeddingService(model="qwen3-embedding:0.6b")
    query_vector = embedder.embed_texts(
      [text_to_search]
    )[0]

    query = table.search(
      query_vector,
      query_type="vector",
      vector_column_name="embedding"
    )

    if excluded_chunk_ids:
      escaped_ids = [
        "'" + chunk_id.replace("'", "''") + "'"
        for chunk_id in excluded_chunk_ids
      ]

      query = query.where(
        f"chunk_id NOT IN ({', '.join(escaped_ids)})",
        prefilter=True
      )

    return query.distance_type("cosine").limit(limit).to_list()


  ### SEARCH BY SECTION ###

  def search_by_section(
      self, 
      text_to_search: str, 
      sections: SectionLabel | Collection[SectionLabel], 
      limit: int = 10,
      excluded_chunk_ids: list[str] | None = None
    ) -> list[dict]:

    if limit <= 0:
      raise ValueError("Limit must be positive")

    if isinstance(sections, str):
      requested_sections=[sections]
    else:
      requested_sections = list(sections)

    if not requested_sections:
      raise ValueError("At least one section label is required.")

    escaped_sections = [
      "'" + section.replace("'", "''") + "'"
      for section in requested_sections
    ]

    db = lancedb.connect("data/lancedb")
    table = db.open_table("paper_chunks")

    embedder = OllamaEmbeddingService(model="qwen3-embedding:0.6b")
    query_vector = embedder.embed_texts([text_to_search])[0]

    if len(escaped_sections) == 1:
      section_filter = f"classification_label = {escaped_sections[0]}"
    else:
      section_filter = (
        f"classification_label IN ({', '.join(escaped_sections)})"
      )

    filters = [section_filter]

    if excluded_chunk_ids:
      escaped_ids = [
        "'" + chunk_id.replace("'", "''") + "'"
        for chunk_id in excluded_chunk_ids
      ]
      filters.append(
        f"chunk_id NOT IN ({', '.join(escaped_ids)})"
      )

    combined_filter = " AND ".join(filters)

    return (
      table.search(
        query_vector,
        query_type="vector",
        vector_column_name="embedding"
      ).where(combined_filter, prefilter=True).distance_type("cosine").limit(limit).to_list()
    )