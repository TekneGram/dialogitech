
import lancedb
from .embedding_service import OllamaEmbeddingService

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

    return query.limit(limit).to_list()

  