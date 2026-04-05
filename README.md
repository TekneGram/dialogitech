# Dialogitech

Dialogitech is a small academic-paper processing pipeline built around filtered Markdown.

The workflow is:

1. extract metadata from Marker JSON
2. convert the document to filtered Markdown
3. split the Markdown into heading-based chunks
4. classify each chunk as `abstract`, `introduction`, `method`, `results`, or `discussion`
5. optionally send unresolved chunks to Gemma 4 for fallback classification

## Repository Layout

- `chunker/`: core pipeline code
- `marker/`: local Marker checkout and conversion outputs
- `pdfs/`: source PDFs

- `chunker/metadata_extractor.py`: title, journal, author, and reference extraction
- `chunker/boilerplate_filter.py`: converts Marker JSON into filtered Markdown
- `chunker/markdown_section_chunker.py`: splits filtered Markdown into section chunks
- `chunker/section_classifier.py`: deterministic section classification and enrichment
- `chunker/llm_section_classifier.py`: LLM fallback classification, quintiles, and context retrieval
- `chunker/run_section_classification.py`: CLI runner for classifying a filtered Markdown file

## Requirements

- Python 3.12
- macOS on Apple Silicon for the MLX Gemma 4 path
- a virtual environment at `.venv`

Install the repo dependencies:

```bash
./.venv/bin/pip install -r requirements.txt
```

## Basic Flow

There are two entry points:

1. start from an existing filtered Markdown file
2. start from Marker JSON, then convert it to filtered Markdown before chunking/classifying

## Run Deterministic Classification

This uses heading-based deterministic rules only:

```bash
./.venv/bin/python -m chunker.run_section_classification \
  marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md
```

Stdout output contains one row per chunk with:

- heading
- chunk index
- word count
- label
- source
- confidence
- whether context was used
- reason

## Run With Gemma 4 Fallback

The working setup uses a separate Unsloth MLX environment:

- Python executable:
  `/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python`
- model:
  `unsloth/gemma-4-E4B-it-UD-MLX-4bit`

Run the classifier with LLM fallback enabled for unresolved chunks:

```bash
./.venv/bin/python -m chunker.run_section_classification \
  marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md \
  --model-path unsloth/gemma-4-E4B-it-UD-MLX-4bit \
  --python-executable /Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python
```

Force the LLM to classify every chunk:

```bash
./.venv/bin/python -m chunker.run_section_classification \
  marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md \
  --model-path unsloth/gemma-4-E4B-it-UD-MLX-4bit \
  --python-executable /Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python \
  --force-llm
```

## LLM Behavior

The LLM path is controller-based:

- the model sees the chunk text and which quintile of the article it comes from
- if uncertain, it can request one round of context
- context is: previous heading section + current chunk + next heading section
- the final response is normalized into the same classification metadata used by the deterministic pipeline

## Example Pipeline From Marker JSON

If you already have a Marker JSON document in `marker/conversion_results/...`, the Python flow is:

```python
from pathlib import Path

from chunker import BoilerplateFilter, MetadataExtractor, MarkdownSectionChunker
from chunker.llm_section_classifier import ChunkClassificationLLM
from chunker.section_classifier import ChunkClassificationEnricher

json_path = Path("marker/conversion_results/2025_Uchida/2025_Uchida.json")
document = MetadataExtractor().load_json(json_path)
metadata = MetadataExtractor().extract_all(document)
markdown = BoilerplateFilter().convert_json_to_markdown(document, metadata=metadata)
heading_splits = MarkdownSectionChunker().process(markdown)

llm = ChunkClassificationLLM(
    filtered_markdown=markdown,
    heading_splits=heading_splits,
    model_path="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
    python_executable="/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python",
)

classified = ChunkClassificationEnricher(llm_classifier=llm).enrich_heading_splits(heading_splits)
```

## Notes

- The repo `.venv` is kept Marker-compatible.
- The Gemma 4 runtime is intentionally isolated in a separate Unsloth environment.
- The large downloaded Gemma model files live in the Hugging Face cache, not in this repo.
- Generated outputs such as classified JSON files should be treated as derived artifacts.
