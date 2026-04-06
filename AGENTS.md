# AGENTS.md

## Purpose

This repository processes academic papers into structured, classified chunk data and stores those chunks in LanceDB for later research retrieval.

The pipeline is:

1. start from a PDF or an existing Marker JSON artifact
2. extract paper metadata from Marker JSON
3. convert Marker JSON into filtered Markdown
4. remove the `References` section and everything after it
5. chunk the remaining paper body by heading
6. classify chunks as `abstract`, `introduction`, `method`, `results`, or `discussion`
7. use Gemma 4 fallback for unresolved chunks
8. generate local embeddings with Ollama
9. insert chunks plus metadata into LanceDB

The key architectural rule is that incomplete classification is not acceptable. Unresolved chunks must not be inserted into the database.

## Key Directories

- `chunker/`: active project code for metadata extraction, filtering, chunking, and classification
- `dbinsert/`: LanceDB schema, embedding, ingestion, indexing, and full-pipeline orchestration
- `dbquery/`: query rewriting, retrieval, ranking, and summarization over LanceDB
- `dbxquery/`: grounded follow-up querying over prior synthesized summaries
- `resquery/`: iterative research sessions over LanceDB using prior claims as turn context
- `marker/`: local Marker checkout plus conversion outputs
- `pdfs/`: source PDFs

## Core Modules

### `chunker/metadata_extractor.py`

Extracts:

- title
- authors
- journal metadata
- references

from Marker JSON.

### `chunker/boilerplate_filter.py`

Converts Marker JSON into filtered Markdown.

It should:

- remove front matter noise
- remove figures, tables, page furniture, and boilerplate
- remove the `References` section
- drop everything after `References`, including appendices, prompt templates, and supplementary content
- preserve the paper body heading structure

This references cutoff is important. Appendix material must not reach chunking or classification.

### `chunker/markdown_section_chunker.py`

Splits filtered Markdown into heading-based chunks using:

- `MarkdownHeading`
- `SectionChunk`
- `HeadingSplit`
- `MarkdownSectionChunker`

Chunks are word-count based with overlap and sentence-aware boundaries. The overlap logic should not start a new chunk in the middle of a sentence.

### `chunker/section_classifier.py`

Pipeline-facing classification layer. It should contain only:

- `ChunkClassification`
- `ClassifiedSectionChunk`
- `ClassifiedHeadingSplit`
- `DeterministicSectionClassifier`
- `ChunkClassificationEnricher`
- `classify_filtered_markdown(...)`

Responsibilities:

- deterministic heading-based rules
- orchestration between deterministic classification and the LLM classifier
- passing the previous resolved label into the LLM edge-case path when needed

Do not move prompt logic, markdown anchoring, JSON parsing, or MLX runtime control back into this file.

### `chunker/llm_section_classifier.py`

LLM-specific classification layer. It should contain:

- `ChunkClassificationLLM`
- `ChunkLocation`
- `ChunkContext`
- `LLMDecision`
- `article_quintile(...)`

Responsibilities:

- prompt construction
- quintiles
- chunk anchoring inside filtered Markdown
- context retrieval
- response parsing and recovery from malformed JSON-like model output
- external MLX subprocess integration
- chunk-level logging for requests and responses

The current controller behavior is:

1. classify the chunk directly
2. if needed, allow one context round
3. if the model requests context again, send one final constrained prompt using the previous resolved chunk label and require a concrete label

If Gemma still does not provide a concrete label but a previous resolved chunk label exists, the classifier should log the failure and deterministically inherit the previous resolved label. If no previous resolved label exists, the run should still fail rather than silently continue.

### `chunker/mlx_llm_runner.py`

Thin external runner for Gemma 4 when using the separate Unsloth MLX environment.

Reads JSON from stdin and writes raw model output to stdout.

### `chunker/run_section_classification.py`

CLI runner for classifying a filtered Markdown file from the terminal.

### `dbinsert/full_pipeline_service.py`

Main end-to-end pipeline orchestration from PDF to LanceDB.

Current behavior:

- derives `paper_id` from the PDF filename
- reuses Marker JSON unless rerun is forced
- can rerun filtered markdown and classification independently
- writes per-paper Marker and classification logs
- requires all chunks to be resolved before DB insertion

### `dbinsert/run_full_pipeline.py`

CLI entry point for the full PDF-to-LanceDB pipeline.

### `dbinsert/embedding_service.py`

Embedding backends.

The practical default is local Ollama embeddings with:

- model: `qwen3-embedding:0.6b`

### `dbinsert/index_manager.py`

Creates LanceDB vector, full-text, and scalar indexes.

Index creation must be idempotent. Repeated ingests should not fail because an index already exists.

### `dbquery/`

Database-query code should stay modular and split by responsibility.

Preferred shape:

- one main class per file
- one concern per class
- orchestration in a pipeline/coordinator class
- prompt construction and response parsing in the LLM-facing class that owns that prompt
- database retrieval in a retrieval-specific class
- ranking/fusion in a ranking-specific class
- output rendering in a writer-specific class

If logic starts getting long:

- move shared helpers into small modules under `dbquery/utils/`
- keep helpers narrow and reusable
- avoid growing one file into a mixed retrieval-plus-prompt-plus-output module

For `dbquery`, prefer small files and explicit dependency injection over large utility-heavy base classes.

### `dbxquery/`

Follow-up query code should stay modular and split by responsibility.

Preferred shape:

- one main class per file
- one concern per class
- orchestration in a follow-up pipeline/coordinator class
- evidence planning in an LLM-facing planner class
- plan validation in a validator class
- evidence retrieval in a retrieval adapter class
- grounded answer writing in an LLM-facing writer class
- shared parsing or prompt helpers in `dbxquery/utils/`

Current scope:

- single-turn only
- synchronous only
- follow-up input is `prior synthesized summary + user question`
- Gemma emits a strict JSON evidence plan
- the plan may call only one tool: `retrieve_evidence`
- the final answer must be written from retrieved evidence only

Allowed follow-up filters are:

- `paper_id_in`
- `authors_any`
- `year_min`
- `year_max`
- `classification_label_in`
- `section_title_contains`

`dbxquery` should prefer the `## Synthesized Summary` section from a prior `dbquery` output artifact when present, rather than reusing the entire prior query markdown file as prompt context.

### `resquery/`

Iterative research-session code should stay modular and split by responsibility.

Preferred shape:

- one main class per file
- one concern per class
- orchestration in a session pipeline/coordinator class
- query-context construction in a dedicated builder class
- claim extraction in a dedicated LLM-facing extractor class
- follow-up suggestion generation in a dedicated LLM-facing suggester class
- session persistence in a state store class
- output rendering in a writer class
- shared parsing or prompt helpers in `resquery/utils/`

Current scope:

- multi-turn only
- synchronous only
- prior claims are the main reusable session memory
- follow-up suggestions are user-facing suggestions only, not resolved/unresolved state
- `dbquery` remains the retrieval and synthesis engine
- the next turn query should be contextualized as:
  `This is what we know so far + prior claims + new user question`
- claims should be extracted from individual `dbquery` batch summaries, not from the final synthesized summary and not from raw fused chunks
- claim evidence should be attached programmatically from the `chunk_ids` already stored on each batch summary

`resquery` should preserve a compact research session with:

- session metadata
- ordered turns
- prior claims
- follow-up suggestions
- evidence index metadata

`resquery` should not reintroduce `active_focus` or resolved/unresolved question tracking unless there is a clear new requirement.

## Runtime Model Setup

The repo `.venv` is for the repo itself and should stay Marker-compatible.

Do not install Gemma 4 support directly into the repo `.venv` unless there is a strong reason and you understand the dependency impact.

The working Gemma 4 setup is:

- external Python:
  `/Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python`
- model identifier:
  `unsloth/gemma-4-E4B-it-UD-MLX-4bit`

The downloaded model weights are not stored in this repo. They live in the Hugging Face cache under the user home directory.

Important operational note:

- inside the Codex sandbox, MLX/Metal may fail to initialize
- outside the sandbox, the Unsloth MLX runtime works correctly on this machine

If you need to prove the real Gemma path works, run it outside the sandbox.

## Important Output Semantics

Each classified chunk must preserve:

- original chunk text
- chunk index
- word count
- final label
- source
- confidence
- used_context
- reason

This metadata is downstream pipeline input, not just display output. Preserve alignment with the original chunk objects.

For database rows, also preserve paper-level metadata such as:

- `paper_id`
- `paper_title`
- `authors`
- `journal`
- `volume`
- `issue`
- `year`
- `doi`
- `issn`
- `references`
- `pdf_path`
- `marker_json_path`
- `markdown_path`

## Working Rules

- Prefer filtered Markdown as the source of truth for chunk context.
- Avoid re-chunking unnecessarily if chunk data already exists in the current pipeline step.
- Keep deterministic classification conservative.
- Prefer passing unresolved cases to the LLM rather than adding weak heuristic text rules.
- Never allow unresolved chunks into LanceDB.
- Preserve repo `.venv` dependency integrity.
- Keep new code modular; avoid putting orchestration, prompting, parsing, and runtime control into one file.
- Treat `References` as the end of the paper body.
- In `dbquery`, keep files short and focused.
- In `dbquery`, classes should own a single responsibility and should not become long mixed-purpose classes.
- In `dbquery`, move formatting, ranking math, or text-budget helpers into subfolders such as `dbquery/utils/` when that keeps primary classes small.
- In `dbxquery`, keep files short and focused.
- In `dbxquery`, classes should own a single responsibility and should not become long mixed-purpose classes.
- In `dbxquery`, keep planner output tightly schema-constrained and validate it before retrieval runs.
- In `dbxquery`, final answers must be grounded only in retrieved evidence; if evidence is weak or missing, the answer should say so explicitly.
- In `resquery`, keep files short and focused.
- In `resquery`, classes should own a single responsibility and should not become long mixed-purpose classes.
- In `resquery`, prior claims should influence the next `dbquery` turn by contextualizing the query text before retrieval runs.
- In `resquery`, claim provenance should come from known `dbquery` batch-summary `chunk_ids`, not from model-generated chunk IDs.
- In `resquery`, follow-up suggestions are suggestions for the user, not a state machine.

## Full Pipeline Behavior

The full runner is resumable by artifact:

- reuse existing Marker JSON unless `--rerun-marker`
- reuse existing filtered markdown unless `--rerun-filtered-markdown`
- reuse existing classified chunks unless `--rerun-classification`

The full runner writes:

- `marker/conversion_results/<paper_id>/<paper_id>_marker.log`
- `marker/conversion_results/<paper_id>/<paper_id>_classification.log`

Marker output should be streamed to a log file, not captured fully in memory.

During long Marker runs, the pipeline should print elapsed-time progress and allow the user to stop the run without starting parallel jobs.

## Validation

When changing classification or filtering code, verify at least:

1. `./.venv/bin/python -m compileall chunker dbinsert`
2. deterministic classification on `marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md`
3. if touching the LLM path, one real fallback chunk through the external Gemma runtime
4. if touching filtering, confirm appendix content after references is absent from regenerated filtered markdown

When changing `dbquery`, `dbxquery`, or `resquery`, verify at least:

1. `./.venv/bin/python -m compileall dbquery dbxquery resquery`
2. one `dbquery` run that writes an output artifact with a synthesized summary
3. if touching `dbxquery`, one real follow-up run through the external Gemma runtime
4. if touching `resquery`, one real multi-turn `resquery` session through the external Gemma runtime
5. if touching `resquery`, confirm that:
   - prior claims are present in the contextualized query text sent to `dbquery`
   - claims are extracted from batch summaries
   - claim evidence comes from batch-summary `chunk_ids`
   - follow-up suggestions are written to session state and output artifacts
6. if touching follow-up filtering, confirm the saved trace records the tool calls, retrieved evidence, and grounded final answer

## Useful Commands

Deterministic run:

```bash
./.venv/bin/python -m chunker.run_section_classification \
  marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md
```

Run with Gemma fallback:

```bash
./.venv/bin/python -m chunker.run_section_classification \
  marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md \
  --model-path unsloth/gemma-4-E4B-it-UD-MLX-4bit \
  --python-executable /Users/danielparsons/.unsloth/unsloth_gemma4_mlx/bin/python
```

Run the full pipeline from a PDF:

```bash
./.venv/bin/python -m dbinsert.run_full_pipeline \
  pdfs/2025_Zhu.pdf \
  --db-path data/lancedb \
  --replace-existing
```

Rerun filtered markdown and classification only:

```bash
./.venv/bin/python -m dbinsert.run_full_pipeline \
  pdfs/2025_Zhu.pdf \
  --db-path data/lancedb \
  --replace-existing \
  --rerun-filtered-markdown \
  --rerun-classification
```

Run a follow-up query over a prior synthesized summary:

```bash
./.venv/bin/python -m dbxquery.run_followup \
  "Tell me how Zhu and Uchida differ on this issue?" \
  --prior-summary-path outputs/llms_cefr_quality_query.md \
  --db-path data/lancedb \
  --output-path xoutputs/zhu_vs_uchida_differences_followup.md
```

Run an iterative `resquery` session turn:

```bash
./.venv/bin/python -m resquery.run_session \
  "What are the weaknesses of LLMs?" \
  --session-path ressessions/llm_weaknesses.json \
  --db-path data/lancedb \
  --query-output-path resoutputs/llm_weaknesses_t1_dbquery.md \
  --output-path resoutputs/llm_weaknesses_t1_resquery.md
```

Continue the same `resquery` session with prior-claim context:

```bash
./.venv/bin/python -m resquery.run_session \
  "How does chain of thought impact the accuracy of CEFR generated text?" \
  --session-path ressessions/llm_weaknesses.json \
  --db-path data/lancedb \
  --query-output-path resoutputs/llm_weaknesses_t2_dbquery.md \
  --output-path resoutputs/llm_weaknesses_t2_resquery.md
```
