# AGENTS.md

## Purpose

This repository appears to contain Python utilities for document processing and chunking, with primary source code in `chunker/`.

## Working Agreement

- Keep changes targeted and minimal.
- Prefer small, composable Python modules over large scripts.
- Preserve existing directory structure unless there is a clear reason to reorganize it.
- Avoid committing generated artifacts, local environments, caches, or large input/output documents.

## Code Style

- Target Python 3.
- Follow PEP 8 and use descriptive names.
- Prefer type hints for new or modified Python code.
- Add comments only where the intent is not obvious from the code.

## Project Layout

- `chunker/`: active project code.
- `marker/`: ignored local or vendored dependency/work area.
- `pdfs/`: ignored document input/output area.

## Validation

- Run focused checks for the files you change.
- If tests are added later, prefer narrow test runs before broad suite runs.

## Git Hygiene

- Keep commits scoped to one logical change.
- Do not commit secrets, virtual environments, caches, generated files, or PDF assets.
