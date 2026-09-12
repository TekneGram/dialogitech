# VecQuery Module Instructions

## Aims

`VecQuery` is the repository's general-purpose query and research orchestration module for the LanceDB paper-chunk database.

Its aims are to:

- let users ask natural-language questions over a collection of academic papers;
- put Gemma 4 at the forefront of query interpretation and workflow selection;
- use native Gemma tool calling where supported, with explicitly registered and bounded Python tools;
- support semantic, lexical, metadata-filtered, comparative, multi-hop, evidence, aggregation, and similarity-network queries;
- preserve the distinction between model reasoning, tool selection, tool execution, retrieval, state, and output;
- maintain reliable provenance from every answer back to retrieved chunk IDs, paper IDs, and source artifacts;
- support context-aware multi-step queries without allowing unbounded conversation growth;
- remain modular enough to add new query types without creating a large universal controller;
- produce inspectable traces so every model decision and tool result can be reviewed.

`VecQuery` should reuse suitable retrieval and embedding components from `dbquery` and `dbinsert` rather than duplicating equivalent LanceDB or Ollama functionality. It should add native-tool orchestration and general query workflows around those lower-level capabilities.

## Code-generation principles

### Separation of concerns

The module must keep these responsibilities separate:

```text
Gemma communication
    != tool definitions and registration
    != tool-call parsing
    != tool validation
    != tool execution
    != LanceDB retrieval
    != query state and context
    != workflow control
    != output rendering
```

Gemma may interpret the user's intent, select a registered tool, choose tool arguments, decompose a multi-hop task, and write a grounded answer. Python remains responsible for validation, dispatch, database access, limits, state mutation, persistence, and provenance.

### Classes and files

- Prefer one principal class per file.
- Each principal class must have one specific responsibility.
- Do not create a large general-purpose utility class that combines retrieval, prompting, parsing, state, and rendering.
- Keep orchestration in controller or pipeline classes.
- Keep Gemma prompts and model-response parsing in LLM-facing classes.
- Keep LanceDB and embedding access in retrieval-specific classes.
- Keep tool implementations as bounded adapters over retrieval or analysis services.
- Use explicit dependency injection so classes can be tested with fakes or mocks.
- Keep data structures in small model files. Models may contain related dataclasses, enums, and type aliases, but should not contain workflow logic.
- Avoid circular dependencies between controllers, tools, retrieval, and state packages.
- Prefer narrow interfaces and typed return values over dictionaries passed through many layers.

### File size and complexity

- Aim for no more than 250 lines of code per file.
- Treat 250 lines as a strong warning threshold, not a target to exceed casually.
- If a file grows beyond that size, split it by responsibility before adding more functionality.
- Keep methods short enough that their inputs, side effects, and outputs are clear.
- Move shared parsing, formatting, ranking, and token-budget helpers into focused modules under an appropriate subfolder.
- Avoid hiding substantial behavior in `utils`; helpers must remain narrow and reusable.

### Native tool calling

- Register tools explicitly through a tool registry.
- Expose only tools that the current controller intends to make available.
- Validate tool names and arguments before execution.
- Enforce limits on result counts, tool-call counts, recursion, time, and context size.
- Never allow the model to invoke arbitrary Python functions, shell commands, or unrestricted file/database operations.
- Preserve raw model responses and parsed tool calls in the query trace.
- Distinguish a native tool-call response from a final answer response.
- Tool execution must be deterministic and owned by Python, even when Gemma selects the tool.

### Evidence and provenance

- Every retrieved chunk must retain its `chunk_id` and `paper_id`.
- Every claim used in an answer should be traceable to retrieved evidence.
- Context compaction may shorten text, but must preserve evidence identifiers and provenance metadata.
- If evidence is weak, missing, or conflicting, the final answer must say so explicitly.
- Do not treat model-generated citations or chunk IDs as authoritative when programmatic provenance is available.

### Runtime and persistence

- Keep the Gemma runtime boundary in the LLM package.
- The persistent model process may remain alive across tool-loop turns, but conversational state belongs to the application.
- Persist structured query state separately from raw model messages.
- Make long-running workflows interruptible and inspectable.
- Keep output artifacts useful for debugging and later research review.

## First-workflow folder structure

The first workflow should use the following structure. Add files only when the corresponding capability is needed.

```text
VecQuery/
├── AGENTS.md
├── __init__.py
├── models.py
├── pipeline.py
├── run_query.py
│
├── llm/
│   ├── __init__.py
│   ├── gemma_client.py
│   ├── native_tool_formatter.py
│   ├── tool_call_parser.py
│   ├── response_parser.py
│   └── context_compactor.py
│
├── controllers/
│   ├── __init__.py
│   ├── query_controller.py
│   ├── tool_loop_controller.py
│   └── context_controller.py
│
├── tools/
│   ├── __init__.py
│   ├── tool_definition.py
│   ├── tool_registry.py
│   ├── semantic_search_tool.py
│   ├── exact_search_tool.py
│   ├── filtered_search_tool.py
│   ├── similar_papers_tool.py
│   ├── metadata_aggregation_tool.py
│   └── evidence_tool.py
│
├── retrieval/
│   ├── __init__.py
│   ├── vector_retriever.py
│   ├── full_text_retriever.py
│   ├── hybrid_retriever.py
│   ├── metadata_filter.py
│   └── result_ranker.py
│
├── state/
│   ├── __init__.py
│   ├── query_state_store.py
│   ├── query_context_builder.py
│   ├── evidence_store.py
│   └── claim_store.py
│
├── validation/
│   ├── __init__.py
│   ├── tool_call_validator.py
│   ├── query_limits_validator.py
│   └── result_validator.py
│
└── output/
    ├── __init__.py
    ├── result_writer.py
    └── trace_writer.py
```

### Folder responsibilities

- `llm/`: Owns all communication with Gemma, native tool-call formatting, model-response parsing, runtime errors, and context compaction. It must not execute retrieval tools.
- `controllers/`: Owns workflow decisions and the model/tool loop. Controllers decide which capabilities are available, when to continue, and when to stop.
- `tools/`: Defines and implements the explicitly registered capabilities exposed to Gemma. Each tool should validate its own domain-specific arguments and delegate substantial work to another service.
- `retrieval/`: Owns vector, full-text, hybrid, metadata-filtered, ranking, and related LanceDB operations. It should be usable independently of Gemma.
- `state/`: Owns structured query state, evidence and claim provenance, context construction, persistence, and resumability.
- `validation/`: Owns safety and correctness checks for tool calls, query budgets, result shapes, and other model-produced or external inputs.
- `output/`: Owns human-readable results, machine-readable traces, citations, diagnostics, and saved query artifacts.

The top-level files have narrow roles:

- `models.py`: Shared typed request, response, tool-call, evidence, state, and result models.
- `pipeline.py`: Assembles the components and exposes the main end-to-end workflow.
- `run_query.py`: Thin command-line entry point that parses arguments, constructs dependencies, runs the pipeline, and prints or saves results.

## Core pipeline

The first workflow should follow this sequence:

```text
user question
    ↓
QueryController
    ↓
build initial QueryState and conversation context
    ↓
ToolLoopController
    ↓
GemmaClient with registered tool definitions
    ↓
native tool-call or final-answer response
    ↓
ResponseParser
    ↓
ToolCallValidator and query-limit checks
    ↓
ToolRegistry dispatch
    ↓
selected retrieval or analysis tool
    ↓
retrieval/analysis service
    ↓
ResultValidator
    ↓
QueryState update and provenance recording
    ↓
append tool result to context or compact context when necessary
    ↓
GemmaClient again
    ↓
final grounded answer
    ↓
ResultWriter and TraceWriter
```

Gemma does not execute Python functions directly. It returns a native tool-call request; Python validates and executes the corresponding registered handler, then sends the tool result back into the model conversation.

## Core architectural responsibilities

### 1. GemmaClient: only model communication

`GemmaClient` owns the external or local Gemma runtime, model lifecycle, message transport, timeouts, and runtime errors. It must not know what retrieval tools mean or access LanceDB directly.

### 2. Native tool formatting and parsing

Native tool formatting, tool-call parsing, and final-response parsing belong in dedicated LLM-facing classes. They must distinguish tool calls from ordinary final text and preserve raw model output for tracing.

### 3. ToolRegistry: explicit tool ownership

`ToolRegistry` owns the mapping between registered tool names, tool definitions sent to Gemma, and safe Python handlers. It must reject unknown tools and must not expose arbitrary callables.

### 4. One tool class per capability

Each tool class should implement one bounded capability, such as semantic search, exact search, filtered search, similar-paper lookup, metadata aggregation, or evidence lookup. Tools should not contain the general agent loop or unrelated prompts.

### 5. Retrieval services separate from tools

Tools should delegate to retrieval services. Vector search, FTS, hybrid retrieval, filters, ranking, and paper aggregation belong under `retrieval/`, allowing those services to be tested and reused without Gemma.

### 6. ToolLoopController: the native tool-calling loop

The tool-loop controller sends context and tools to Gemma, parses the response, validates and executes tool calls, appends tool results, and repeats until a final answer or a safety limit is reached. It owns loop limits, not retrieval details.

### 7. QueryController: user-facing workflow control

`QueryController` selects the high-level workflow, available tools, system prompt, response requirements, and limits. Later specialized controllers may support comparison, literature mapping, evidence review, or network queries without overloading one universal controller.

### 8. Query state must be explicit

Structured query state should preserve the original question, current goal, completed and pending steps, tool calls, tool results, retrieved chunks, papers, claims, exclusions, and context summary. Raw conversation messages are not a substitute for structured provenance-aware state.

### 9. Context compaction is a distinct component

`ContextCompactor` should reduce long conversations into a compact state containing active goals, known entities, findings, open questions, next actions, and all relevant evidence identifiers. Compaction must preserve chunk and paper provenance.

### 10. Validation is layered

Use separate validation for tool-call syntax and names, query limits and budgets, and returned result shapes. Model output must be treated as untrusted input, and every tool call must be bounded before execution.

## Initial implementation guidance

The first implementation should support a small, useful workflow:

- semantic search;
- exact/full-text search;
- metadata-filtered search;
- evidence lookup by known chunk IDs;
- native Gemma tool calls;
- a bounded multi-step tool loop;
- grounded final answers;
- saved query traces.

Comparison, similar-paper networks, clustering, temporal analysis, aggregation, and persistent multi-turn research sessions should be added as separate capabilities after this core workflow is stable.

When adding a capability, first decide whether it belongs to:

1. an LLM-facing prompt/parser class;
2. a controller;
3. a registered tool;
4. a retrieval or analysis service;
5. a state/provenance component; or
6. an output writer.

If it appears to belong to more than one category, split the implementation rather than adding mixed responsibilities to an existing file.
