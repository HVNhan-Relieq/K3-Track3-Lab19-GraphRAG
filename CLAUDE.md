# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A student lab assignment (AICB-K34, Track 3, Day 19), not an application. The whole pipeline lives in `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb` — 36 cells, rewritten from the instructor's guide to actually run end-to-end and executed with real outputs. No package, no entrypoint script, no test suite (`pytest` is in `requirements.txt` but no tests exist); verification happens through in-notebook asserts.

Docs (`README.md`, `ASSIGNMENT.md`, `RUBRIC.md`, report templates) are in Vietnamese — keep new prose Vietnamese. `ASSIGNMENT.md` + `RUBRIC.md` are the spec; when changing notebook logic, check the rubric row it maps to (the mapping table is in the notebook's final markdown cell).

## Commands

```bash
pip install -r requirements.txt
py -m pip install jupyter nbclient ipykernel     # to execute the notebook headlessly
jupyter lab Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb
```

Windows notes: use `py`, not `python` (the `python` on PATH is the Microsoft Store stub). Prefix any script that prints notebook content with `PYTHONIOENCODING=utf-8` — the console codepage is cp1258 and the content is Vietnamese.

Headless execution (what produced the committed outputs) uses a cell-by-cell runner kept in the session scratchpad, not in the repo: it runs `nbclient` with `stop_after` so a long LLM stage can be run and inspected in isolation. `jupyter nbconvert --execute` works too but re-runs everything.

**Re-running is cheap because the expensive stages cache to `outputs/cache/`** (`coref.csv`, `raw_triples.csv`, `graphrag_eval_checkpoint.csv`). Delete a cache file to force that stage to hit the LLM again. The eval checkpoint resumes per question.

## Live services this notebook needs

- **Neo4j 5.x** at `NEO4J_URI`. The AuraDB instance in the original `.env` is dead. This machine runs Neo4j 5.26 Community natively (JDK 21 at `C:\Program Files\Microsoft\jdk-21.0.12.8-hotspot`, server unzipped under the session scratchpad, `bolt://localhost:7687`, user `neo4j`). Docker is installed but **its published ports are unreachable from the host** — WSL2 is in `networkingMode=mirrored` and Docker Desktop's port proxy binds 7687 without forwarding it, which also silently blocks native Neo4j from binding that port. If a container is running, remove it before starting native Neo4j.
- **Groq** for coreference, extraction, seeds, and generation. `llama-3.3-70b-versatile` is retired; use `openai/gpt-oss-120b`. Free tier is **8.000 TPM / 1.000 requests per day** — the TPM ceiling, not latency, sets the runtime (coref over 700 chunks took ~25 min).
- **Judge** via OpenRouter (`OPENAI_BASE_URL=https://openrouter.ai/api/v1`). The account has zero credits, so only `:free` models work (`nvidia/nemotron-3-super-120b-a12b:free`, fallback `google/gemma-4-31b-it:free`), capped around 50 requests/day.
- **Hugging Face**: the dataset is `gated: auto` — the account must click "Agree and access repository" once, or `load_dataset` raises `DatasetNotFoundError` even with a valid token.

`load_dotenv` is called with `override=True` on purpose: this machine has a system-wide `OPENAI_API_KEY` holding an NVIDIA NIM key that otherwise shadows the repo's `.env`.

## Data facts that drive the design

- The HackerNoon dump has **no article body** — only `title` + `description` (~32 words). Text unit is `title + ". " + description`, so one article ≈ one chunk and multi-hop links must cross snippets rather than travel within a long document.
- `data/hackernoon_subset.csv` is the **first 5000 rows of split `train`, in stream order** (gitignored). The golden dataset references evidence by 0-based row index into exactly that window, so any other stopping rule (e.g. the original "stop at 300 MB") misaligns every `reference_evidence`. Cell 1.5 asserts row 33 is the Aeris/Ericsson article dated 2022-12-07 13:45.
- Golden data is `data/graphrag_golden_50_first5000.csv` (50 questions, all `reference_answer` filled) plus a `_detailed` variant carrying `evidence_row_ids_0based`, `expected_hops`, `seed_entities`. The notebook's original 5 starter questions are unused.
- Only ~2.100 of the 5.000 rows survive (46% have an empty `description`, then exact dedup removes ~570).

## Invariants the grader checks

- **Bulk insert only**: `UNWIND $rows AS row`, batches of 1000. Per-row `CREATE`/`MERGE` is forbidden by `ASSIGNMENT.md`.
- **Edge provenance is total**: every relationship carries `source_chunk_id` + `published_date`; the Cypher check must return 0, asserted in cells 2.4 and 5.1.
- **Super-node caps**: `SUPER_NODE_DEGREE=100`, `SUPER_NODE_EDGE_CAP=50` (newest first), `GLOBAL_EDGE_CAP=250`, `MAX_GRAPH_CONTEXT_CHARS=14000`. The lab graph may have no node above degree 100 — cell 5.1 then lowers the threshold to the real p95 and says so, rather than claiming a super-node it doesn't have.
- **Schema allowlist**: nodes `Company|Person|Technology` (+ `Entity`); relations `ACQUIRED, DEVELOPED, INVESTED_IN, FOUNDED, WORKED_AT, PARTNERED_WITH, USES, LEADS`. Relation/label strings are interpolated into Cypher (they cannot be parameterised), so the allowlist is a security control, not just schema hygiene — drop unknown values, never widen.
- **Entity resolution audit** classifies each decision `MERGE_MANUAL` / `MERGE_VECTOR` / `REJECT_GUARD` with a `reason`; cosine threshold 0.90, plus five lexical guards with a unit test over `Sam Altman`/`Steve Altman`, `Apple`/`Apple Watch`, `GPT-4`/`GPT-3`.
- `SEED = 42` throughout, and `reset_graph()` runs before ingestion so re-running the notebook doesn't double-count.

## Report deliverable

`README.md` and the first deliverables block in `ASSIGNMENT.md` require a single `reports/lab_report.md` (matching `templates/lab_report.md`); a later block in `ASSIGNMENT.md` and `RUBRIC.md` §4 instead name three files. The single-file form is what ships and what is filled in.
