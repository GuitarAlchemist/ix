# DuckDB IX Opportunities — Schema Sketch

This document drafts a minimal DuckDB schema for IX local analytics. These tables are designed to be populated from the JSONL and Parquet artifacts already emitted by the IX and GA ecosystems, following the "local-first" and "no-cloud" principles.

## Principles

1.  **Local-First**: Data stays on the local machine. Analytics run in-process using DuckDB.
2.  **Schema-on-Read**: DuckDB lenses read existing JSONL/Parquet artifacts.
3.  **No Cloud**: No external database or cloud-based telemetry service is required.
4.  **Privacy**: Secrets and raw private logs (e.g., full conversation history with sensitive data) are explicitly **out of scope**.

## Candidate Tables

### 1. `corpus_files`
Tracks the inventory of files in the IX corpus (voicings, thinking-machine probes, learnings, etc.).

| Column | Type | Description |
|--------|------|-------------|
| `file_path` | VARCHAR | Primary key; relative path to the artifact. |
| `corpus_type` | VARCHAR | Category: `voicing`, `thinking-machine`, `learning`, `contract`. |
| `format` | VARCHAR | `jsonl`, `parquet`, `json`, `md`. |
| `file_size_bytes` | BIGINT | Size on disk. |
| `row_count` | BIGINT | Number of records (for tabular formats). |
| `last_modified` | TIMESTAMP | Last write time. |
| `checksum` | VARCHAR | Content hash (BLAKE3). |

### 2. `exploration_candidates`
Potential paths or data points identified for exploration, optimization, or adversarial testing.

| Column | Type | Description |
|--------|------|-------------|
| `candidate_id` | VARCHAR | Unique identifier for the candidate. |
| `source_id` | VARCHAR | Reference to a corpus file or specific record. |
| `candidate_type` | VARCHAR | e.g., `adversarial_probe`, `edge_case_voicing`. |
| `metadata` | JSON | Domain-specific context. |
| `score` | DOUBLE | Priority or heuristic score. |
| `status` | VARCHAR | `pending`, `explored`, `rejected`. |
| `created_at` | TIMESTAMP | Discovery timestamp. |

### 3. `aiw_episodes`
Tracks "episodes" of AI work, such as Claude Code sessions or automated maintenance loops.

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | VARCHAR | Unique episode ID. |
| `session_id` | VARCHAR | Parent session identifier. |
| `start_time` | TIMESTAMP | Start of the episode. |
| `end_time` | TIMESTAMP | End of the episode. |
| `summary` | VARCHAR | Brief description of the work performed. |
| `verdict_count` | INTEGER | Number of governance/algorithmic verdicts emitted. |
| `token_usage` | INTEGER | Total tokens consumed. |
| `cost_estimate` | DOUBLE | Estimated USD cost. |

### 4. `budget_ledger_entries`
FinOps and budget tracking for AI operations.

| Column | Type | Description |
|--------|------|-------------|
| `entry_id` | UUID | Unique ledger entry ID. |
| `timestamp` | TIMESTAMP | Time of the transaction/usage. |
| `account_id` | VARCHAR | Internal budget account. |
| `provider` | VARCHAR | e.g., `anthropic`, `openai`, `local-llama`. |
| `operation` | VARCHAR | e.g., `sampling`, `embedding`. |
| `amount` | DOUBLE | Transaction amount. |
| `currency` | VARCHAR | `USD`, `credits`. |
| `tags` | JSON | Project, Persona, or Task tags. |

### 5. `trace_events`
Detailed telemetry from tool calls, agent actions, and pipeline execution.

| Column | Type | Description |
|--------|------|-------------|
| `trace_id` | VARCHAR | Unique trace identifier. |
| `span_id` | VARCHAR | Specific operation identifier. |
| `parent_span_id` | VARCHAR | Hierarchy pointer. |
| `event_name` | VARCHAR | e.g., `ix_kmeans_run`, `tool_call`. |
| `timestamp` | TIMESTAMP | Event occurrence. |
| `duration_ms` | DOUBLE | Execution time. |
| `tool_name` | VARCHAR | The name of the IX tool invoked. |
| `input_json` | JSON | Tool arguments. |
| `output_json` | JSON | Tool results. |
| `outcome` | VARCHAR | `success`, `failure`, `timeout`. |

### 6. `vector_benchmarks`
Results of vector search and embedding evaluations (e.g., OPTIC-K search quality).

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | VARCHAR | Unique benchmark run ID. |
| `timestamp` | TIMESTAMP | Run time. |
| `model_name` | VARCHAR | e.g., `bge-large-en-v1.5`. |
| `dimension` | INTEGER | Embedding size. |
| `metric` | VARCHAR | `cosine`, `euclidean`, `dot_product`. |
| `k` | INTEGER | Top-K depth evaluated. |
| `recall_at_k` | DOUBLE | Evaluation metric. |
| `latency_ms_p50` | DOUBLE | Median latency. |
| `latency_ms_p95` | DOUBLE | Tail latency. |

### 7. `algorithm_runs`
Telemetry from individual executions of IX algorithms.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | VARCHAR | Unique execution ID. |
| `algorithm_name` | VARCHAR | e.g., `pso`, `dbscan`, `viterbi`. |
| `parameters` | JSON | Hyperparameters used. |
| `input_size` | BIGINT | Number of input data points/dimensions. |
| `duration_ms` | DOUBLE | Total execution time. |
| `result_summary` | JSON | High-level results (e.g., loss, cluster count). |
| `timestamp` | TIMESTAMP | Execution start time. |

## Query Opportunities

With these tables, we can answer questions such as:

*   **Yield Trends**: "What is the average yield (coverage_max) for `thinking-machine` episodes before and after a specific algorithm update?"
*   **Cost Anomaly Detection**: "Which tool calls in the last 24 hours exceeded the average `cost_estimate` by 2x?"
*   **Recall Drift**: "Has the `recall_at_10` for voicing embeddings regressed since the last corpus update?"
*   **Failure Clustering**: "What are the most frequent `outcome = 'failure'` tool calls, and do they share common input patterns?"
