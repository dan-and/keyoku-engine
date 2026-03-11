# Architecture

This document describes the internal architecture of keyoku-engine.

## Overview

```
                    ┌─────────────────────────┐
                    │    HTTP API Server       │
                    │  cmd/keyoku-server/      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Keyoku (root package)  │
                    │   keyoku.go              │
                    │   heartbeat.go           │
                    │   schedule.go            │
                    │   events.go              │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐ ┌─────────▼─────────┐ ┌─────────▼─────────┐
│      engine/      │ │      llm/         │ │     embedder/     │
│  Extract, dedup,  │ │  OpenAI, Gemini,  │ │  Text → vector    │
│  conflict, decay  │ │  Anthropic        │ │  (OpenAI, noop)   │
└─────────┬─────────┘ └───────────────────┘ └─────────┬─────────┘
          │                                           │
          │              ┌────────────────┐           │
          └──────────────▶   storage/     ◀───────────┘
                         │  SQLite (WAL)  │
                         └────────┬───────┘
                                  │
                         ┌────────▼───────┐
                         │  vectorindex/  │
                         │  HNSW index    │
                         └────────────────┘
```

## Package Responsibilities

### Root package (`keyoku`)

The public API surface. All external consumers import this package.

- **keyoku.go** — `Keyoku` struct: the main entry point. Wraps engine, embedder, LLM, storage, and scheduler. Exposes `Remember()`, `Search()`, `ListMemories()`, `DeleteMemory()`, etc.
- **heartbeat.go** — `HeartbeatCheck()` (zero-token local query) and `HeartbeatContext()` (combined heartbeat + semantic search + optional LLM analysis).
- **schedule.go** — Cron-tagged memory scheduling with acknowledgment tracking.
- **events.go** — Server-Sent Events (SSE) for real-time memory change notifications.
- **config.go** — `Config` struct with all knobs and `DefaultConfig()`.
- **watcher.go** — File watcher for hot-reloading configuration.
- **patterns.go** — Regex patterns for date/time extraction from natural language.

### `engine/`

Core memory processing pipeline:

- **engine.go** — `Engine` struct coordinating extraction → dedup → conflict → store.
- **dedup.go** — Semantic deduplication (cosine similarity) and content-hash dedup.
- **conflict.go** — Detects contradictory memories and resolves via recency/confidence.
- **decay.go** — Time-based relevance decay with configurable half-life.
- **retrieval.go** — Multi-signal ranked retrieval combining vector similarity, recency, access frequency, and importance.
- **ranker.go** — Scoring functions and retrieval mode presets (balanced, recent, important, historical, comprehensive).
- **scorer.go** — Composite scoring with configurable signal weights.
- **entity.go** — Entity extraction and resolution (canonical names, aliases, types).
- **relationship.go** — Relationship detection between entities with confidence tracking.
- **graph.go** — Knowledge graph construction from entities and relationships.
- **budget.go** — Token budget management for LLM extraction calls.
- **query_expand.go** — Query expansion for improved recall.

### `storage/`

Persistence layer:

- **interface.go** — `Store` interface defining all storage operations.
- **sqlite.go** — Pure-Go SQLite implementation (WAL mode, no CGO).
- **models.go** — Data models: `Memory`, `Entity`, `Relationship`, `Team`, etc.
- **json_types.go** — JSON serialization helpers for SQLite columns.
- **errors.go** — Typed storage errors.

### `llm/`

LLM provider abstraction:

- **provider.go** — `Provider` interface and auto-detection logic.
- **openai.go** — OpenAI implementation (GPT-4o-mini default).
- **anthropic.go** — Anthropic/Claude implementation.
- **gemini.go** — Google Gemini implementation.
- **types.go** — `ExtractionResponse`, `ExtractedMemory`, `ExtractedEntity`, etc.
- **validate.go** — Response validation and sanitization.

### `embedder/`

Text-to-vector embedding:

- **embedder.go** — `Embedder` interface.
- **openai.go** — OpenAI text-embedding-3-small (default).
- **noop.go** — No-op embedder for testing.

### `vectorindex/`

In-process similarity search:

- **index.go** — `VectorIndex` interface.
- **hnsw.go** — HNSW (Hierarchical Navigable Small World) implementation.
- **math.go** — Cosine similarity and vector operations.

### `cache/`

Performance optimization:

- **lru.go** — Hot LRU cache for Tier 1 memory retrieval. Stores recently accessed memories with decoded embeddings for sub-millisecond brute-force cosine search.

### `jobs/`

Background maintenance:

- **scheduler.go** — In-memory job scheduler with configurable intervals.
- **decay_processor.go** — Applies time-based decay to memory relevance scores.
- **consolidation_processor.go** — Merges similar memories using LLM.
- **archival_processor.go** — Moves low-relevance memories to cold storage.
- **eviction_processor.go** — Removes memories below minimum thresholds.
- **purge_processor.go** — Permanently deletes soft-deleted memories after retention period.

### `cmd/keyoku-server/`

HTTP server binary:

- **main.go** — Server startup, signal handling, graceful shutdown.
- **handlers.go** — HTTP handlers for all API endpoints.
- **config.go** — Server-specific config (port, CORS, session token).
- **sse.go** — SSE hub for real-time event streaming.

## Data Flow

### Remember (Store Memories)

```
Client POST /api/remember
    │
    ▼
Keyoku.Remember(entityId, content)
    │
    ├─▶ LLM.Extract(content)         → ExtractedMemory[]
    │
    ├─▶ Embedder.Embed(memory.content) → []float32
    │
    ├─▶ Engine.Deduplicate(memory)     → skip if duplicate
    │
    ├─▶ Engine.DetectConflicts(memory) → resolve or flag
    │
    ├─▶ Storage.CreateMemory(memory)   → SQLite
    │
    └─▶ VectorIndex.Add(id, embedding) → HNSW
```

### Search (Recall Memories)

```
Client POST /api/search
    │
    ▼
Keyoku.Search(entityId, query)
    │
    ├─▶ Cache.Search(query)            → hit? return immediately
    │
    ├─▶ Embedder.Embed(query)          → []float32
    │
    ├─▶ VectorIndex.Search(embedding)  → candidate IDs
    │
    ├─▶ Storage.GetMemoriesByIDs(ids)  → Memory[]
    │
    ├─▶ Ranker.Score(memories, query)  → ranked results
    │
    └─▶ Cache.Put(results)             → warm cache
```

### Heartbeat Check (Zero-Token)

```
Client POST /api/heartbeat/check
    │
    ▼
Keyoku.HeartbeatCheck(entityId)
    │
    ├─▶ Storage queries (no LLM, no embeddings):
    │     • Pending work (state = pending)
    │     • Deadlines (expires_at within window)
    │     • Scheduled (cron-tagged, unacknowledged)
    │     • Decaying (relevance below threshold)
    │     • Conflicts (unresolved)
    │     • Stale monitors (not accessed recently)
    │
    └─▶ Return HeartbeatCheckResponse
         { should_act, pending_work[], deadlines[], ... }
```

## Storage Schema

SQLite with WAL mode. Key tables:

| Table | Purpose |
|-------|---------|
| `memories` | Core memory storage with content, type, state, importance, confidence, sentiment, tags, embedding (blob) |
| `entities` | Extracted entities with canonical names, types, aliases |
| `relationships` | Entity-to-entity relationships with type, confidence, evidence count |
| `relationship_evidence` | Individual evidence items supporting relationships |
| `entity_mentions` | Links between memories and entities they mention |
| `event_history` | Audit log of all memory operations |
| `session_messages` | Conversation history for context-aware extraction |
| `teams` | Team definitions for multi-agent visibility |
| `team_members` | Agent membership in teams |

## Configuration

See `Config` in [config.go](../config.go) for all options. Key tuning parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SemanticDuplicateThresh` | 0.95 | Cosine similarity above which memories are considered duplicates |
| `NearDuplicateThresh` | 0.85 | Threshold for near-duplicate detection (triggers merge) |
| `ConflictSimilarityThresh` | 0.6 | Similarity threshold for conflict detection |
| `DecayThreshold` | 0.3 | Relevance score below which memories are flagged as decaying |
| `ArchivalDays` | 30 | Days before low-relevance memories are archived |
| `PurgeRetentionDays` | 90 | Days before soft-deleted memories are permanently purged |
