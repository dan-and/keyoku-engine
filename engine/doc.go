// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2026 Keyoku. All rights reserved.

// Package engine implements the core memory processing pipeline.
//
// The engine handles extraction, deduplication, conflict detection, decay,
// and consolidation of memories. It coordinates between the LLM provider
// (for fact extraction), the storage layer (SQLite), and the vector index
// (HNSW) to provide intelligent memory management.
//
// Key components:
//   - Extraction: LLM-powered fact extraction from conversations
//   - Deduplication: semantic and content-hash dedup to prevent redundancy
//   - Conflict detection: identifies contradictory memories and resolves them
//   - Decay: time-based relevance decay with configurable half-life
//   - Entity and relationship tracking: knowledge graph construction
//   - Retrieval: multi-signal ranked search combining vector similarity,
//     recency, access frequency, and importance
package engine
