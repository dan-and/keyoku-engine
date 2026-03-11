// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2025 Keyoku. All rights reserved.

// Package storage provides the persistence layer for Keyoku.
//
// The primary implementation is a pure-Go SQLite backend that stores
// memories, entities, relationships, and event history. It uses
// WAL mode for concurrent read/write access and includes built-in
// vector storage for HNSW index reconstruction.
//
// The Store interface defines all storage operations:
//   - Memory CRUD and batch operations
//   - Vector similarity search (delegated to HNSW)
//   - Entity and relationship management
//   - Schedule and event tracking
//   - Session message history
package storage
