// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2025 Keyoku. All rights reserved.

// Package vectorindex provides in-process vector similarity search.
//
// The primary implementation is an HNSW (Hierarchical Navigable Small World)
// index that runs entirely in-process with no external dependencies. It
// supports efficient approximate nearest-neighbor search over high-dimensional
// embedding vectors.
//
// The index is reconstructed from stored embeddings on startup and updated
// incrementally as memories are added or removed.
package vectorindex
