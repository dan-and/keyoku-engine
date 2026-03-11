// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2025 Keyoku. All rights reserved.

// Package embedder provides text-to-vector embedding for semantic search.
//
// The Embedder interface converts text into dense vector representations
// used by the HNSW index for similarity search. Implementations include
// OpenAI's text-embedding models and a no-op embedder for testing.
//
// Embeddings are generated during memory storage and query time to enable
// semantic recall — finding memories by meaning rather than keyword match.
package embedder
