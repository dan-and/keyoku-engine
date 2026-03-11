// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2025 Keyoku. All rights reserved.

// Package jobs provides background job scheduling for memory maintenance.
//
// Keyoku runs periodic background jobs to keep the memory store healthy:
//   - Decay: reduces memory relevance scores over time based on configurable half-life
//   - Consolidation: merges similar memories to reduce redundancy
//   - Archival: moves low-relevance memories to cold storage
//   - Eviction: removes memories that fall below minimum thresholds
//   - Purge: permanently deletes soft-deleted memories after retention period
//
// Jobs are scheduled using in-memory timers and run within the engine process.
package jobs
