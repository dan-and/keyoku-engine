// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2026 Keyoku. All rights reserved.

// Package llm provides the LLM provider abstraction for memory extraction.
//
// Keyoku uses LLMs to extract structured facts, entities, and relationships
// from conversation text. This package defines the Provider interface and
// includes implementations for OpenAI, Anthropic, and Google Gemini.
//
// Each provider handles prompt construction, API communication, and response
// parsing into standardized ExtractionResponse structs. Provider selection
// is automatic based on which API key is configured.
package llm
