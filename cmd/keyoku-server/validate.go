// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2026 Keyoku. All rights reserved.

package main

import (
	"fmt"
	"regexp"
	"unicode/utf8"
)

// --- Validation constants ---

const maxBodySize = 1 << 20 // 1MB
const maxContentLength = 50000
const maxLimit = 1000
const maxIDLength = 256

// validID matches safe identifier characters (alphanumeric, hyphens, underscores, colons, dots).
var validID = regexp.MustCompile(`^[a-zA-Z0-9_:.\-]+$`)

// --- Validation functions ---

// validateMemoryContent checks that content is non-empty, within length bounds,
// and valid UTF-8.
func validateMemoryContent(content string) error {
	if content == "" {
		return fmt.Errorf("content is required")
	}
	if len(content) > maxContentLength {
		return fmt.Errorf("content too large (max %d chars)", maxContentLength)
	}
	if !utf8.ValidString(content) {
		return fmt.Errorf("content contains invalid UTF-8 encoding")
	}
	return nil
}

// validateEntityID checks that an entity ID is well-formed.
func validateEntityID(id string) error {
	if id == "" {
		return fmt.Errorf("entity_id is required")
	}
	if len(id) > maxIDLength {
		return fmt.Errorf("entity_id too long (max %d chars)", maxIDLength)
	}
	if !validID.MatchString(id) {
		return fmt.Errorf("entity_id contains invalid characters")
	}
	return nil
}

// validateAgentID checks that an agent ID is well-formed (same rules as entity ID).
func validateAgentID(id string) error {
	if id == "" {
		return fmt.Errorf("agent_id is required")
	}
	if len(id) > maxIDLength {
		return fmt.Errorf("agent_id too long (max %d chars)", maxIDLength)
	}
	if !validID.MatchString(id) {
		return fmt.Errorf("agent_id contains invalid characters")
	}
	return nil
}

// validateOptionalID checks an ID only if it is non-empty (for optional fields).
func validateOptionalID(id, fieldName string) error {
	if id == "" {
		return nil
	}
	if len(id) > maxIDLength {
		return fmt.Errorf("%s too long (max %d chars)", fieldName, maxIDLength)
	}
	if !validID.MatchString(id) {
		return fmt.Errorf("%s contains invalid characters", fieldName)
	}
	return nil
}
