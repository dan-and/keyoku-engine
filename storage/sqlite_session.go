// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2026 Keyoku. All rights reserved.

package storage

import (
	"context"
	"time"

	"github.com/oklog/ulid/v2"
)

// --- Session Messages ---

func (s *SQLiteStore) AddSessionMessage(ctx context.Context, msg *SessionMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if msg.ID == "" {
		msg.ID = ulid.Make().String()
	}
	now := time.Now().UTC().Format(time.RFC3339)

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO session_messages (id, entity_id, agent_id, session_id, role, content, turn_number, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		msg.ID, msg.EntityID, msg.AgentID, msg.SessionID, msg.Role, msg.Content, msg.TurnNumber, now)
	return err
}

func (s *SQLiteStore) GetRecentSessionMessages(ctx context.Context, entityID string, limit int) ([]*SessionMessage, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, entity_id, agent_id, session_id, role, content, turn_number, created_at
		FROM session_messages WHERE entity_id = ?
		ORDER BY created_at DESC LIMIT ?`, entityID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []*SessionMessage
	for rows.Next() {
		var m SessionMessage
		var createdStr string
		if err := rows.Scan(&m.ID, &m.EntityID, &m.AgentID, &m.SessionID, &m.Role, &m.Content, &m.TurnNumber, &createdStr); err != nil {
			return nil, err
		}
		m.CreatedAt, _ = time.Parse(time.RFC3339, createdStr)
		messages = append(messages, &m)
	}
	return messages, nil
}
