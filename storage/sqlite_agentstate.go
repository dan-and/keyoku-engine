// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2026 Keyoku. All rights reserved.

package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"time"

	"github.com/oklog/ulid/v2"
)

// --- Agent State CRUD ---

func (s *SQLiteStore) CreateAgentState(ctx context.Context, state *AgentState) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if state.ID == "" {
		state.ID = ulid.Make().String()
	}
	now := time.Now().UTC()
	state.CreatedAt = now
	state.LastUpdatedAt = &now

	currentStateJSON, _ := json.Marshal(state.CurrentState)
	schemaJSON, _ := json.Marshal(state.SchemaDefinition)
	rulesJSON, _ := json.Marshal(state.TransitionRules)

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO agent_states (id, entity_id, agent_id, schema_name, current_state, schema_definition, transition_rules, last_updated_at, created_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		state.ID, state.EntityID, state.AgentID, state.SchemaName,
		string(currentStateJSON), string(schemaJSON), string(rulesJSON),
		now.Format(time.RFC3339), now.Format(time.RFC3339),
	)
	return err
}

func (s *SQLiteStore) GetAgentState(ctx context.Context, entityID, agentID, schemaName string) (*AgentState, error) {
	var st AgentState
	var currentStateStr, schemaStr, rulesStr string
	var lastUpdatedStr, createdStr string

	err := s.db.QueryRowContext(ctx,
		`SELECT id, entity_id, agent_id, schema_name, current_state, schema_definition, transition_rules, last_updated_at, created_at
		 FROM agent_states WHERE entity_id = ? AND agent_id = ? AND schema_name = ?`,
		entityID, agentID, schemaName,
	).Scan(&st.ID, &st.EntityID, &st.AgentID, &st.SchemaName,
		&currentStateStr, &schemaStr, &rulesStr,
		&lastUpdatedStr, &createdStr,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	json.Unmarshal([]byte(currentStateStr), &st.CurrentState)
	json.Unmarshal([]byte(schemaStr), &st.SchemaDefinition)
	json.Unmarshal([]byte(rulesStr), &st.TransitionRules)

	if t, err := time.Parse(time.RFC3339, lastUpdatedStr); err == nil {
		st.LastUpdatedAt = &t
	}
	if t, err := time.Parse(time.RFC3339, createdStr); err == nil {
		st.CreatedAt = t
	}

	return &st, nil
}

func (s *SQLiteStore) UpdateAgentState(ctx context.Context, id string, newState map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	stateJSON, _ := json.Marshal(newState)
	now := time.Now().UTC().Format(time.RFC3339)

	_, err := s.db.ExecContext(ctx,
		`UPDATE agent_states SET current_state = ?, last_updated_at = ? WHERE id = ?`,
		string(stateJSON), now, id,
	)
	return err
}

func (s *SQLiteStore) GetAgentStateHistory(ctx context.Context, stateID string, limit int) ([]*AgentStateHistory, error) {
	if limit <= 0 {
		limit = 50
	}

	rows, err := s.db.QueryContext(ctx,
		`SELECT id, state_id, previous_state, new_state, changed_fields, trigger_content, confidence, reasoning, created_at
		 FROM agent_state_history WHERE state_id = ? ORDER BY created_at DESC LIMIT ?`,
		stateID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var entries []*AgentStateHistory
	for rows.Next() {
		var h AgentStateHistory
		var prevStr, newStr, fieldsStr, createdStr string
		if err := rows.Scan(&h.ID, &h.StateID, &prevStr, &newStr, &fieldsStr,
			&h.TriggerContent, &h.Confidence, &h.Reasoning, &createdStr); err != nil {
			return nil, err
		}
		json.Unmarshal([]byte(prevStr), &h.PreviousState)
		json.Unmarshal([]byte(newStr), &h.NewState)
		h.ChangedFields.Scan(fieldsStr)
		if t, err := time.Parse(time.RFC3339, createdStr); err == nil {
			h.CreatedAt = t
		}
		entries = append(entries, &h)
	}
	return entries, nil
}

func (s *SQLiteStore) LogAgentStateHistory(ctx context.Context, entry *AgentStateHistory) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if entry.ID == "" {
		entry.ID = ulid.Make().String()
	}
	now := time.Now().UTC()
	entry.CreatedAt = now

	prevJSON, _ := json.Marshal(entry.PreviousState)
	newJSON, _ := json.Marshal(entry.NewState)
	fieldsJSON, _ := entry.ChangedFields.Value()

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO agent_state_history (id, state_id, previous_state, new_state, changed_fields, trigger_content, confidence, reasoning, created_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		entry.ID, entry.StateID, string(prevJSON), string(newJSON), fieldsJSON,
		entry.TriggerContent, entry.Confidence, entry.Reasoning, now.Format(time.RFC3339),
	)
	return err
}
