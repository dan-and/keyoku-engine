// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2026 Keyoku. All rights reserved.

package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/keyoku-ai/keyoku-engine/vectorindex"
	"github.com/oklog/ulid/v2"
)

// allowedEntityColumns restricts UPDATE SET columns for entities.
var allowedEntityColumns = map[string]bool{
	"name": true, "description": true, "entity_type": true, "metadata": true,
}

// --- Entity CRUD ---

func (s *SQLiteStore) CreateEntity(ctx context.Context, entity *Entity) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if entity.ID == "" {
		entity.ID = ulid.Make().String()
	}
	now := time.Now().UTC().Format(time.RFC3339)
	aliases, _ := entity.Aliases.Value()
	attrs, _ := entity.Attributes.Value()

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO entities (id, owner_entity_id, agent_id, canonical_name, type, description,
			aliases, embedding, attributes, mention_count, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		entity.ID, entity.OwnerEntityID, entity.AgentID, entity.CanonicalName,
		string(entity.Type), entity.Description, aliases, entity.Embedding,
		attrs, entity.MentionCount, now, now)
	return err
}

func (s *SQLiteStore) GetEntity(ctx context.Context, id string) (*Entity, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, owner_entity_id, agent_id, canonical_name, type, description,
			aliases, embedding, attributes, mention_count, last_mentioned_at, created_at, updated_at
		FROM entities WHERE id = ?`, id)
	return scanEntity(row)
}

func (s *SQLiteStore) GetEntityByName(ctx context.Context, ownerEntityID, name string, entityType EntityType) (*Entity, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, owner_entity_id, agent_id, canonical_name, type, description,
			aliases, embedding, attributes, mention_count, last_mentioned_at, created_at, updated_at
		FROM entities WHERE owner_entity_id = ? AND canonical_name = ? AND type = ?`,
		ownerEntityID, name, string(entityType))
	entity, err := scanEntity(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return entity, err
}

func (s *SQLiteStore) FindEntityByAlias(ctx context.Context, ownerEntityID, alias string) (*Entity, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, owner_entity_id, agent_id, canonical_name, type, description,
			aliases, embedding, attributes, mention_count, last_mentioned_at, created_at, updated_at
		FROM entities WHERE owner_entity_id = ? AND canonical_name = ?`,
		ownerEntityID, alias)
	entity, err := scanEntity(row)
	if err == nil {
		return entity, nil
	}

	rows, err := s.db.QueryContext(ctx,
		`SELECT id, owner_entity_id, agent_id, canonical_name, type, description,
			aliases, embedding, attributes, mention_count, last_mentioned_at, created_at, updated_at
		FROM entities WHERE owner_entity_id = ?`, ownerEntityID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		e, err := scanEntityFromRows(rows)
		if err != nil {
			continue
		}
		for _, a := range e.Aliases {
			if strings.EqualFold(a, alias) {
				return e, nil
			}
		}
	}
	return nil, nil
}

func (s *SQLiteStore) FindSimilarEntities(ctx context.Context, embedding []float32, ownerEntityID string, limit int, minScore float64) ([]*Entity, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, owner_entity_id, agent_id, canonical_name, type, description,
			aliases, embedding, attributes, mention_count, last_mentioned_at, created_at, updated_at
		FROM entities WHERE owner_entity_id = ? AND embedding IS NOT NULL`,
		ownerEntityID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type scored struct {
		entity *Entity
		sim    float64
	}
	var candidates []scored

	for rows.Next() {
		e, err := scanEntityFromRows(rows)
		if err != nil {
			continue
		}
		entityVec := decodeEmbedding(e.Embedding)
		if len(entityVec) == 0 || len(entityVec) != len(embedding) {
			continue
		}
		sim := float64(vectorindex.CosineSimilarity(embedding, entityVec))
		if sim >= minScore {
			candidates = append(candidates, scored{entity: e, sim: sim})
		}
	}

	for i := 1; i < len(candidates); i++ {
		for j := i; j > 0 && candidates[j].sim > candidates[j-1].sim; j-- {
			candidates[j], candidates[j-1] = candidates[j-1], candidates[j]
		}
	}

	if len(candidates) > limit {
		candidates = candidates[:limit]
	}

	result := make([]*Entity, len(candidates))
	for i, c := range candidates {
		result[i] = c.entity
	}
	return result, nil
}

func (s *SQLiteStore) QueryEntities(ctx context.Context, query EntityQuery) ([]*Entity, error) {
	var where []string
	var args []any

	if query.OwnerEntityID != "" {
		where = append(where, "owner_entity_id = ?")
		args = append(args, query.OwnerEntityID)
	}
	if query.AgentID != "" {
		where = append(where, "agent_id = ?")
		args = append(args, query.AgentID)
	}
	if len(query.Types) > 0 {
		ph := make([]string, len(query.Types))
		for i, t := range query.Types {
			ph[i] = "?"
			args = append(args, string(t))
		}
		where = append(where, fmt.Sprintf("type IN (%s)", strings.Join(ph, ",")))
	}

	whereClause := ""
	if len(where) > 0 {
		whereClause = "WHERE " + strings.Join(where, " AND ")
	}

	limit := query.Limit
	if limit <= 0 {
		limit = 100
	}

	q := fmt.Sprintf(
		`SELECT id, owner_entity_id, agent_id, canonical_name, type, description,
			aliases, embedding, attributes, mention_count, last_mentioned_at, created_at, updated_at
		FROM entities %s ORDER BY mention_count DESC LIMIT ? OFFSET ?`,
		whereClause)
	args = append(args, limit, query.Offset)

	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var entities []*Entity
	for rows.Next() {
		e, err := scanEntityFromRows(rows)
		if err != nil {
			return nil, err
		}
		entities = append(entities, e)
	}
	return entities, nil
}

func (s *SQLiteStore) UpdateEntity(ctx context.Context, id string, updates map[string]any) (*Entity, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var setClauses []string
	var args []any

	for k, v := range updates {
		if !allowedEntityColumns[k] {
			return nil, fmt.Errorf("invalid entity update column: %q", k)
		}
		setClauses = append(setClauses, k+" = ?")
		args = append(args, v)
	}
	setClauses = append(setClauses, "updated_at = ?")
	args = append(args, time.Now().UTC().Format(time.RFC3339))
	args = append(args, id)

	_, err := s.db.ExecContext(ctx,
		fmt.Sprintf("UPDATE entities SET %s WHERE id = ?", strings.Join(setClauses, ", ")),
		args...)
	if err != nil {
		return nil, err
	}
	return s.GetEntity(ctx, id)
}

func (s *SQLiteStore) UpdateEntityMentionCount(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.db.ExecContext(ctx,
		"UPDATE entities SET mention_count = mention_count + 1, last_mentioned_at = ?, updated_at = ? WHERE id = ?",
		now, now, id)
	return err
}

func (s *SQLiteStore) AddEntityAlias(ctx context.Context, id, alias string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var aliasesStr string
	err := s.db.QueryRowContext(ctx, "SELECT aliases FROM entities WHERE id = ?", id).Scan(&aliasesStr)
	if err != nil {
		return err
	}

	var aliases []string
	json.Unmarshal([]byte(aliasesStr), &aliases)

	for _, a := range aliases {
		if a == alias {
			return nil
		}
	}

	aliases = append(aliases, alias)
	newAliases, _ := json.Marshal(aliases)

	_, err = s.db.ExecContext(ctx,
		"UPDATE entities SET aliases = ?, updated_at = ? WHERE id = ?",
		string(newAliases), time.Now().UTC().Format(time.RFC3339), id)
	return err
}

func (s *SQLiteStore) DeleteEntity(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(ctx, "DELETE FROM entities WHERE id = ?", id)
	return err
}

func (s *SQLiteStore) DeleteAllEntitiesForOwner(ctx context.Context, ownerEntityID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	result, err := s.db.ExecContext(ctx,
		"DELETE FROM entities WHERE owner_entity_id = ?", ownerEntityID)
	if err != nil {
		return 0, err
	}
	n, _ := result.RowsAffected()
	return int(n), nil
}

func (s *SQLiteStore) CreateEntityMention(ctx context.Context, mention *EntityMention) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if mention.ID == "" {
		mention.ID = ulid.Make().String()
	}
	now := time.Now().UTC().Format(time.RFC3339)

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO entity_mentions (id, entity_id, memory_id, mention_text, confidence, context_snippet, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)`,
		mention.ID, mention.EntityID, mention.MemoryID, mention.MentionText,
		mention.Confidence, mention.ContextSnippet, now)
	return err
}

func (s *SQLiteStore) GetEntityMentions(ctx context.Context, entityID string, limit int) ([]*EntityMention, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, entity_id, memory_id, mention_text, confidence, context_snippet, created_at
		FROM entity_mentions WHERE entity_id = ? ORDER BY created_at DESC LIMIT ?`,
		entityID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var mentions []*EntityMention
	for rows.Next() {
		var m EntityMention
		var createdStr string
		if err := rows.Scan(&m.ID, &m.EntityID, &m.MemoryID, &m.MentionText,
			&m.Confidence, &m.ContextSnippet, &createdStr); err != nil {
			return nil, err
		}
		m.CreatedAt, _ = time.Parse(time.RFC3339, createdStr)
		mentions = append(mentions, &m)
	}
	return mentions, nil
}

func (s *SQLiteStore) GetMemoryEntities(ctx context.Context, memoryID string) ([]*Entity, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT e.id, e.owner_entity_id, e.agent_id, e.canonical_name, e.type, e.description,
			e.aliases, e.embedding, e.attributes, e.mention_count, e.last_mentioned_at, e.created_at, e.updated_at
		FROM entities e
		JOIN entity_mentions em ON em.entity_id = e.id
		WHERE em.memory_id = ?`, memoryID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var entities []*Entity
	for rows.Next() {
		e, err := scanEntityFromRows(rows)
		if err != nil {
			return nil, err
		}
		entities = append(entities, e)
	}
	return entities, nil
}
