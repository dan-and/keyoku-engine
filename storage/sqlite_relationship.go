// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2026 Keyoku. All rights reserved.

package storage

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	"github.com/oklog/ulid/v2"
)

// allowedRelationshipColumns restricts UPDATE SET columns for relationships.
var allowedRelationshipColumns = map[string]bool{
	"strength": true, "description": true, "relationship_type": true, "metadata": true,
}

// --- Relationship CRUD ---

func (s *SQLiteStore) CreateRelationship(ctx context.Context, rel *Relationship) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if rel.ID == "" {
		rel.ID = ulid.Make().String()
	}
	now := time.Now().UTC().Format(time.RFC3339)
	attrs, _ := rel.Attributes.Value()
	bidir := 0
	if rel.IsBidirectional {
		bidir = 1
	}

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO relationships (id, owner_entity_id, agent_id, source_entity_id, target_entity_id,
			relationship_type, description, strength, confidence, is_bidirectional,
			evidence_count, attributes, first_seen_at, last_seen_at, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		rel.ID, rel.OwnerEntityID, rel.AgentID, rel.SourceEntityID, rel.TargetEntityID,
		rel.RelationshipType, rel.Description, rel.Strength, rel.Confidence, bidir,
		rel.EvidenceCount, attrs, now, now, now, now)
	return err
}

func (s *SQLiteStore) GetRelationship(ctx context.Context, id string) (*Relationship, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, owner_entity_id, agent_id, source_entity_id, target_entity_id,
			relationship_type, description, strength, confidence, is_bidirectional,
			evidence_count, attributes, first_seen_at, last_seen_at, created_at, updated_at
		FROM relationships WHERE id = ?`, id)
	return scanRelationship(row)
}

func (s *SQLiteStore) FindRelationship(ctx context.Context, ownerEntityID, sourceID, targetID, relType string) (*Relationship, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, owner_entity_id, agent_id, source_entity_id, target_entity_id,
			relationship_type, description, strength, confidence, is_bidirectional,
			evidence_count, attributes, first_seen_at, last_seen_at, created_at, updated_at
		FROM relationships
		WHERE owner_entity_id = ? AND source_entity_id = ? AND target_entity_id = ? AND relationship_type = ?`,
		ownerEntityID, sourceID, targetID, relType)
	rel, err := scanRelationship(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return rel, err
}

func (s *SQLiteStore) GetEntityRelationships(ctx context.Context, ownerEntityID, entityID, direction string) ([]*Relationship, error) {
	var q string
	var args []any

	switch direction {
	case "outgoing":
		q = `SELECT id, owner_entity_id, agent_id, source_entity_id, target_entity_id,
				relationship_type, description, strength, confidence, is_bidirectional,
				evidence_count, attributes, first_seen_at, last_seen_at, created_at, updated_at
			FROM relationships WHERE owner_entity_id = ? AND source_entity_id = ?`
		args = []any{ownerEntityID, entityID}
	case "incoming":
		q = `SELECT id, owner_entity_id, agent_id, source_entity_id, target_entity_id,
				relationship_type, description, strength, confidence, is_bidirectional,
				evidence_count, attributes, first_seen_at, last_seen_at, created_at, updated_at
			FROM relationships WHERE owner_entity_id = ? AND target_entity_id = ?`
		args = []any{ownerEntityID, entityID}
	default:
		q = `SELECT id, owner_entity_id, agent_id, source_entity_id, target_entity_id,
				relationship_type, description, strength, confidence, is_bidirectional,
				evidence_count, attributes, first_seen_at, last_seen_at, created_at, updated_at
			FROM relationships WHERE owner_entity_id = ? AND (source_entity_id = ? OR target_entity_id = ?)`
		args = []any{ownerEntityID, entityID, entityID}
	}

	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	return scanRelationships(rows)
}

func (s *SQLiteStore) QueryRelationships(ctx context.Context, query RelationshipQuery) ([]*Relationship, error) {
	var where []string
	var args []any

	if query.OwnerEntityID != "" {
		where = append(where, "owner_entity_id = ?")
		args = append(args, query.OwnerEntityID)
	}
	if query.EntityID != "" {
		where = append(where, "(source_entity_id = ? OR target_entity_id = ?)")
		args = append(args, query.EntityID, query.EntityID)
	}
	if len(query.RelationshipTypes) > 0 {
		ph := make([]string, len(query.RelationshipTypes))
		for i, t := range query.RelationshipTypes {
			ph[i] = "?"
			args = append(args, t)
		}
		where = append(where, fmt.Sprintf("relationship_type IN (%s)", strings.Join(ph, ",")))
	}
	if query.MinStrength > 0 {
		where = append(where, "strength >= ?")
		args = append(args, query.MinStrength)
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
		`SELECT id, owner_entity_id, agent_id, source_entity_id, target_entity_id,
			relationship_type, description, strength, confidence, is_bidirectional,
			evidence_count, attributes, first_seen_at, last_seen_at, created_at, updated_at
		FROM relationships %s ORDER BY strength DESC LIMIT ? OFFSET ?`,
		whereClause)
	args = append(args, limit, query.Offset)

	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanRelationships(rows)
}

func (s *SQLiteStore) UpdateRelationship(ctx context.Context, id string, updates map[string]any) (*Relationship, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var setClauses []string
	var args []any
	for k, v := range updates {
		if !allowedRelationshipColumns[k] {
			return nil, fmt.Errorf("invalid relationship update column: %q", k)
		}
		setClauses = append(setClauses, k+" = ?")
		args = append(args, v)
	}
	setClauses = append(setClauses, "updated_at = ?")
	args = append(args, time.Now().UTC().Format(time.RFC3339))
	args = append(args, id)

	_, err := s.db.ExecContext(ctx,
		fmt.Sprintf("UPDATE relationships SET %s WHERE id = ?", strings.Join(setClauses, ", ")),
		args...)
	if err != nil {
		return nil, err
	}
	return s.GetRelationship(ctx, id)
}

func (s *SQLiteStore) IncrementRelationshipEvidence(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.db.ExecContext(ctx,
		"UPDATE relationships SET evidence_count = evidence_count + 1, last_seen_at = ?, updated_at = ? WHERE id = ?",
		now, now, id)
	return err
}

func (s *SQLiteStore) DeleteRelationship(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(ctx, "DELETE FROM relationships WHERE id = ?", id)
	return err
}

func (s *SQLiteStore) CreateRelationshipEvidence(ctx context.Context, evidence *RelationshipEvidence) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if evidence.ID == "" {
		evidence.ID = ulid.Make().String()
	}
	now := time.Now().UTC().Format(time.RFC3339)

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO relationship_evidence (id, relationship_id, memory_id, evidence_text, confidence, created_at)
		VALUES (?, ?, ?, ?, ?, ?)`,
		evidence.ID, evidence.RelationshipID, evidence.MemoryID,
		evidence.EvidenceText, evidence.Confidence, now)
	return err
}

func (s *SQLiteStore) GetRelationshipEvidence(ctx context.Context, relationshipID string, limit int) ([]*RelationshipEvidence, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, relationship_id, memory_id, evidence_text, confidence, created_at
		FROM relationship_evidence WHERE relationship_id = ? ORDER BY created_at DESC LIMIT ?`,
		relationshipID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var evidence []*RelationshipEvidence
	for rows.Next() {
		var e RelationshipEvidence
		var createdStr string
		if err := rows.Scan(&e.ID, &e.RelationshipID, &e.MemoryID,
			&e.EvidenceText, &e.Confidence, &createdStr); err != nil {
			return nil, err
		}
		e.CreatedAt, _ = time.Parse(time.RFC3339, createdStr)
		evidence = append(evidence, &e)
	}
	return evidence, nil
}

func (s *SQLiteStore) GetRelationshipPath(ctx context.Context, ownerEntityID, fromEntityID, toEntityID string, maxDepth int) ([]string, error) {
	if fromEntityID == toEntityID {
		return []string{fromEntityID}, nil
	}

	visited := map[string]bool{fromEntityID: true}
	parent := map[string]string{}
	queue := []string{fromEntityID}

	for depth := 0; depth < maxDepth && len(queue) > 0; depth++ {
		nextQueue := make([]string, 0)

		for _, current := range queue {
			rels, err := s.GetEntityRelationships(ctx, ownerEntityID, current, "both")
			if err != nil {
				continue
			}

			for _, rel := range rels {
				var targetID string
				if rel.SourceEntityID == current {
					targetID = rel.TargetEntityID
				} else {
					targetID = rel.SourceEntityID
				}

				if targetID == toEntityID {
					path := []string{toEntityID}
					curr := current
					for curr != fromEntityID {
						path = append([]string{curr}, path...)
						curr = parent[curr]
					}
					return append([]string{fromEntityID}, path...), nil
				}

				if !visited[targetID] {
					visited[targetID] = true
					parent[targetID] = current
					nextQueue = append(nextQueue, targetID)
				}
			}
		}

		queue = nextQueue
	}

	return nil, fmt.Errorf("no path found")
}

func (s *SQLiteStore) DeleteAllRelationshipsForOwner(ctx context.Context, ownerEntityID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	result, err := s.db.ExecContext(ctx,
		"DELETE FROM relationships WHERE owner_entity_id = ?", ownerEntityID)
	if err != nil {
		return 0, err
	}
	n, _ := result.RowsAffected()
	return int(n), nil
}
