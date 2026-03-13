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

	"github.com/oklog/ulid/v2"
)

// allowedSchemaColumns restricts UPDATE SET columns for extraction schemas.
var allowedSchemaColumns = map[string]bool{
	"name": true, "description": true, "schema_definition": true,
	"is_active": true, "version": true,
}

// --- Schema CRUD ---

func (s *SQLiteStore) CreateSchema(ctx context.Context, schema *ExtractionSchema) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if schema.ID == "" {
		schema.ID = ulid.Make().String()
	}
	now := time.Now().UTC().Format(time.RFC3339)
	schemaDef, _ := json.Marshal(schema.SchemaDefinition)
	isActive := 0
	if schema.IsActive {
		isActive = 1
	}

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO extraction_schemas (id, entity_id, name, description, version, schema_definition, is_active, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		schema.ID, schema.EntityID, schema.Name, schema.Description, schema.Version,
		string(schemaDef), isActive, now, now)
	if err != nil {
		return err
	}

	schema.CreatedAt, _ = time.Parse(time.RFC3339, now)
	schema.UpdatedAt = schema.CreatedAt
	return nil
}

func (s *SQLiteStore) GetSchema(ctx context.Context, id string) (*ExtractionSchema, error) {
	var es ExtractionSchema
	var schemaDefStr string
	var isActive int
	var createdStr, updatedStr string

	err := s.db.QueryRowContext(ctx,
		`SELECT id, entity_id, name, description, version, schema_definition, is_active, created_at, updated_at
		FROM extraction_schemas WHERE id = ?`, id).Scan(
		&es.ID, &es.EntityID, &es.Name, &es.Description, &es.Version,
		&schemaDefStr, &isActive, &createdStr, &updatedStr)
	if err != nil {
		return nil, err
	}

	es.IsActive = isActive != 0
	json.Unmarshal([]byte(schemaDefStr), &es.SchemaDefinition)
	es.CreatedAt, _ = time.Parse(time.RFC3339, createdStr)
	es.UpdatedAt, _ = time.Parse(time.RFC3339, updatedStr)
	return &es, nil
}

func (s *SQLiteStore) GetSchemaByName(ctx context.Context, entityID, name string) (*ExtractionSchema, error) {
	var es ExtractionSchema
	var schemaDefStr string
	var isActive int
	var createdStr, updatedStr string

	err := s.db.QueryRowContext(ctx,
		`SELECT id, entity_id, name, description, version, schema_definition, is_active, created_at, updated_at
		FROM extraction_schemas WHERE entity_id = ? AND name = ?`, entityID, name).Scan(
		&es.ID, &es.EntityID, &es.Name, &es.Description, &es.Version,
		&schemaDefStr, &isActive, &createdStr, &updatedStr)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}

	es.IsActive = isActive != 0
	json.Unmarshal([]byte(schemaDefStr), &es.SchemaDefinition)
	es.CreatedAt, _ = time.Parse(time.RFC3339, createdStr)
	es.UpdatedAt, _ = time.Parse(time.RFC3339, updatedStr)
	return &es, nil
}

func (s *SQLiteStore) QuerySchemas(ctx context.Context, query SchemaQuery) ([]*ExtractionSchema, error) {
	var where []string
	var args []any

	if query.EntityID != "" {
		where = append(where, "entity_id = ?")
		args = append(args, query.EntityID)
	}
	if query.ActiveOnly {
		where = append(where, "is_active = 1")
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
		`SELECT id, entity_id, name, description, version, schema_definition, is_active, created_at, updated_at
		FROM extraction_schemas %s ORDER BY created_at DESC LIMIT ? OFFSET ?`,
		whereClause)
	args = append(args, limit, query.Offset)

	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var schemas []*ExtractionSchema
	for rows.Next() {
		var es ExtractionSchema
		var schemaDefStr string
		var isActive int
		var createdStr, updatedStr string
		if err := rows.Scan(&es.ID, &es.EntityID, &es.Name, &es.Description, &es.Version,
			&schemaDefStr, &isActive, &createdStr, &updatedStr); err != nil {
			return nil, err
		}
		es.IsActive = isActive != 0
		json.Unmarshal([]byte(schemaDefStr), &es.SchemaDefinition)
		es.CreatedAt, _ = time.Parse(time.RFC3339, createdStr)
		es.UpdatedAt, _ = time.Parse(time.RFC3339, updatedStr)
		schemas = append(schemas, &es)
	}
	return schemas, nil
}

func (s *SQLiteStore) UpdateSchema(ctx context.Context, id string, updates map[string]any) (*ExtractionSchema, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var setClauses []string
	var args []any

	for k, v := range updates {
		if !allowedSchemaColumns[k] {
			return nil, fmt.Errorf("invalid schema update column: %q", k)
		}
		if k == "schema_definition" {
			if def, ok := v.(map[string]any); ok {
				jsonBytes, _ := json.Marshal(def)
				setClauses = append(setClauses, "schema_definition = ?")
				args = append(args, string(jsonBytes))
				continue
			}
		}
		if k == "is_active" {
			if active, ok := v.(bool); ok {
				val := 0
				if active {
					val = 1
				}
				setClauses = append(setClauses, "is_active = ?")
				args = append(args, val)
				continue
			}
		}
		setClauses = append(setClauses, k+" = ?")
		args = append(args, v)
	}

	setClauses = append(setClauses, "updated_at = ?")
	args = append(args, time.Now().UTC().Format(time.RFC3339))
	args = append(args, id)

	_, err := s.db.ExecContext(ctx,
		fmt.Sprintf("UPDATE extraction_schemas SET %s WHERE id = ?", strings.Join(setClauses, ", ")),
		args...)
	if err != nil {
		return nil, err
	}
	return s.GetSchema(ctx, id)
}

func (s *SQLiteStore) DeleteSchema(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(ctx, "DELETE FROM extraction_schemas WHERE id = ?", id)
	return err
}

// --- Custom Extraction CRUD ---

func (s *SQLiteStore) CreateCustomExtraction(ctx context.Context, extraction *CustomExtraction) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if extraction.ID == "" {
		extraction.ID = ulid.Make().String()
	}
	now := time.Now().UTC().Format(time.RFC3339)
	dataJSON, _ := json.Marshal(extraction.ExtractedData)

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO custom_extractions (id, entity_id, memory_id, schema_id, extracted_data, extraction_provider, extraction_model, confidence, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		extraction.ID, extraction.EntityID, extraction.MemoryID, extraction.SchemaID,
		string(dataJSON), extraction.ExtractionProvider, extraction.ExtractionModel,
		extraction.Confidence, now)
	if err != nil {
		return err
	}

	extraction.CreatedAt, _ = time.Parse(time.RFC3339, now)
	return nil
}

func (s *SQLiteStore) GetCustomExtraction(ctx context.Context, id string) (*CustomExtraction, error) {
	var ce CustomExtraction
	var dataStr string
	var createdStr string

	err := s.db.QueryRowContext(ctx,
		`SELECT id, entity_id, memory_id, schema_id, extracted_data, extraction_provider, extraction_model, confidence, created_at
		FROM custom_extractions WHERE id = ?`, id).Scan(
		&ce.ID, &ce.EntityID, &ce.MemoryID, &ce.SchemaID,
		&dataStr, &ce.ExtractionProvider, &ce.ExtractionModel,
		&ce.Confidence, &createdStr)
	if err != nil {
		return nil, err
	}

	json.Unmarshal([]byte(dataStr), &ce.ExtractedData)
	ce.CreatedAt, _ = time.Parse(time.RFC3339, createdStr)
	return &ce, nil
}

func (s *SQLiteStore) GetCustomExtractionsByMemory(ctx context.Context, memoryID string) ([]*CustomExtraction, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, entity_id, memory_id, schema_id, extracted_data, extraction_provider, extraction_model, confidence, created_at
		FROM custom_extractions WHERE memory_id = ? ORDER BY created_at DESC`, memoryID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	return scanCustomExtractions(rows)
}

func (s *SQLiteStore) QueryCustomExtractions(ctx context.Context, query CustomExtractionQuery) ([]*CustomExtraction, error) {
	var where []string
	var args []any

	if query.EntityID != "" {
		where = append(where, "entity_id = ?")
		args = append(args, query.EntityID)
	}
	if query.MemoryID != "" {
		where = append(where, "memory_id = ?")
		args = append(args, query.MemoryID)
	}
	if query.SchemaID != "" {
		where = append(where, "schema_id = ?")
		args = append(args, query.SchemaID)
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
		`SELECT id, entity_id, memory_id, schema_id, extracted_data, extraction_provider, extraction_model, confidence, created_at
		FROM custom_extractions %s ORDER BY created_at DESC LIMIT ? OFFSET ?`,
		whereClause)
	args = append(args, limit, query.Offset)

	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	return scanCustomExtractions(rows)
}

func (s *SQLiteStore) DeleteCustomExtraction(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(ctx, "DELETE FROM custom_extractions WHERE id = ?", id)
	return err
}

func (s *SQLiteStore) DeleteCustomExtractionsBySchema(ctx context.Context, schemaID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(ctx, "DELETE FROM custom_extractions WHERE schema_id = ?", schemaID)
	return err
}
