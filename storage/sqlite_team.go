// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2026 Keyoku. All rights reserved.

package storage

import (
	"context"
	"database/sql"
	"time"

	"github.com/oklog/ulid/v2"
)

// --- Team CRUD ---

func (s *SQLiteStore) CreateTeam(ctx context.Context, team *Team) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if team.ID == "" {
		team.ID = ulid.Make().String()
	}
	now := time.Now().UTC()
	team.CreatedAt = now
	team.UpdatedAt = now

	visibility := string(team.DefaultVisibility)
	if visibility == "" {
		visibility = string(VisibilityTeam)
	}

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO teams (id, name, description, default_visibility, created_at, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		team.ID, team.Name, team.Description, visibility,
		now.Format(time.RFC3339), now.Format(time.RFC3339),
	)
	return err
}

func (s *SQLiteStore) GetTeam(ctx context.Context, id string) (*Team, error) {
	var t Team
	var visStr, createdStr, updatedStr string

	err := s.db.QueryRowContext(ctx,
		`SELECT id, name, description, default_visibility, created_at, updated_at FROM teams WHERE id = ?`, id,
	).Scan(&t.ID, &t.Name, &t.Description, &visStr, &createdStr, &updatedStr)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	t.DefaultVisibility = MemoryVisibility(visStr)
	t.CreatedAt, _ = time.Parse(time.RFC3339, createdStr)
	t.UpdatedAt, _ = time.Parse(time.RFC3339, updatedStr)
	return &t, nil
}

func (s *SQLiteStore) DeleteTeam(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(ctx, "DELETE FROM teams WHERE id = ?", id)
	return err
}

func (s *SQLiteStore) AddTeamMember(ctx context.Context, teamID, agentID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO team_members (team_id, agent_id, role, joined_at) VALUES (?, ?, 'member', ?)`,
		teamID, agentID, now,
	)
	return err
}

func (s *SQLiteStore) RemoveTeamMember(ctx context.Context, teamID, agentID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(ctx,
		`DELETE FROM team_members WHERE team_id = ? AND agent_id = ?`,
		teamID, agentID,
	)
	return err
}

func (s *SQLiteStore) GetTeamMembers(ctx context.Context, teamID string) ([]*TeamMember, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT team_id, agent_id, role, joined_at FROM team_members WHERE team_id = ?`, teamID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var members []*TeamMember
	for rows.Next() {
		var m TeamMember
		var joinedStr string
		if err := rows.Scan(&m.TeamID, &m.AgentID, &m.Role, &joinedStr); err != nil {
			return nil, err
		}
		m.JoinedAt, _ = time.Parse(time.RFC3339, joinedStr)
		members = append(members, &m)
	}
	return members, nil
}

func (s *SQLiteStore) GetTeamForAgent(ctx context.Context, agentID string) (string, error) {
	var teamID string
	err := s.db.QueryRowContext(ctx,
		`SELECT team_id FROM team_members WHERE agent_id = ?`, agentID,
	).Scan(&teamID)
	if err == sql.ErrNoRows {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	return teamID, nil
}
