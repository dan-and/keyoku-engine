// SPDX-License-Identifier: BSL-1.1
// Copyright (c) 2025 Keyoku. All rights reserved.

package keyoku

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/keyoku-ai/keyoku-engine/llm"
	"github.com/keyoku-ai/keyoku-engine/storage"
)

// HeartbeatResult contains the result of a HeartbeatCheck.
type HeartbeatResult struct {
	ShouldAct     bool
	PendingWork   []*Memory       // Active plans/tasks needing attention
	Deadlines     []*Memory       // Approaching expiry
	Scheduled     []*Memory       // Cron-tagged memories that are due
	Decaying      []*Memory       // Important memories nearing decay threshold
	Conflicts     []ConflictPair  // Unresolved contradictions
	StaleMonitors []*Memory       // Monitoring tasks overdue for check
	Summary       string          // Pre-built context string for LLM

	// Extended heartbeat signals (populated by new check types)
	GoalProgress  []GoalProgressItem  // Plan vs activity cross-reference
	Continuity    *ContinuityItem     // Interrupted session detection
	Sentiment     *SentimentTrend     // Emotional trend analysis
	Relationships []RelationshipAlert // Silent entities near deadlines
	KnowledgeGaps []KnowledgeGap      // Unanswered questions
	Patterns      []BehavioralPattern // Day-of-week behavioral patterns

	// LLM prioritization fields (populated only when WithLLMPrioritization is set)
	PriorityAction string   // The single most important action
	ActionItems    []string // All items ordered by priority
	Urgency        string   // "immediate", "soon", "can_wait"

	// Team heartbeat fields (populated only in team heartbeat mode)
	ByAgent map[string]*AgentHeartbeatSummary // per-agent breakdown (team mode only)
}

// GoalProgressItem tracks a plan's progress based on related activity memories.
type GoalProgressItem struct {
	Plan       *Memory
	Activities []*Memory
	Progress   float64 // 0.0-1.0 heuristic based on activity count weighted by recency
	DaysLeft   float64 // days until plan expires (-1 if no expiry)
	Status     string  // "on_track", "at_risk", "stalled", "no_activity"
}

// ContinuityItem detects interrupted sessions for resume suggestions.
type ContinuityItem struct {
	LastSessionMemories []*Memory
	SessionAge          time.Duration
	WasInterrupted      bool
	ResumeSuggestion    string
}

// SentimentTrend analyzes emotional direction across recent memories.
type SentimentTrend struct {
	RecentAvg   float64   // avg sentiment of most recent memories
	PreviousAvg float64   // avg sentiment of earlier memories
	Direction   string    // "improving", "declining", "stable"
	Delta       float64   // absolute change
	Notable     []*Memory // memories with extreme sentiment (|s| > 0.7)
}

// RelationshipAlert flags entities that have gone silent but have active plans.
type RelationshipAlert struct {
	Entity       *storage.Entity
	LastMentioned time.Time
	DaysSilent    int
	RelatedPlans  []*Memory
	Urgency       string // "info", "attention", "urgent"
}

// KnowledgeGap represents an unanswered question found in memories.
type KnowledgeGap struct {
	Question   string
	AskedAt    time.Time
	HasRelated bool
}

// AgentHeartbeatSummary contains heartbeat counts attributed to a specific agent.
type AgentHeartbeatSummary struct {
	AgentID     string
	PendingWork int
	Deadlines   int
	Scheduled   int
	Decaying    int
	Conflicts   int
}

// ConflictPair re-exported from storage for public API.
type ConflictPair = storage.ConflictPair

// HeartbeatOption configures a HeartbeatCheck call.
type HeartbeatOption func(*heartbeatConfig)

type heartbeatConfig struct {
	deadlineWindow  time.Duration
	decayThreshold  float64
	importanceFloor float64
	maxResults      int
	agentID         string
	checks          []HeartbeatCheckType

	// Team heartbeat
	teamID        string // When set, runs team-wide heartbeat
	teamHeartbeat bool   // Enables team heartbeat mode

	// LLM prioritization (opt-in)
	llmProvider    llm.Provider
	agentContext   string
	entityContext  string
}

// HeartbeatCheckType represents a specific check to run.
type HeartbeatCheckType string

const (
	CheckPendingWork  HeartbeatCheckType = "pending_work"
	CheckDeadlines    HeartbeatCheckType = "deadlines"
	CheckScheduled    HeartbeatCheckType = "scheduled"
	CheckDecaying     HeartbeatCheckType = "decaying"
	CheckConflicts    HeartbeatCheckType = "conflicts"
	CheckStale        HeartbeatCheckType = "stale_monitors"
	CheckGoalProgress HeartbeatCheckType = "goal_progress"
	CheckContinuity   HeartbeatCheckType = "continuity"
	CheckSentiment    HeartbeatCheckType = "sentiment"
	CheckRelationship HeartbeatCheckType = "relationship"
	CheckKnowledge    HeartbeatCheckType = "knowledge_gaps"
	CheckPatterns     HeartbeatCheckType = "patterns"
)

var allChecks = []HeartbeatCheckType{
	CheckPendingWork, CheckDeadlines, CheckScheduled,
	CheckDecaying, CheckConflicts, CheckStale,
	CheckGoalProgress, CheckContinuity, CheckSentiment,
	CheckRelationship, CheckKnowledge, CheckPatterns,
}

// WithDeadlineWindow sets how far ahead to look for deadlines (default: 24h).
func WithDeadlineWindow(d time.Duration) HeartbeatOption {
	return func(c *heartbeatConfig) { c.deadlineWindow = d }
}

// WithDecayThreshold sets the decay factor below which memories are flagged (default: 0.4).
func WithDecayThreshold(f float64) HeartbeatOption {
	return func(c *heartbeatConfig) { c.decayThreshold = f }
}

// WithImportanceFloor sets the minimum importance for flagging (default: 0.7).
func WithImportanceFloor(f float64) HeartbeatOption {
	return func(c *heartbeatConfig) { c.importanceFloor = f }
}

// WithMaxResults sets the maximum results per check category (default: 20).
func WithMaxResults(n int) HeartbeatOption {
	return func(c *heartbeatConfig) { c.maxResults = n }
}

// WithHeartbeatAgentID scopes the check to a specific agent.
func WithHeartbeatAgentID(id string) HeartbeatOption {
	return func(c *heartbeatConfig) { c.agentID = id }
}

// WithChecks enables only specific checks.
func WithChecks(checks ...HeartbeatCheckType) HeartbeatOption {
	return func(c *heartbeatConfig) { c.checks = checks }
}

// WithTeamHeartbeat enables team-wide heartbeat mode.
// Queries team-visible and global memories across all agents in the team.
// Results include ByAgent attribution showing which agent owns each signal.
func WithTeamHeartbeat(teamID string) HeartbeatOption {
	return func(c *heartbeatConfig) {
		c.teamID = teamID
		c.teamHeartbeat = true
	}
}

// WithLLMPrioritization enables LLM-powered action prioritization on heartbeat results.
// Only fires when ShouldAct is true. The provider should be the same one used for memory extraction.
func WithLLMPrioritization(provider llm.Provider, agentContext, entityContext string) HeartbeatOption {
	return func(c *heartbeatConfig) {
		c.llmProvider = provider
		c.agentContext = agentContext
		c.entityContext = entityContext
	}
}

// HeartbeatCheck performs a zero-token local query against SQLite.
// Returns whether the agent should act and what needs attention.
func (k *Keyoku) HeartbeatCheck(ctx context.Context, entityID string, opts ...HeartbeatOption) (*HeartbeatResult, error) {
	cfg := &heartbeatConfig{
		deadlineWindow:  24 * time.Hour,
		decayThreshold:  0.4,
		importanceFloor: 0.7,
		maxResults:      20,
		checks:          allChecks,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	// Build visibility context for team-aware queries
	var visibilityFor *storage.VisibilityContext
	if cfg.teamHeartbeat && cfg.teamID != "" {
		// Team heartbeat: see team + global memories (no private — agentID left empty)
		visibilityFor = &storage.VisibilityContext{TeamID: cfg.teamID}
	} else if cfg.agentID != "" {
		// Agent-level: auto-resolve team if agent belongs to one
		if teamID, err := k.store.GetTeamForAgent(ctx, cfg.agentID); err == nil && teamID != "" {
			visibilityFor = &storage.VisibilityContext{AgentID: cfg.agentID, TeamID: teamID}
		}
	}

	result := &HeartbeatResult{}
	checksToRun := make(map[HeartbeatCheckType]bool)
	for _, c := range cfg.checks {
		checksToRun[c] = true
	}

	// Helper to build a query with visibility applied
	buildQuery := func(q storage.MemoryQuery) storage.MemoryQuery {
		if visibilityFor != nil {
			q.VisibilityFor = visibilityFor
			// When using visibility filtering, don't also filter by agent_id
			// (VisibilityFor handles the scoping)
			q.AgentID = ""
		}
		return q
	}

	// 1. PENDING WORK — PLAN/ACTIVITY memories, active state, importance > threshold
	if checksToRun[CheckPendingWork] {
		pending, err := k.store.QueryMemories(ctx, buildQuery(storage.MemoryQuery{
			EntityID:   entityID,
			AgentID:    cfg.agentID,
			Types:      []storage.MemoryType{storage.TypePlan, storage.TypeActivity},
			States:     []storage.MemoryState{storage.StateActive},
			Limit:      cfg.maxResults,
			OrderBy:    "importance",
			Descending: true,
		}))
		if err == nil {
			for _, m := range pending {
				if m.Importance >= cfg.importanceFloor {
					result.PendingWork = append(result.PendingWork, m)
				}
			}
		}
	}

	// 2. DEADLINES — Memories with expires_at approaching
	if checksToRun[CheckDeadlines] {
		allActive, err := k.store.QueryMemories(ctx, buildQuery(storage.MemoryQuery{
			EntityID:   entityID,
			AgentID:    cfg.agentID,
			States:     []storage.MemoryState{storage.StateActive},
			Limit:      cfg.maxResults * 5,
			OrderBy:    "importance",
			Descending: true,
		}))
		if err == nil {
			deadlineHorizon := time.Now().Add(cfg.deadlineWindow)
			for _, m := range allActive {
				if m.ExpiresAt != nil && m.ExpiresAt.Before(deadlineHorizon) {
					result.Deadlines = append(result.Deadlines, m)
					if len(result.Deadlines) >= cfg.maxResults {
						break
					}
				}
			}
		}
	}

	// 3. SCHEDULED (CRON) — Memories tagged "cron:*" where due
	if checksToRun[CheckScheduled] {
		allActive, err := k.store.QueryMemories(ctx, buildQuery(storage.MemoryQuery{
			EntityID: entityID,
			AgentID:  cfg.agentID,
			States:   []storage.MemoryState{storage.StateActive},
			Limit:    cfg.maxResults * 5,
		}))
		if err == nil {
			now := time.Now()
			for _, m := range allActive {
				sched, err := ParseScheduleFromTags(m.Tags)
				if err != nil || sched == nil {
					continue
				}
				lastRun := m.CreatedAt
				if m.LastAccessedAt != nil {
					lastRun = *m.LastAccessedAt
				}
				if sched.IsDue(lastRun, now) {
					result.Scheduled = append(result.Scheduled, m)

					// Auto-acknowledge: advance last_accessed_at so the task
					// doesn't re-fire until the next scheduled interval.
					_ = k.store.UpdateAccessStats(ctx, []string{m.ID})

					// One-time cleanup: archive cron:once:* memories after they fire.
					if sched.Type == ScheduleOnce {
						archivedState := storage.StateArchived
						_, _ = k.store.UpdateMemory(ctx, m.ID, storage.MemoryUpdate{
							State: &archivedState,
						})
					}

					if len(result.Scheduled) >= cfg.maxResults {
						break
					}
				}
			}
		}
	}

	// 4. DECAYING ALERTS — Important memories with high decay
	if checksToRun[CheckDecaying] {
		stale, err := k.store.GetStaleMemories(ctx, entityID, cfg.decayThreshold)
		if err == nil {
			for _, m := range stale {
				if m.Importance >= 0.8 {
					// Apply visibility filter for team-aware heartbeat
					if visibilityFor != nil && !storage.IsVisibleTo(m.Visibility, m.AgentID, m.TeamID, visibilityFor) {
						continue
					}
					result.Decaying = append(result.Decaying, m)
					if len(result.Decaying) >= cfg.maxResults {
						break
					}
				}
			}
		}
	}

	// 5. CONFLICTS — Unresolved conflicts (flagged in confidence_factors)
	if checksToRun[CheckConflicts] {
		allActive, err := k.store.QueryMemories(ctx, buildQuery(storage.MemoryQuery{
			EntityID: entityID,
			AgentID:  cfg.agentID,
			States:   []storage.MemoryState{storage.StateActive},
			Limit:    cfg.maxResults * 5,
		}))
		if err == nil {
			for _, m := range allActive {
				for _, factor := range m.ConfidenceFactors {
					if strings.HasPrefix(factor, "conflict_flagged:") {
						result.Conflicts = append(result.Conflicts, ConflictPair{
							MemoryA: m,
							Reason:  strings.TrimPrefix(factor, "conflict_flagged: "),
						})
						break
					}
				}
				if len(result.Conflicts) >= cfg.maxResults {
					break
				}
			}
		}
	}

	// 6. STALE MONITORS — PLAN memories tagged "monitor" not accessed within expected window
	if checksToRun[CheckStale] {
		plans, err := k.store.QueryMemories(ctx, buildQuery(storage.MemoryQuery{
			EntityID: entityID,
			AgentID:  cfg.agentID,
			Types:    []storage.MemoryType{storage.TypePlan},
			States:   []storage.MemoryState{storage.StateActive},
			Limit:    cfg.maxResults * 3,
		}))
		if err == nil {
			now := time.Now()
			for _, m := range plans {
				if hasTag(m.Tags, "monitor") {
					lastAccess := m.CreatedAt
					if m.LastAccessedAt != nil {
						lastAccess = *m.LastAccessedAt
					}
					// Default check interval: 24 hours
					if now.Sub(lastAccess) > 24*time.Hour {
						result.StaleMonitors = append(result.StaleMonitors, m)
						if len(result.StaleMonitors) >= cfg.maxResults {
							break
						}
					}
				}
			}
		}
	}

	// 7. GOAL PROGRESS — Cross-reference PLAN memories against ACTIVITY memories
	if checksToRun[CheckGoalProgress] {
		plans, err := k.store.QueryMemories(ctx, buildQuery(storage.MemoryQuery{
			EntityID:   entityID,
			AgentID:    cfg.agentID,
			Types:      []storage.MemoryType{storage.TypePlan},
			States:     []storage.MemoryState{storage.StateActive},
			Limit:      cfg.maxResults,
			OrderBy:    "importance",
			Descending: true,
		}))
		if err == nil && len(plans) > 0 {
			now := time.Now()
			for _, plan := range plans {
				item := GoalProgressItem{Plan: plan, DaysLeft: -1}

				if plan.ExpiresAt != nil {
					item.DaysLeft = plan.ExpiresAt.Sub(now).Hours() / 24
				}

				// Find related activities by embedding similarity
				if k.emb != nil {
					emb, embErr := k.emb.Embed(ctx, plan.Content)
					if embErr == nil {
						similar, _ := k.store.FindSimilar(ctx, emb, entityID, 20, 0.4)
						for _, s := range similar {
							if s.Memory.Type == storage.TypeActivity && s.Memory.State == storage.StateActive {
								item.Activities = append(item.Activities, s.Memory)
							}
						}
					}
				}

				// Compute progress heuristic
				activityCount := len(item.Activities)
				if activityCount == 0 {
					item.Status = "no_activity"
					item.Progress = 0
				} else {
					// Weight by recency: activities in last 7 days count more
					recentCount := 0
					for _, a := range item.Activities {
						if now.Sub(a.CreatedAt) < 7*24*time.Hour {
							recentCount++
						}
					}
					item.Progress = float64(activityCount) / 10.0
					if item.Progress > 1.0 {
						item.Progress = 1.0
					}

					if recentCount > 0 {
						item.Status = "on_track"
					} else if item.DaysLeft >= 0 && item.DaysLeft < 14 {
						item.Status = "at_risk"
					} else {
						item.Status = "stalled"
					}
				}

				result.GoalProgress = append(result.GoalProgress, item)
			}
		}
	}

	// 8. CONTEXT CONTINUITY — Detect interrupted sessions
	if checksToRun[CheckContinuity] {
		recent, err := k.store.QueryMemories(ctx, buildQuery(storage.MemoryQuery{
			EntityID:   entityID,
			AgentID:    cfg.agentID,
			Types:      []storage.MemoryType{storage.TypeContext, storage.TypeActivity},
			States:     []storage.MemoryState{storage.StateActive},
			Limit:      5,
			OrderBy:    "created_at",
			Descending: true,
		}))
		if err == nil && len(recent) > 0 {
			newest := recent[0]
			sessionAge := time.Since(newest.CreatedAt)

			if sessionAge < 48*time.Hour {
				// Check if there are unresolved plans/activities
				hasUnresolved := false
				var lastTopics []string
				for _, m := range recent {
					if m.Type == storage.TypeActivity || m.Type == storage.TypePlan {
						hasUnresolved = true
						if len(lastTopics) < 3 {
							content := m.Content
							if len(content) > 80 {
								content = content[:80] + "..."
							}
							lastTopics = append(lastTopics, content)
						}
					}
				}

				if hasUnresolved {
					suggestion := "You were working on: " + strings.Join(lastTopics, "; ")
					result.Continuity = &ContinuityItem{
						LastSessionMemories: recent,
						SessionAge:          sessionAge,
						WasInterrupted:      true,
						ResumeSuggestion:    suggestion,
					}
				}
			}
		}
	}

	// 9. SENTIMENT TRENDS — Pure SQL analysis, zero LLM cost
	if checksToRun[CheckSentiment] {
		recent, err := k.store.QueryMemories(ctx, buildQuery(storage.MemoryQuery{
			EntityID:   entityID,
			AgentID:    cfg.agentID,
			States:     []storage.MemoryState{storage.StateActive},
			Limit:      20,
			OrderBy:    "created_at",
			Descending: true,
		}))
		if err == nil && len(recent) >= 6 {
			mid := len(recent) / 2
			recentHalf := recent[:mid]
			previousHalf := recent[mid:]

			var recentSum, prevSum float64
			var notable []*Memory
			for _, m := range recentHalf {
				recentSum += m.Sentiment
				if m.Sentiment > 0.7 || m.Sentiment < -0.7 {
					notable = append(notable, m)
				}
			}
			for _, m := range previousHalf {
				prevSum += m.Sentiment
				if m.Sentiment > 0.7 || m.Sentiment < -0.7 {
					notable = append(notable, m)
				}
			}

			recentAvg := recentSum / float64(len(recentHalf))
			prevAvg := prevSum / float64(len(previousHalf))
			delta := recentAvg - prevAvg

			direction := "stable"
			if delta > 0.3 {
				direction = "improving"
			} else if delta < -0.3 {
				direction = "declining"
			}

			// Only report if there's a notable trend or extreme sentiment
			if direction != "stable" || len(notable) > 0 {
				absDelta := delta
				if absDelta < 0 {
					absDelta = -absDelta
				}
				result.Sentiment = &SentimentTrend{
					RecentAvg:   recentAvg,
					PreviousAvg: prevAvg,
					Direction:   direction,
					Delta:       absDelta,
					Notable:     notable,
				}
			}
		}
	}

	// 10. RELATIONSHIP CONTINUITY — Silent entities near deadlines
	if checksToRun[CheckRelationship] {
		entities, err := k.store.QueryEntities(ctx, storage.EntityQuery{
			OwnerEntityID: entityID,
			Limit:         50,
		})
		if err == nil {
			now := time.Now()
			for _, ent := range entities {
				if ent.LastMentionedAt == nil {
					continue
				}
				daysSilent := int(now.Sub(*ent.LastMentionedAt).Hours() / 24)
				if daysSilent < 7 {
					continue // recently active
				}

				// Find related active plans for this entity
				var relatedPlans []*Memory
				if k.emb != nil {
					emb, embErr := k.emb.Embed(ctx, ent.CanonicalName)
					if embErr == nil {
						similar, _ := k.store.FindSimilar(ctx, emb, entityID, 10, 0.4)
						for _, s := range similar {
							if s.Memory.Type == storage.TypePlan && s.Memory.State == storage.StateActive {
								relatedPlans = append(relatedPlans, s.Memory)
							}
						}
					}
				}

				urgency := "info"
				if len(relatedPlans) > 0 {
					urgency = "attention"
					// Check if any related plan has an approaching deadline
					for _, p := range relatedPlans {
						if p.ExpiresAt != nil && p.ExpiresAt.Sub(now).Hours()/24 < 14 {
							urgency = "urgent"
							break
						}
					}
				}

				if urgency != "info" || daysSilent > 14 {
					result.Relationships = append(result.Relationships, RelationshipAlert{
						Entity:        ent,
						LastMentioned: *ent.LastMentionedAt,
						DaysSilent:    daysSilent,
						RelatedPlans:  relatedPlans,
						Urgency:       urgency,
					})
				}
			}
		}
	}

	// 11. KNOWLEDGE GAPS — Unanswered questions in recent memories
	if checksToRun[CheckKnowledge] {
		recent, err := k.store.QueryMemories(ctx, buildQuery(storage.MemoryQuery{
			EntityID:   entityID,
			AgentID:    cfg.agentID,
			States:     []storage.MemoryState{storage.StateActive},
			Limit:      50,
			OrderBy:    "created_at",
			Descending: true,
		}))
		if err == nil && k.emb != nil {
			for _, m := range recent {
				if !looksLikeQuestion(m.Content) {
					continue
				}
				// Search for related memories that might answer the question
				emb, embErr := k.emb.Embed(ctx, m.Content)
				if embErr != nil {
					continue
				}
				similar, _ := k.store.FindSimilar(ctx, emb, entityID, 3, 0.6)
				hasAnswer := false
				for _, s := range similar {
					if s.Memory.ID != m.ID {
						hasAnswer = true
						break
					}
				}
				if !hasAnswer {
					result.KnowledgeGaps = append(result.KnowledgeGaps, KnowledgeGap{
						Question:   m.Content,
						AskedAt:    m.CreatedAt,
						HasRelated: false,
					})
					if len(result.KnowledgeGaps) >= 5 {
						break
					}
				}
			}
		}
	}

	// 12. BEHAVIORAL PATTERNS — Day-of-week topic frequency analysis
	if checksToRun[CheckPatterns] {
		result.Patterns = k.detectBehavioralPatterns(ctx, entityID)
	}

	// Build ByAgent attribution for team heartbeat mode
	if cfg.teamHeartbeat {
		result.ByAgent = buildByAgentAttribution(result)
	}

	// Determine if action is needed
	result.ShouldAct = len(result.PendingWork) > 0 ||
		len(result.Deadlines) > 0 ||
		len(result.Scheduled) > 0 ||
		len(result.Decaying) > 0 ||
		len(result.Conflicts) > 0 ||
		len(result.StaleMonitors) > 0 ||
		len(result.GoalProgress) > 0 ||
		result.Continuity != nil ||
		result.Sentiment != nil ||
		len(result.Relationships) > 0 ||
		len(result.KnowledgeGaps) > 0 ||
		len(result.Patterns) > 0

	// Build summary if action is needed
	if result.ShouldAct {
		result.Summary = buildSummary(result)

		// LLM prioritization (opt-in, only when there's something to prioritize)
		if cfg.llmProvider != nil && result.Summary != "" {
			priorityResp, err := cfg.llmProvider.PrioritizeActions(ctx, llm.ActionPriorityRequest{
				Summary:       result.Summary,
				AgentContext:  cfg.agentContext,
				EntityContext: cfg.entityContext,
			})
			if err == nil && priorityResp != nil {
				result.PriorityAction = priorityResp.PriorityAction
				result.ActionItems = priorityResp.ActionItems
				result.Urgency = priorityResp.Urgency
			}
			// LLM failure is non-fatal — heartbeat still returns local results
		}
	}

	return result, nil
}

// --- helpers ---

// Deprecated: parseCronTag is superseded by ParseScheduleFromTags in schedule.go.
// Kept temporarily for any external callers; will be removed in a future release.
func parseCronTag(tags []string) (time.Duration, bool) {
	for _, tag := range tags {
		if !strings.HasPrefix(tag, "cron:") {
			continue
		}

		rest := strings.TrimPrefix(tag, "cron:")

		switch {
		case rest == "hourly":
			return 1 * time.Hour, true
		case rest == "daily" || strings.HasPrefix(rest, "daily:"):
			return 24 * time.Hour, true
		case rest == "weekly" || strings.HasPrefix(rest, "weekly:"):
			return 7 * 24 * time.Hour, true
		case rest == "monthly":
			return 30 * 24 * time.Hour, true
		case strings.HasPrefix(rest, "every:"):
			durStr := strings.TrimPrefix(rest, "every:")
			d, err := time.ParseDuration(durStr)
			if err == nil {
				return d, true
			}
		}
	}
	return 0, false
}

func looksLikeQuestion(content string) bool {
	if strings.Contains(content, "?") {
		return true
	}
	lower := strings.ToLower(content)
	questionStarters := []string{"how ", "what ", "why ", "when ", "where ", "who ", "which ", "can ", "does ", "is there ", "should "}
	for _, q := range questionStarters {
		if strings.HasPrefix(lower, q) || strings.Contains(lower, " "+q) {
			return true
		}
	}
	return false
}

func hasTag(tags []string, target string) bool {
	for _, t := range tags {
		if t == target {
			return true
		}
	}
	return false
}

func buildByAgentAttribution(result *HeartbeatResult) map[string]*AgentHeartbeatSummary {
	byAgent := make(map[string]*AgentHeartbeatSummary)

	getOrCreate := func(agentID string) *AgentHeartbeatSummary {
		if s, ok := byAgent[agentID]; ok {
			return s
		}
		s := &AgentHeartbeatSummary{AgentID: agentID}
		byAgent[agentID] = s
		return s
	}

	for _, m := range result.PendingWork {
		getOrCreate(m.AgentID).PendingWork++
	}
	for _, m := range result.Deadlines {
		getOrCreate(m.AgentID).Deadlines++
	}
	for _, m := range result.Scheduled {
		getOrCreate(m.AgentID).Scheduled++
	}
	for _, m := range result.Decaying {
		getOrCreate(m.AgentID).Decaying++
	}
	for _, c := range result.Conflicts {
		getOrCreate(c.MemoryA.AgentID).Conflicts++
	}

	return byAgent
}

func buildSummary(result *HeartbeatResult) string {
	var parts []string

	isTeam := result.ByAgent != nil

	if len(result.PendingWork) > 0 {
		parts = append(parts, fmt.Sprintf("PENDING WORK (%d):", len(result.PendingWork)))
		for _, m := range result.PendingWork {
			if isTeam {
				parts = append(parts, fmt.Sprintf("  - [%s] %s (importance: %.2f, agent: %s)", m.Type, m.Content, m.Importance, m.AgentID))
			} else {
				parts = append(parts, fmt.Sprintf("  - [%s] %s (importance: %.2f)", m.Type, m.Content, m.Importance))
			}
		}
	}

	if len(result.Deadlines) > 0 {
		parts = append(parts, fmt.Sprintf("APPROACHING DEADLINES (%d):", len(result.Deadlines)))
		for _, m := range result.Deadlines {
			remaining := time.Until(*m.ExpiresAt).Round(time.Minute)
			parts = append(parts, fmt.Sprintf("  - %s (expires in %s)", m.Content, remaining))
		}
	}

	if len(result.Scheduled) > 0 {
		parts = append(parts, fmt.Sprintf("SCHEDULED TASKS DUE (%d):", len(result.Scheduled)))
		for _, m := range result.Scheduled {
			schedTag := ""
			for _, t := range m.Tags {
				if strings.HasPrefix(t, "cron:") {
					schedTag = t
					break
				}
			}
			if schedTag != "" {
				parts = append(parts, fmt.Sprintf("  - %s [schedule: %s]", m.Content, schedTag))
			} else {
				parts = append(parts, fmt.Sprintf("  - %s", m.Content))
			}
		}
	}

	if len(result.Decaying) > 0 {
		parts = append(parts, fmt.Sprintf("IMPORTANT MEMORIES DECAYING (%d):", len(result.Decaying)))
		for _, m := range result.Decaying {
			parts = append(parts, fmt.Sprintf("  - %s (importance: %.2f)", m.Content, m.Importance))
		}
	}

	if len(result.Conflicts) > 0 {
		parts = append(parts, fmt.Sprintf("UNRESOLVED CONFLICTS (%d):", len(result.Conflicts)))
		for _, c := range result.Conflicts {
			parts = append(parts, fmt.Sprintf("  - %s: %s", c.MemoryA.Content, c.Reason))
		}
	}

	if len(result.StaleMonitors) > 0 {
		parts = append(parts, fmt.Sprintf("STALE MONITORS (%d):", len(result.StaleMonitors)))
		for _, m := range result.StaleMonitors {
			parts = append(parts, fmt.Sprintf("  - %s", m.Content))
		}
	}

	if len(result.GoalProgress) > 0 {
		parts = append(parts, fmt.Sprintf("GOAL PROGRESS (%d):", len(result.GoalProgress)))
		for _, g := range result.GoalProgress {
			daysStr := "no deadline"
			if g.DaysLeft >= 0 {
				daysStr = fmt.Sprintf("%.0f days left", g.DaysLeft)
			}
			parts = append(parts, fmt.Sprintf("  - %s (%.0f%% done, %s, status: %s)",
				g.Plan.Content, g.Progress*100, daysStr, g.Status))
		}
	}

	if result.Continuity != nil && result.Continuity.WasInterrupted {
		parts = append(parts, "SESSION CONTINUITY:")
		parts = append(parts, fmt.Sprintf("  - %s (last active %s ago)",
			result.Continuity.ResumeSuggestion,
			result.Continuity.SessionAge.Round(time.Minute)))
	}

	if result.Sentiment != nil {
		parts = append(parts, fmt.Sprintf("SENTIMENT TREND: %s (recent: %.2f, previous: %.2f, delta: %.2f)",
			result.Sentiment.Direction, result.Sentiment.RecentAvg, result.Sentiment.PreviousAvg, result.Sentiment.Delta))
		for _, m := range result.Sentiment.Notable {
			parts = append(parts, fmt.Sprintf("  - [sentiment: %.2f] %s", m.Sentiment, m.Content))
		}
	}

	if len(result.Relationships) > 0 {
		parts = append(parts, fmt.Sprintf("RELATIONSHIP ALERTS (%d):", len(result.Relationships)))
		for _, r := range result.Relationships {
			parts = append(parts, fmt.Sprintf("  - %s: silent for %d days [%s]",
				r.Entity.CanonicalName, r.DaysSilent, r.Urgency))
		}
	}

	if len(result.KnowledgeGaps) > 0 {
		parts = append(parts, fmt.Sprintf("KNOWLEDGE GAPS (%d):", len(result.KnowledgeGaps)))
		for _, g := range result.KnowledgeGaps {
			parts = append(parts, fmt.Sprintf("  - %s", g.Question))
		}
	}

	if len(result.Patterns) > 0 {
		parts = append(parts, fmt.Sprintf("BEHAVIORAL PATTERNS (%d):", len(result.Patterns)))
		for _, p := range result.Patterns {
			parts = append(parts, fmt.Sprintf("  - %s (confidence: %.0f%%)", p.Description, p.Confidence*100))
		}
	}

	return strings.Join(parts, "\n")
}
